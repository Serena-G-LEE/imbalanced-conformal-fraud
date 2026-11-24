"""Unit tests for stage-3 class-conditional adaptive conformal inference (ACI).

The adaptive scheme has three properties that have to be pinned down, because
each of them is easy to break without any test failing on the other two:

* **Causality.**  ``C_t`` may only use labels up to ``t - 1``.  A prediction set
  that peeks at its own label would look perfect and be worthless in production.
* **Class-conditional update.**  Only the level of the class that was just
  observed may move, so the majority class cannot wash out the minority class'
  correction.
* **Recovery under drift.**  On a stream whose score distribution shifts away
  from the calibration window, static Mondrian CP under-covers fraud while the
  adaptive scheme closes most of the gap to ``1 - alpha``.

Runs with the standard library plus ``numpy`` only, except the runner
integration test at the bottom which drives ``run_experiment.main`` with mocked
model and loader and therefore needs no booster either.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal import (  # noqa: E402
    ADAPTIVE_ALPHA_CEIL,
    ADAPTIVE_ALPHA_FLOOR,
    SET_BOTH,
    SET_EMPTY,
    SET_FRAUD,
    SET_NORMAL,
    AdaptiveMondrianCP,
    MondrianCP,
    codes_to_membership,
    conformal_quantile,
    predict_adaptive_mondrian,
)
from src.metrics import stream_half_report  # noqa: E402


def drifting_stream(
    n_stream: int = 20_000,
    fraud_rate: float = 0.03,
    n_cal: int = 20_000,
    shift: float = 0.6,
    seed: int = 0,
):
    """Calibration block plus a stream whose fraud scores decay over time.

    The legitimate score distribution is the same in both blocks; the fraud
    scores are multiplied by ``1 - shift * t / T`` inside the stream, so late
    frauds look more like legitimate transactions than anything the calibration
    window saw.  That is a stylised version of the IEEE-CIS drift: the model
    keeps ranking, but the score scale of the minority class moves away from
    the calibration window, so a threshold calibrated once stops covering it.
    """
    rng = np.random.default_rng(seed)

    cal_labels = (rng.random(n_cal) < fraud_rate).astype(int)
    cal_prob = np.empty(n_cal, dtype=float)
    cal_prob[cal_labels == 0] = rng.beta(0.4, 20.0, size=int((cal_labels == 0).sum()))
    cal_prob[cal_labels == 1] = rng.beta(2.0, 3.0, size=int((cal_labels == 1).sum()))

    labels = (rng.random(n_stream) < fraud_rate).astype(int)
    prob = np.empty(n_stream, dtype=float)
    prob[labels == 0] = rng.beta(0.4, 20.0, size=int((labels == 0).sum()))
    position = np.linspace(0.0, 1.0, n_stream)[labels == 1]
    base = rng.beta(2.0, 3.0, size=int((labels == 1).sum()))
    prob[labels == 1] = base * (1.0 - shift * position)

    return (
        np.clip(cal_prob, 1e-6, 1.0 - 1e-6),
        cal_labels,
        np.clip(prob, 1e-6, 1.0 - 1e-6),
        labels,
    )


def class_coverage(codes, labels):
    membership = codes_to_membership(codes)
    covered = membership[np.arange(len(labels)), labels]
    return {
        "normal": float(covered[labels == 0].mean()),
        "fraud": float(covered[labels == 1].mean()),
        "marginal": float(covered.mean()),
    }


class TestAdaptiveInterface(unittest.TestCase):
    """Shape, validity, determinism and the frozen control path."""

    @classmethod
    def setUpClass(cls):
        cls.prob_cal, cls.y_cal, cls.prob_stream, cls.y_stream = drifting_stream(
            n_stream=2_000, n_cal=4_000, seed=3
        )

    def test_returns_one_valid_code_per_transaction(self):
        cp = AdaptiveMondrianCP(alpha=0.05, gamma=0.01).calibrate(self.prob_cal, self.y_cal)
        result = cp.predict_stream(self.prob_stream, self.y_stream)
        self.assertEqual(result.codes.shape, (self.prob_stream.size,))
        self.assertTrue(
            np.all(np.isin(result.codes, [SET_NORMAL, SET_FRAUD, SET_BOTH, SET_EMPTY]))
        )
        self.assertEqual(result.alpha_levels.shape, (self.prob_stream.size, 2))
        self.assertEqual(result.alpha_final.shape, (2,))
        self.assertEqual(result.errors.shape, (self.prob_stream.size,))

    def test_codes_match_the_levels_used_at_each_step(self):
        # Every emitted set has to be exactly the set implied by the level in
        # the history at that step: no look-ahead, no stale threshold.
        cp = AdaptiveMondrianCP(alpha=0.05, gamma=0.01).calibrate(self.prob_cal, self.y_cal)
        result = cp.predict_stream(self.prob_stream, self.y_stream)
        calibration_normal = cp.scores_[0]
        calibration_fraud = cp.scores_[1]
        for step in range(0, self.prob_stream.size, 97):
            q_normal = conformal_quantile(calibration_normal, result.alpha_levels[step, 0])
            q_fraud = conformal_quantile(calibration_fraud, result.alpha_levels[step, 1])
            prob = float(self.prob_stream[step])
            in_normal = prob <= q_normal
            in_fraud = (1.0 - prob) <= q_fraud
            expected = (
                SET_BOTH if in_normal and in_fraud
                else SET_NORMAL if in_normal
                else SET_FRAUD if in_fraud
                else SET_EMPTY
            )
            self.assertEqual(int(result.codes[step]), expected, msg=f"step={step}")

    def test_is_deterministic(self):
        first = predict_adaptive_mondrian(
            self.prob_cal, self.y_cal, self.prob_stream, self.y_stream, alpha=0.05, gamma=0.01
        )
        second = predict_adaptive_mondrian(
            self.prob_cal, self.y_cal, self.prob_stream, self.y_stream, alpha=0.05, gamma=0.01
        )
        np.testing.assert_array_equal(first.codes, second.codes)
        np.testing.assert_array_equal(first.alpha_levels, second.alpha_levels)

    def test_frozen_updates_reproduce_static_mondrian(self):
        # With update=False the loop must be exactly the static scheme: this
        # pins the repeated quantile lookup to conformal_quantile and proves
        # that no adaptation happens on any other code path.
        for alpha in (0.01, 0.05, 0.3):
            static = MondrianCP(alpha=alpha).calibrate(self.prob_cal, self.y_cal)
            frozen = AdaptiveMondrianCP(alpha=alpha, gamma=0.01).calibrate(
                self.prob_cal, self.y_cal
            )
            np.testing.assert_array_equal(
                static.predict_set_codes(self.prob_stream),
                frozen.predict_stream(self.prob_stream, self.y_stream, update=False).codes,
                err_msg=f"alpha={alpha}",
            )

    def test_repeated_quantile_matches_the_reference_quantile(self):
        cp = AdaptiveMondrianCP(alpha=0.05).calibrate(self.prob_cal, self.y_cal)
        source = cp.quantiles_[1]
        scores = cp.scores_[1]
        for alpha in (0.2, 0.1, 0.05, 0.0499, 0.01):
            self.assertAlmostEqual(
                source.at(alpha), conformal_quantile(scores, alpha), places=12
            )

    def test_describe_reports_the_adaptive_state(self):
        cp = AdaptiveMondrianCP(alpha=0.05, gamma=0.02).calibrate(self.prob_cal, self.y_cal)
        before = cp.describe()
        self.assertEqual(before["method"], "adaptive_mondrian")
        self.assertEqual(before["aci_gamma"], 0.02)
        self.assertAlmostEqual(before["alpha_final_normal"], 0.05)
        self.assertAlmostEqual(before["alpha_final_fraud"], 0.05)

        cp.predict_stream(self.prob_stream, self.y_stream)
        after = cp.describe()
        self.assertAlmostEqual(
            after["alpha_final_fraud"], cp.alpha_levels_[1], places=12
        )
        self.assertNotEqual(after["alpha_final_fraud"], 0.05)

    def test_describe_before_and_after_prediction_is_a_csv_shape(self):
        cp = AdaptiveMondrianCP(alpha=0.05).calibrate(self.prob_cal, self.y_cal)
        description = cp.describe()
        for key in ("q_normal", "q_fraud", "n_cal_fraud", "aci_gamma"):
            self.assertIn(key, description)


class TestCausality(unittest.TestCase):
    """The label of transaction t enters only after C_t has been formed."""

    @classmethod
    def setUpClass(cls):
        cls.prob_cal, cls.y_cal, cls.prob_stream, cls.y_stream = drifting_stream(
            n_stream=1_500, n_cal=3_000, seed=11
        )

    def _run(self, labels, prob=None):
        cp = AdaptiveMondrianCP(alpha=0.05, gamma=0.01).calibrate(self.prob_cal, self.y_cal)
        return cp.predict_stream(self.prob_stream if prob is None else prob, labels)

    def test_future_labels_do_not_change_earlier_sets(self):
        baseline = self._run(self.y_stream)
        for cut in (1, 7, 400, self.y_stream.size - 1):
            flipped = self.y_stream.copy()
            flipped[cut:] = 1 - flipped[cut:]
            perturbed = self._run(flipped)
            np.testing.assert_array_equal(
                baseline.codes[:cut], perturbed.codes[:cut], err_msg=f"cut={cut}"
            )
            np.testing.assert_array_equal(
                baseline.alpha_levels[:cut], perturbed.alpha_levels[:cut], err_msg=f"cut={cut}"
            )

    def test_prefix_run_matches_the_stream_run(self):
        # Truncating the stream is the same as ending the deployment early:
        # every earlier decision must be identical.
        full = self._run(self.y_stream)
        for cut in (5, 250, 1_000):
            prefix = self._run(self.y_stream[:cut], self.prob_stream[:cut])
            np.testing.assert_array_equal(full.codes[:cut], prefix.codes, err_msg=f"cut={cut}")

    def test_single_label_flip_changes_the_next_step_only(self):
        baseline = self._run(self.y_stream)
        cut = 100
        flipped = self.y_stream.copy()
        flipped[cut] = 1 - flipped[cut]
        perturbed = self._run(flipped)
        np.testing.assert_array_equal(baseline.codes[: cut + 1], perturbed.codes[: cut + 1])
        np.testing.assert_array_equal(baseline.alpha_levels[: cut + 1], perturbed.alpha_levels[: cut + 1])
        self.assertFalse(
            np.array_equal(baseline.alpha_levels[cut + 1], perturbed.alpha_levels[cut + 1])
        )


class TestClassConditionalUpdate(unittest.TestCase):
    """Only the level of the observed class moves, and it moves by the ACI rule."""

    def test_levels_follow_the_manual_aci_recursion(self):
        prob_cal, y_cal, prob, labels = drifting_stream(n_stream=400, n_cal=2_000, seed=5)
        alpha, gamma = 0.05, 0.02
        cp = AdaptiveMondrianCP(alpha=alpha, gamma=gamma).calibrate(prob_cal, y_cal)
        result = cp.predict_stream(prob, labels)

        expected = np.full(2, alpha)
        for t in range(prob.size):
            np.testing.assert_allclose(result.alpha_levels[t], expected, atol=1e-12)
            error = result.errors[t]
            label = int(labels[t])
            expected[label] = float(
                np.clip(expected[label] + gamma * (alpha - error), ADAPTIVE_ALPHA_FLOOR, ADAPTIVE_ALPHA_CEIL)
            )
        np.testing.assert_allclose(result.alpha_final, expected, atol=1e-12)

    def test_only_the_observed_class_level_moves(self):
        prob_cal, y_cal, prob, labels = drifting_stream(n_stream=300, n_cal=2_000, seed=8)
        cp = AdaptiveMondrianCP(alpha=0.05, gamma=0.05).calibrate(prob_cal, y_cal)
        result = cp.predict_stream(prob, labels)
        for t in range(1, prob.size):
            previous = result.alpha_levels[t - 1]
            current = result.alpha_levels[t]
            observed = int(labels[t - 1])
            untouched = 1 - observed
            self.assertAlmostEqual(current[untouched], previous[untouched], places=12)

    def test_levels_stay_inside_the_open_unit_interval(self):
        # An adversarial stream where fraud is always missed pushes the fraud
        # level down to the floor, where the quantile includes the class for
        # every transaction; the level then climbs back.  Both ends are
        # legitimate ACI states and neither may leave the interval.
        prob_cal = np.concatenate([np.linspace(0.0, 0.2, 500), np.linspace(0.3, 1.0, 500)])
        y_cal = np.concatenate([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
        stream_prob = np.concatenate([np.zeros(600), np.zeros(600)])
        stream_labels = np.concatenate([np.zeros(600, dtype=int), np.ones(600, dtype=int)])

        result = predict_adaptive_mondrian(
            prob_cal, y_cal, stream_prob, stream_labels, alpha=0.05, gamma=0.1
        )
        levels = result.alpha_levels
        self.assertTrue(np.all(levels >= ADAPTIVE_ALPHA_FLOOR))
        self.assertTrue(np.all(levels <= ADAPTIVE_ALPHA_CEIL))
        # the missed half pushes the fraud level to its floor ...
        self.assertLessEqual(float(levels[600:, 1].min()), ADAPTIVE_ALPHA_FLOOR + 1e-12)
        # ... where the fraud label is included for every transaction, so no
        # transaction is auto-declined on that half ...
        self.assertFalse(np.any(result.codes[600:] == SET_FRAUD))
        # ... and the label feedback moves the level back up again.
        self.assertGreater(float(result.alpha_final[1]), ADAPTIVE_ALPHA_FLOOR)


class TestDriftRecovery(unittest.TestCase):
    """The result stage 3 exists for: ACI buys back the coverage drift costs."""

    alpha = 0.05
    gamma = 0.01

    def test_adaptive_recovers_fraud_coverage_static_mondrian_loses(self):
        for seed in (0, 1, 2):
            prob_cal, y_cal, prob, labels = drifting_stream(seed=seed)
            static = MondrianCP(alpha=self.alpha).calibrate(prob_cal, y_cal)
            static_coverage = class_coverage(static.predict_set_codes(prob), labels)
            adaptive = predict_adaptive_mondrian(
                prob_cal,
                y_cal,
                prob,
                labels,
                alpha=self.alpha,
                gamma=self.gamma,
            )
            adaptive_coverage = class_coverage(adaptive.codes, labels)

            self.assertLess(
                static_coverage["fraud"], 0.93, msg=f"seed={seed} static should under-cover"
            )
            self.assertGreaterEqual(
                adaptive_coverage["fraud"], 0.93, msg=f"seed={seed} adaptive should recover"
            )
            self.assertGreater(
                adaptive_coverage["fraud"],
                static_coverage["fraud"] + 0.01,
                msg=f"seed={seed}",
            )

    def test_late_half_gap_is_where_the_recovery_shows(self):
        prob_cal, y_cal, prob, labels = drifting_stream(seed=4)
        static = MondrianCP(alpha=self.alpha).calibrate(prob_cal, y_cal)
        adaptive = predict_adaptive_mondrian(
            prob_cal, y_cal, prob, labels, alpha=self.alpha, gamma=self.gamma
        )
        static_half = stream_half_report(static.predict_set_codes(prob), labels)
        adaptive_half = stream_half_report(adaptive.codes, labels)

        self.assertGreater(
            static_half["coverage_fraud_first_half"],
            static_half["coverage_fraud_second_half"] + 0.05,
            msg="static coverage should decay into the newer half",
        )
        self.assertGreater(
            adaptive_half["coverage_fraud_second_half"],
            static_half["coverage_fraud_second_half"] + 0.03,
        )

    def test_stationary_legitimate_class_is_not_widened_unnecessarily(self):
        # The stream only drifts inside the fraud class, so the legitimate
        # level may wander a little but must stay near alpha: ACI must not pay
        # for fraud coverage with a review queue it does not need.
        prob_cal, y_cal, prob, labels = drifting_stream(seed=6)
        result = predict_adaptive_mondrian(
            prob_cal, y_cal, prob, labels, alpha=self.alpha, gamma=self.gamma
        )
        self.assertAlmostEqual(result.alpha_final[0], self.alpha, delta=0.02)
        coverage = class_coverage(result.codes, labels)
        self.assertAlmostEqual(coverage["normal"], 1.0 - self.alpha, delta=0.02)


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.prob_cal = np.linspace(0.0, 1.0, 200)
        self.y_cal = np.array([0] * 190 + [1] * 10, dtype=int)

    def test_gamma_must_lie_in_the_open_unit_interval(self):
        for gamma in (0.0, -0.01, 1.0, 1.5, float("nan")):
            with self.assertRaises(ValueError, msg=f"gamma={gamma}"):
                AdaptiveMondrianCP(alpha=0.05, gamma=gamma)

    def test_alpha_validation_is_inherited(self):
        with self.assertRaises(ValueError):
            AdaptiveMondrianCP(alpha=0.0)
        with self.assertRaises(ValueError):
            AdaptiveMondrianCP(alpha=1.0)

    def test_predict_without_labels_raises(self):
        cp = AdaptiveMondrianCP(alpha=0.05).calibrate(self.prob_cal, self.y_cal)
        with self.assertRaises(ValueError):
            cp.predict_set_codes(np.linspace(0.0, 1.0, 10))

    def test_predict_before_calibration_raises(self):
        with self.assertRaises(RuntimeError):
            AdaptiveMondrianCP(alpha=0.05).predict_stream([0.1, 0.2], [0, 1])

    def test_length_mismatch_raises(self):
        cp = AdaptiveMondrianCP(alpha=0.05).calibrate(self.prob_cal, self.y_cal)
        with self.assertRaises(ValueError):
            cp.predict_stream([0.1, 0.2, 0.3], [0, 1])

    def test_invalid_labels_and_probabilities_raise(self):
        cp = AdaptiveMondrianCP(alpha=0.05).calibrate(self.prob_cal, self.y_cal)
        with self.assertRaises(ValueError):
            cp.predict_stream([0.1, 0.2], [0, 2])
        with self.assertRaises(ValueError):
            cp.predict_stream([0.1, 1.2], [0, 1])


class TestStreamHalfReport(unittest.TestCase):
    def test_halves_are_chronological_and_hand_computable(self):
        # 4 transactions: {0} for a fraud is a miss, {0,1} is covered.
        codes = np.array([SET_NORMAL, SET_BOTH, SET_NORMAL, SET_BOTH])
        labels = np.array([1, 1, 0, 1])
        report = stream_half_report(codes, labels)
        self.assertAlmostEqual(report["coverage_fraud_first_half"], 0.5)
        self.assertAlmostEqual(report["coverage_fraud_second_half"], 1.0)
        self.assertAlmostEqual(report["triage_rate_first_half"], 0.5)
        self.assertAlmostEqual(report["triage_rate_second_half"], 0.5)

    def test_half_without_fraud_reports_nan(self):
        codes = np.array([SET_NORMAL, SET_NORMAL, SET_BOTH, SET_BOTH])
        labels = np.array([0, 0, 1, 1])
        report = stream_half_report(codes, labels)
        self.assertTrue(np.isnan(report["coverage_fraud_first_half"]))
        self.assertAlmostEqual(report["coverage_fraud_second_half"], 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            stream_half_report([SET_NORMAL], [0, 1])


CONFIG_TEMPLATE = """\
dataset: ulb
data:
  path: data/ulb/creditcard.csv
  splits: {train: 0.6, calibration: 0.2, test: 0.2}
models:
  - name: xgboost
    calibrator: none
    variant: unweighted_raw
experiment:
  seeds: [42]
  alpha_grid: [0.05]
  methods: [standard, mondrian, adaptive_mondrian]
  aci_gamma: 0.02
  output_dir: results
"""


def _fake_dataset(name: str):
    import run_experiment  # noqa: F401  (imported for the side effect of the path)
    from src.data import SplitDataset

    rng = np.random.default_rng(7)
    n = 120

    def block(n_rows: int):
        X = rng.normal(size=(n_rows, 3)).astype(np.float32)
        y = np.zeros(n_rows, dtype=np.int64)
        y[::4] = 1  # fraud spread through the block, so both halves have some
        return X, y

    X_train, y_train = block(n)
    X_cal, y_cal = block(n)
    X_test, y_test = block(n)
    return SplitDataset(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        feature_names=["f0", "f1", "f2"],
    )


def _fake_prediction(name, X_train, y_train, X_cal, X_test, **kwargs):
    from src.models import ModelPrediction

    rng = np.random.default_rng(11)
    prob_cal = rng.uniform(size=len(X_cal))
    prob_test = rng.uniform(size=len(X_test))
    return ModelPrediction(
        model_name=name,
        prob_cal=prob_cal,
        prob_test=prob_test,
        raw_prob_cal=prob_cal.copy(),
        raw_prob_test=prob_test.copy(),
        calibrator=str(kwargs.get("calibrator") or "none"),
        variant=str(kwargs.get("variant") or ""),
    )


class TestRunnerIntegration(unittest.TestCase):
    """The adaptive scheme has to survive the CLI path, not just the class API."""

    def test_stage3_run_writes_adaptive_rows(self):
        import run_experiment
        from src.metrics import METRIC_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "ulb_stage3_stub.yaml"
            config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
            output_dir = tmp_path / "results"
            argv = [
                "--dataset", "ulb",
                "--config", str(config_path),
                "--output-dir", str(output_dir),
                "--no-figures",
            ]
            with mock.patch.object(
                run_experiment, "load_dataset", side_effect=lambda *a, **k: _fake_dataset("ulb")
            ), mock.patch.object(run_experiment, "train_and_predict", side_effect=_fake_prediction):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_experiment.main(argv)

            summary_path = output_dir / "ulb" / "ulb_summary_metrics.csv"
            self.assertTrue(summary_path.is_file())
            text = summary_path.read_text(encoding="utf-8")
            self.assertIn("adaptive_mondrian", text)
            per_seed = (output_dir / "ulb" / "ulb_per_seed_metrics.csv").read_text(encoding="utf-8")
            header = per_seed.splitlines()[0].split(",")
            for column in (
                "aci_gamma",
                "alpha_final_normal",
                "alpha_final_fraud",
                "coverage_fraud_first_half",
                "coverage_fraud_second_half",
            ):
                self.assertIn(column, header)
                self.assertIn(column, METRIC_COLUMNS)

            # One row per (method, alpha) and the adaptive row carries gamma.
            import csv

            with (output_dir / "ulb" / "ulb_per_seed_metrics.csv").open(encoding="utf-8") as handle:
                per_seed_rows = list(csv.DictReader(handle))
            gammas = {
                row["method"]: float(row["aci_gamma"])
                for row in per_seed_rows
                if row["aci_gamma"] not in {"", "nan"}
            }
            self.assertEqual(gammas, {"adaptive_mondrian": 0.02})

            with (output_dir / "ulb" / "ulb_summary_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["method"] for row in rows}, {"standard", "mondrian", "adaptive_mondrian"})
            adaptive_rows = [row for row in rows if row["method"] == "adaptive_mondrian"]
            self.assertEqual(len(adaptive_rows), 1)
            final_level = float(adaptive_rows[0]["alpha_final_fraud_mean"])
            self.assertTrue(0.0 < final_level < 1.0)


if __name__ == "__main__":
    unittest.main()
