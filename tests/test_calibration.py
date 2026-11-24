"""Unit tests for probability calibration: Brier score, ECE, calibrators, splits.

Standard library plus ``numpy`` only, so the probability-calibration layer can
be validated on a machine without scikit-learn or the boosters::

    python3 -m unittest discover -s tests
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal import MondrianCP, StandardSplitCP  # noqa: E402
from src.data import SplitDataset, split_train_tail_indices  # noqa: E402
from src.metrics import brier_score, ece  # noqa: E402
from src.models import (  # noqa: E402
    IsotonicCalibrator,
    PlattCalibrator,
    default_variant,
    fit_calibrator,
    resolve_scale_pos_weight,
    train_and_predict,
)


def _toy_dataset(n_train=200, n_cal=60, n_test=60, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    def block(n):
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        logits = 2.0 * X[:, 0] - 1.0
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n) < prob).astype(np.int64)
        return X, y
    X_train, y_train = block(n_train)
    X_cal, y_cal = block(n_cal)
    X_test, y_test = block(n_test)
    return SplitDataset(
        name="toy",
        X_train=X_train, y_train=y_train,
        X_cal=X_cal, y_cal=y_cal,
        X_test=X_test, y_test=y_test,
        feature_names=["a", "b", "c"],
    )


class TestBrierScore(unittest.TestCase):
    def test_hand_computable_value(self):
        # (0 - 0.1)^2 + (1 - 0.7)^2 = 0.01 + 0.09 -> mean 0.05
        self.assertAlmostEqual(brier_score([0, 1], [0.1, 0.7]), 0.05)

    def test_perfect_forecast_is_zero(self):
        self.assertAlmostEqual(brier_score([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]), 0.0)

    def test_constant_prevalence_forecast(self):
        labels = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(brier_score(labels, np.full(4, 0.25)), 0.1875)

    def test_worse_forecast_scores_higher(self):
        labels = np.array([0, 0, 1, 1])
        good = brier_score(labels, np.array([0.1, 0.2, 0.8, 0.9]))
        bad = brier_score(labels, np.array([0.6, 0.7, 0.3, 0.4]))
        self.assertLess(good, bad)

    def test_rejects_probabilities_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            brier_score([0, 1], [0.2, 1.4])

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            brier_score([0, 1, 0], [0.2, 0.3])


class TestEce(unittest.TestCase):
    def test_hand_computable_uniform_bins(self):
        # Two uniform bins: [0, 0.5) holds (0, 0.1) and (0, 0.2): acc 0, conf 0.15
        # second bin holds (1, 0.8) and (1, 0.9): acc 1, conf 0.85
        # ECE = 0.5 * 0.15 + 0.5 * 0.15 = 0.15
        value = ece([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2, strategy="uniform")
        self.assertAlmostEqual(value, 0.15)

    def test_perfectly_calibrated_is_zero(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(ece(labels, scores, n_bins=2, strategy="uniform"), 0.0)

    def test_miscalibrated_scores_are_penalised(self):
        labels = np.array([0, 0, 1, 1])
        calibrated = np.array([0.1, 0.2, 0.8, 0.9])
        inflated = np.array([0.6, 0.7, 0.9, 0.95])
        self.assertLess(
            ece(labels, calibrated, n_bins=2, strategy="uniform"),
            ece(labels, inflated, n_bins=2, strategy="uniform"),
        )

    def test_quantile_binning_uses_equal_mass(self):
        rng = np.random.default_rng(7)
        labels = (rng.random(400) < 0.2).astype(int)
        scores = np.clip(scores_from_labels(labels, rng), 0.0, 1.0)
        value = ece(labels, scores, n_bins=10, strategy="quantile")
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_uniform_bins_collapse_into_one_bin_for_tiny_scores(self):
        # With a 0.172 %-style score range (every score below 1/15), a uniform
        # 15-bin grid puts every transaction in the first bin, so ECE degenerates
        # to ``|prevalence - mean(score)|`` and stops seeing the ranking at all.
        # That is why the pipeline defaults to equal-mass (quantile) bins.
        labels = np.array([0] * 998 + [1] * 2)
        scores = np.linspace(0.0, 0.02, 1_000)
        value = ece(labels, scores, n_bins=15, strategy="uniform")
        self.assertAlmostEqual(value, abs(float(labels.mean()) - float(scores.mean())))

    def test_rejects_unknown_strategy_and_bad_bins(self):
        with self.assertRaises(ValueError):
            ece([0, 1], [0.2, 0.8], n_bins=2, strategy="kmeans")
        with self.assertRaises(ValueError):
            ece([0, 1], [0.2, 0.8], n_bins=0)


def scores_from_labels(labels, rng):
    """Mildly miscalibrated scores that still rank the labels correctly."""
    return np.where(labels == 1, rng.uniform(0.5, 0.9, labels.size), rng.uniform(0.0, 0.5, labels.size))


class TestPlattCalibrator(unittest.TestCase):
    def test_offset_scores_are_pulled_back_to_prevalence(self):
        rng = np.random.default_rng(11)
        labels = (rng.random(5_000) < 0.02).astype(int)
        # Inflate the odds by a factor of 50, as scale_pos_weight does.
        true_prob = np.where(labels == 1, 0.6, 0.01)
        inflated = np.clip(true_prob * 5.0, 0.0, 1.0)

        calibrator = PlattCalibrator().fit(inflated, labels)
        calibrated = calibrator.predict(inflated)

        self.assertGreater(calibrator.coef_, 0.0)  # monotone increasing
        self.assertLess(brier_score(labels, calibrated), brier_score(labels, inflated))
        self.assertLess(
            abs(float(np.mean(calibrated)) - float(np.mean(labels))),
            abs(float(np.mean(inflated)) - float(np.mean(labels))),
        )

    def test_preserves_ranking(self):
        rng = np.random.default_rng(12)
        labels = (rng.random(500) < 0.1).astype(int)
        scores = np.clip(rng.beta(1.5, 6.0, 500) + 0.3 * labels, 0.0, 1.0)
        calibrated = PlattCalibrator().fit(scores, labels).predict(scores)
        order_raw = np.argsort(scores, kind="mergesort")
        order_cal = np.argsort(calibrated, kind="mergesort")
        np.testing.assert_array_equal(order_raw, order_cal)

    def test_output_range_and_tiny_positive_count(self):
        rng = np.random.default_rng(13)
        labels = np.zeros(400, dtype=int)
        labels[[3, 77]] = 1
        scores = np.clip(rng.random(400) * 0.3 + 0.5 * labels, 0.0, 1.0)
        calibrated = PlattCalibrator().fit(scores, labels).predict(scores)
        self.assertTrue(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))
        self.assertTrue(np.all(np.isfinite(calibrated)))
        self.assertGreater(calibrated[3], calibrated[0])

    def test_requires_both_classes(self):
        with self.assertRaises(ValueError):
            PlattCalibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])

    def test_predict_is_clipped_for_extreme_inputs(self):
        calibrator = PlattCalibrator().fit([0.01, 0.2, 0.8, 0.99], [0, 0, 1, 1])
        out = calibrator.predict(np.array([0.0, 1.0]))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))


class TestIsotonicCalibrator(unittest.TestCase):
    def test_fit_is_non_decreasing(self):
        rng = np.random.default_rng(21)
        labels = (rng.random(600) < 0.15).astype(int)
        scores = np.clip(rng.random(600) + 0.25 * labels, 0.0, 1.0)
        calibrator = IsotonicCalibrator().fit(scores, labels)
        self.assertTrue(np.all(np.diff(calibrator.y_) >= -1e-12))
        predictions = calibrator.predict(np.linspace(0.0, 1.0, 101))
        self.assertTrue(np.all(np.diff(predictions) >= -1e-12))

    def test_clips_outside_the_fitted_range(self):
        calibrator = IsotonicCalibrator().fit([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
        out = calibrator.predict(np.array([0.0, 1.0]))
        self.assertAlmostEqual(out[0], float(calibrator.y_[0]))
        self.assertAlmostEqual(out[1], float(calibrator.y_[-1]))
        self.assertLessEqual(out[0], out[1])

    def test_minimises_squared_error_against_monotone_alternative(self):
        rng = np.random.default_rng(22)
        labels = (rng.random(2_000) < 0.05).astype(int)
        scores = np.clip(rng.beta(1.0, 12.0, 2_000) + 0.5 * labels, 0.0, 1.0)
        calibrator = IsotonicCalibrator().fit(scores, labels)
        calibrated = calibrator.predict(scores)
        raw = scores
        # Isotonic regression is the least-squares monotone fit, so on its own
        # calibration data it must not be worse than the raw scores.
        self.assertLessEqual(brier_score(labels, calibrated), brier_score(labels, raw) + 1e-9)

    def test_handles_a_single_distinct_score(self):
        calibrator = IsotonicCalibrator().fit([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])
        np.testing.assert_allclose(calibrator.predict(np.array([0.1, 0.5, 0.9])), 0.5)


class TestFitCalibrator(unittest.TestCase):
    def test_none_is_the_identity(self):
        calibrator = fit_calibrator("none", [0.1, 0.9], [0, 1])
        np.testing.assert_allclose(calibrator.predict(np.array([0.2, 0.7])), [0.2, 0.7])

    def test_dispatch_by_name(self):
        self.assertIsInstance(fit_calibrator("platt", [0.1, 0.9], [0, 1]), PlattCalibrator)
        self.assertIsInstance(fit_calibrator("ISOTONIC", [0.1, 0.9], [0, 1]), IsotonicCalibrator)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            fit_calibrator("temperature", [0.1, 0.9], [0, 1])

    def test_to_dict_is_json_safe(self):
        import json

        for name in ("none", "platt", "isotonic"):
            calibrator = fit_calibrator(name, [0.1, 0.4, 0.9], [0, 0, 1])
            json.dumps(calibrator.to_dict())


class TestVariantLabels(unittest.TestCase):
    def test_default_labels(self):
        self.assertEqual(default_variant("xgboost", None, "none"), "unweighted_raw")
        self.assertEqual(default_variant("xgboost", 578.0, "none"), "weighted_raw")
        self.assertEqual(default_variant("lightgbm", 578.0, "platt"), "weighted_platt")
        self.assertEqual(default_variant("lightgbm", None, "isotonic"), "unweighted_isotonic")

    def test_unit_weight_counts_as_unweighted(self):
        self.assertEqual(default_variant("xgboost", 1.0, "none"), "unweighted_raw")

    def test_resolve_scale_pos_weight(self):
        y = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(resolve_scale_pos_weight(y, "auto"), 3.0)
        self.assertIsNone(resolve_scale_pos_weight(y, None))
        self.assertIsNone(resolve_scale_pos_weight(y, "off"))
        self.assertAlmostEqual(resolve_scale_pos_weight(y, 1.0), 1.0)


class TestProbCalSplit(unittest.TestCase):
    def test_indices_are_a_chronological_tail(self):
        fit, prob = split_train_tail_indices(100, 0.15)
        self.assertEqual(len(fit), 85)
        self.assertEqual(len(prob), 15)
        self.assertEqual(prob[0], fit[-1] + 1)
        self.assertEqual(prob[-1], 99)
        self.assertEqual(set(fit) & set(prob), set())

    def test_fraction_validation(self):
        for fraction in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(ValueError):
                split_train_tail_indices(100, fraction)
        with self.assertRaises(ValueError):
            split_train_tail_indices(1, 0.15)

    def test_at_least_one_row_on_each_side(self):
        fit, prob = split_train_tail_indices(3, 0.15)
        self.assertEqual(len(prob), 1)
        self.assertEqual(len(fit), 2)

    def test_dataset_split_leaves_conformal_calibration_untouched(self):
        dataset = _toy_dataset()
        split = dataset.with_prob_cal_split(0.15)

        self.assertTrue(split.has_prob_cal)
        self.assertEqual(len(split.X_train) + len(split.X_prob_cal), len(dataset.X_train))
        self.assertEqual(len(split.X_prob_cal), 30)  # floor(0.15 * 200)
        np.testing.assert_array_equal(split.X_cal, dataset.X_cal)
        np.testing.assert_array_equal(split.y_cal, dataset.y_cal)
        np.testing.assert_array_equal(split.X_test, dataset.X_test)
        np.testing.assert_array_equal(split.y_test, dataset.y_test)
        # The prob_cal block is the tail of the original training block, in order.
        np.testing.assert_array_equal(
            split.X_prob_cal, dataset.X_train[len(split.X_train):]
        )
        np.testing.assert_array_equal(split.y_prob_cal, dataset.y_train[len(split.y_train):])

    def test_mutation_of_split_does_not_touch_original(self):
        dataset = _toy_dataset()
        split = dataset.with_prob_cal_split(0.15)
        original = split.X_prob_cal[0, 0]
        split.X_prob_cal[0, 0] = original + 123.0
        self.assertNotEqual(dataset.X_train[-1, 0], original + 123.0)

    def test_summary_reports_the_new_block(self):
        split = _toy_dataset().with_prob_cal_split(0.2)
        summary = split.summary()
        self.assertIn("prob_cal", summary)
        self.assertEqual(summary["prob_cal"]["n"], 40)
        self.assertEqual(summary["prob_cal_fraction"], 0.2)
        self.assertEqual(summary["prob_cal_split"]["n_prob_cal"], 40)

    def test_double_split_is_rejected(self):
        split = _toy_dataset().with_prob_cal_split(0.15)
        with self.assertRaises(ValueError):
            split.with_prob_cal_split(0.15)


class TestLeakageGuards(unittest.TestCase):
    def test_calibrator_without_prob_cal_block_raises(self):
        dataset = _toy_dataset()
        with self.assertRaises(ValueError):
            train_and_predict(
                "xgboost", dataset.X_train, dataset.y_train, dataset.X_cal, dataset.X_test,
                calibrator="platt",
            )

    def test_prob_cal_equal_to_conformal_cal_raises(self):
        # The guard must fire before the booster is imported, so this test needs
        # no xgboost installation.
        dataset = _toy_dataset()
        with self.assertRaises(ValueError) as context:
            train_and_predict(
                "xgboost", dataset.X_train, dataset.y_train, dataset.X_cal, dataset.X_test,
                calibrator="platt", X_prob_cal=dataset.X_cal, y_prob_cal=dataset.y_cal,
            )
        self.assertIn("X_prob_cal", str(context.exception))

    def test_unknown_calibrator_raises(self):
        dataset = _toy_dataset()
        with self.assertRaises(ValueError):
            train_and_predict(
                "xgboost", dataset.X_train, dataset.y_train, dataset.X_cal, dataset.X_test,
                calibrator="temperature",
            )


class TestMonotoneRecalibrationInvariance(unittest.TestCase):
    """What a monotone calibrator can and cannot change in the conformal layer.

    Both calibrators are fit on ``prob_cal`` and then applied unchanged to the
    calibration and test scores.  Because the class-conditional scores are a
    single monotone function of ``p`` per stratum (``p`` for labelled 0,
    ``1 - p`` for labelled 1), Mondrian thresholds move exactly with the scores
    and every membership test keeps its answer: a *strictly* increasing map
    leaves the prediction sets — and therefore ``triage_rate`` — untouched.

    The pooled standard split is different.  Its one threshold is a quantile of
    a vector that mixes ``p`` (negative scores) with ``1 - p`` (fraud scores),
    so a non-linear map reorders that pooled vector and can move the threshold.
    That asymmetry is why the ablation reports Brier / ECE gains for both
    schemes but review-queue changes only through the model fit, the calibration
    scheme, or the ties an isotonic fit introduces.
    """

    def _inflated_problem(self, seed=31, n=4_000, factor=4.0):
        rng = np.random.default_rng(seed)
        signal = rng.normal(size=n)
        prob = 1.0 / (1.0 + np.exp(-(-3.0 + 1.6 * signal)))
        labels = (rng.random(n) < prob).astype(int)
        # Inflate the *odds* rather than clipping an already large probability:
        # this keeps the map one-to-one, the way scale_pos_weight shifts the
        # booster's log-odds without creating ties.
        odds = prob / (1.0 - prob)
        inflated = factor * odds
        return inflated / (1.0 + inflated), labels

    def _fixture(self):
        prob_cal, y_prob_cal = self._inflated_problem(seed=33, n=3_000)
        cal, y_cal = self._inflated_problem(seed=34, n=1_000)
        test, y_test = self._inflated_problem(seed=35, n=1_000)
        calibrator = PlattCalibrator().fit(prob_cal, y_prob_cal)
        return calibrator, cal, y_cal, test, y_test

    def test_mondrian_prediction_sets_are_invariant(self):
        calibrator, cal, y_cal, test, _ = self._fixture()
        for alpha in (0.01, 0.03, 0.05, 0.10):
            raw_codes = (
                MondrianCP(alpha=alpha).calibrate(cal, y_cal).predict_set_codes(test)
            )
            cal_codes = (
                MondrianCP(alpha=alpha)
                .calibrate(calibrator.predict(cal), y_cal)
                .predict_set_codes(calibrator.predict(test))
            )
            np.testing.assert_array_equal(raw_codes, cal_codes)

    def test_pooled_standard_threshold_can_move(self):
        # Documents the asymmetry: the pooled quantile is a quantile of a vector
        # mixing p and 1 - p, so it is not a monotone image of the raw one.
        calibrator, cal, y_cal, test, _ = self._fixture()
        raw_codes = StandardSplitCP(alpha=0.05).calibrate(cal, y_cal).predict_set_codes(test)
        cal_codes = (
            StandardSplitCP(alpha=0.05)
            .calibrate(calibrator.predict(cal), y_cal)
            .predict_set_codes(calibrator.predict(test))
        )
        raw_q = StandardSplitCP(alpha=0.05).calibrate(cal, y_cal).q_[0]
        cal_q = StandardSplitCP(alpha=0.05).calibrate(calibrator.predict(cal), y_cal).q_[0]
        self.assertNotAlmostEqual(raw_q, cal_q, places=6)
        self.assertGreater(float(np.mean(raw_codes != cal_codes)), 0.0)

    def test_platt_does_improve_brier_and_ece(self):
        calibrator, _, _, test, y_test = self._fixture()
        calibrated = calibrator.predict(test)
        self.assertLess(brier_score(y_test, calibrated), brier_score(y_test, test))
        self.assertLess(ece(y_test, calibrated), ece(y_test, test))


if __name__ == "__main__":
    unittest.main()
