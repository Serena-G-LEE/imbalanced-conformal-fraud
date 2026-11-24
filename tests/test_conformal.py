"""Unit tests for split conformal calibration.

Runs with the standard library plus ``numpy`` only, so the conformal layer can
be validated on a machine without pandas, scikit-learn or the boosters::

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

from src.conformal import (  # noqa: E402
    SET_BOTH,
    SET_EMPTY,
    SET_FRAUD,
    SET_NORMAL,
    SET_NAMES,
    MondrianCP,
    StandardSplitCP,
    codes_from_membership,
    codes_to_membership,
    conformal_quantile,
    nonconformity_scores,
)


def synthetic_scores(n_samples: int = 40_000, fraud_rate: float = 0.003, seed: int = 0):
    """Probabilities with a realistic heavy-tailed overlap between the classes.

    The legitimate class is drawn from a left-skewed Beta, so a non-trivial tail
    of legitimate transactions receives moderate fraud scores.  That tail is
    what pulls the pooled calibration quantile up and breaks class-conditional
    coverage under a standard split.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < fraud_rate).astype(int)
    prob = np.empty(n_samples, dtype=float)
    n_negative = int(np.sum(labels == 0))
    n_positive = int(np.sum(labels == 1))
    prob[labels == 0] = rng.beta(0.6, 18.0, size=n_negative)
    prob[labels == 1] = rng.beta(2.0, 2.0, size=n_positive)
    return np.clip(prob, 1e-6, 1.0 - 1e-6), labels


class TestConformalQuantile(unittest.TestCase):
    """The finite-sample corrected quantile is the core of the guarantee."""

    def test_matches_manual_order_statistic(self):
        scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8])
        n = scores.size
        ordered = np.sort(scores)
        for alpha in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
            rank = math.ceil((n + 1) * (1 - alpha))
            expected = ordered[rank - 1] if rank <= n else 1.0
            self.assertAlmostEqual(conformal_quantile(scores, alpha), expected, places=12)

    def test_rank_formula_uses_ceiling_and_n_plus_one(self):
        scores = np.arange(1, 21, dtype=float) / 20.0  # 0.05 ... 1.0
        alpha = 0.1
        rank = math.ceil(21 * 0.9)  # 19
        self.assertAlmostEqual(conformal_quantile(scores, alpha), scores[rank - 1], places=12)

    def test_clipped_to_one_when_rank_exceeds_stratum_size(self):
        scores = np.array([0.2, 0.4])
        # ceil(3 * 0.9) = 3 > 2: the exact quantile is not available, clip to 1.
        self.assertEqual(conformal_quantile(scores, 0.1), 1.0)
        # ceil(3 * 0.5) = 2 <= 2: the largest score is usable.
        self.assertAlmostEqual(conformal_quantile(scores, 0.5), 0.4, places=12)

    def test_empty_stratum_returns_one(self):
        self.assertEqual(conformal_quantile(np.array([]), 0.05), 1.0)

    def test_monotone_decreasing_in_alpha(self):
        scores = np.linspace(0.0, 1.0, 101)
        previous = -1.0
        for alpha in (0.5, 0.25, 0.1, 0.05, 0.01):
            value = conformal_quantile(scores, alpha)
            self.assertGreaterEqual(value, previous)
            previous = value

    def test_rejects_invalid_alpha(self):
        scores = np.array([0.1, 0.2, 0.3])
        for alpha in (-0.1, 0.0, 1.0, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                conformal_quantile(scores, alpha)

    def test_rejects_non_finite_scores(self):
        with self.assertRaises(ValueError):
            conformal_quantile(np.array([0.1, np.nan]), 0.1)


class TestNonconformity(unittest.TestCase):
    def test_score_is_one_minus_probability_of_true_label(self):
        prob = np.array([0.0, 0.25, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1, 0])
        scores = nonconformity_scores(prob, labels)
        expected = np.array([0.0, 0.25, 0.5, 0.1, 1.0])
        np.testing.assert_allclose(scores, expected)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            nonconformity_scores([0.1, 0.2], [0])
        with self.assertRaises(ValueError):
            nonconformity_scores([0.1, 0.2], [0, 2])
        with self.assertRaises(ValueError):
            nonconformity_scores([0.1, 1.2], [0, 1])


class TestPredictionSetLogic(unittest.TestCase):
    """Every one of the four operational outcomes must be reachable and stable."""

    def test_codes_from_membership_four_cases(self):
        membership = np.array(
            [
                [True, False],   # confident legitimate
                [False, True],   # confident fraud
                [True, True],    # manual review
                [False, False],  # empty set
            ]
        )
        codes = codes_from_membership(membership)
        self.assertEqual(codes.tolist(), [SET_NORMAL, SET_FRAUD, SET_BOTH, SET_EMPTY])
        np.testing.assert_array_equal(codes_to_membership(codes), membership)
        self.assertEqual(
            [SET_NAMES[code] for code in codes], ["{0}", "{1}", "{0,1}", "{}"]
        )

    def test_roundtrip_is_identity(self):
        rng = np.random.default_rng(7)
        membership = rng.random((500, 2)) > 0.5
        np.testing.assert_array_equal(
            codes_to_membership(codes_from_membership(membership)), membership
        )

    def test_unknown_code_rejected(self):
        with self.assertRaises(ValueError):
            codes_to_membership(np.array([0, 1, 4]))

    def test_mondrian_thresholds_are_hand_computable(self):
        # alpha = 0.4, class 0 has 3 calibration points, class 1 has 2.
        # q_0 = 3rd smallest of [0.1, 0.2, 0.7] = 0.7
        # q_1 = 2nd smallest of [0.25, 0.05] = 0.25
        cp = MondrianCP(alpha=0.4).calibrate(
            prob_positive=np.array([0.1, 0.2, 0.7, 0.75, 0.95]),
            labels=np.array([0, 0, 0, 1, 1]),
        )
        np.testing.assert_allclose(cp.q_, [0.7, 0.25], atol=1e-12)
        codes = cp.predict_set_codes(np.array([0.2, 0.9, 0.72]))
        self.assertEqual(codes.tolist(), [SET_NORMAL, SET_FRAUD, SET_EMPTY])
        self.assertEqual(cp.set_sizes(np.array([0.2, 0.9, 0.72])).tolist(), [1, 1, 0])

    def test_ambiguous_set_is_reachable(self):
        # q_0 = 0.2 (2nd of [0.1, 0.2]) and q_1 = 0.7 (2nd of [0.6, 0.7]).
        # p = 0.3 satisfies p <= 0.2? no -> but score for class 0 is p = 0.3.
        # Use p = 0.15: score_0 = 0.15 <= 0.2 and score_1 = 0.85 <= 0.7? no.
        # p = 0.3 is chosen so that score_1 = 0.7 <= q_1 and score_0 = 0.3 > q_0.
        cp = MondrianCP(alpha=0.5).calibrate(
            prob_positive=np.array([0.1, 0.3, 0.3, 0.4]),
            labels=np.array([0, 0, 1, 1]),
        )
        # q_0 = 2nd of [0.1, 0.3] = 0.3 ; q_1 = 2nd of [0.7, 0.6] = 0.7
        np.testing.assert_allclose(cp.q_, [0.3, 0.7], atol=1e-12)
        codes = cp.predict_set_codes(np.array([0.3, 0.1, 0.95]))
        # p = 0.3 -> 0 in set (0.3 <= 0.3) and 1 in set (0.7 <= 0.7) -> review
        # p = 0.1 -> 0 in set only
        # p = 0.95 -> 1 in set only
        self.assertEqual(codes.tolist(), [SET_BOTH, SET_NORMAL, SET_FRAUD])

    def test_standard_cp_pools_both_classes(self):
        # Pooled scores: [0.1, 0.2, 0.6, 0.1]; 3rd smallest = 0.2 for both labels.
        cp = StandardSplitCP(alpha=0.5).calibrate(
            prob_positive=np.array([0.1, 0.2, 0.6, 0.9]),
            labels=np.array([0, 0, 0, 1]),
        )
        self.assertAlmostEqual(cp.q_[0], 0.2, places=12)
        self.assertAlmostEqual(cp.q_[1], 0.2, places=12)
        codes = cp.predict_set_codes(np.array([0.1, 0.25, 0.85, 0.5]))
        self.assertEqual(
            codes.tolist(), [SET_NORMAL, SET_EMPTY, SET_FRAUD, SET_EMPTY]
        )

    def test_probability_is_clipped_region_boundary_inclusive(self):
        cp = MondrianCP(alpha=0.5).calibrate(
            prob_positive=np.array([0.1, 0.3, 0.3, 0.4]),
            labels=np.array([0, 0, 1, 1]),
        )
        # score exactly equal to the threshold must be inside the set
        codes = cp.predict_set_codes(np.array([0.3]))
        self.assertEqual(int(codes[0]), SET_BOTH)


class TestClassConditionalCoverage(unittest.TestCase):
    """The motivating phenomenon: pooled CP loses the fraud label."""

    alpha = 0.05

    @classmethod
    def setUpClass(cls):
        prob, labels = synthetic_scores()
        half = prob.size // 2
        cls.prob_cal, cls.y_cal = prob[:half], labels[:half]
        cls.prob_test, cls.y_test = prob[half:], labels[half:]

    def _coverage(self, cp):
        membership = codes_to_membership(cp.predict_set_codes(self.prob_test))
        covered = membership[np.arange(self.prob_test.size), self.y_test]
        return {
            "marginal": float(covered.mean()),
            "normal": float(covered[self.y_test == 0].mean()),
            "fraud": float(covered[self.y_test == 1].mean()),
        }

    def test_standard_cp_keeps_marginal_coverage(self):
        coverage = self._coverage(StandardSplitCP(alpha=self.alpha).calibrate(self.prob_cal, self.y_cal))
        self.assertGreaterEqual(coverage["marginal"], 1 - self.alpha - 0.02)

    def test_standard_cp_fraud_coverage_collapses(self):
        coverage = self._coverage(StandardSplitCP(alpha=self.alpha).calibrate(self.prob_cal, self.y_cal))
        self.assertLess(coverage["fraud"], 0.30)
        self.assertLess(coverage["fraud"], coverage["normal"] - 0.5)

    def test_mondrian_cp_restores_class_conditional_coverage(self):
        coverage = self._coverage(MondrianCP(alpha=self.alpha).calibrate(self.prob_cal, self.y_cal))
        self.assertGreaterEqual(coverage["fraud"], 1 - self.alpha - 0.05)
        self.assertGreaterEqual(coverage["normal"], 1 - self.alpha - 0.05)
        self.assertGreaterEqual(coverage["marginal"], 1 - self.alpha - 0.02)

    def test_mondrian_beats_standard_on_fraud_coverage_across_seeds(self):
        for seed in (1, 2, 3):
            prob, labels = synthetic_scores(seed=seed)
            half = prob.size // 2
            standard = self._coverage(
                StandardSplitCP(alpha=self.alpha).calibrate(prob[:half], labels[:half])
            )
            mondrian = self._coverage(
                MondrianCP(alpha=self.alpha).calibrate(prob[:half], labels[:half])
            )
            self.assertGreaterEqual(mondrian["fraud"], 0.90, msg=f"seed={seed}")
            self.assertGreater(mondrian["fraud"], standard["fraud"] + 0.5, msg=f"seed={seed}")

    def test_triage_rate_tracks_alpha(self):
        rates = []
        for alpha in (0.01, 0.05, 0.10):
            cp = MondrianCP(alpha=alpha).calibrate(self.prob_cal, self.y_cal)
            codes = cp.predict_set_codes(self.prob_test)
            rates.append(float(np.mean(codes == SET_BOTH)))
        # A looser guarantee must never cost more review work.
        self.assertGreaterEqual(rates[0], rates[1] - 1e-9)
        self.assertGreaterEqual(rates[1], rates[2] - 1e-9)


class TestCalibrationEdgeCases(unittest.TestCase):
    def test_empty_fraud_stratum_never_excludes_fraud(self):
        prob = np.array([0.01, 0.02, 0.03, 0.04])
        labels = np.array([0, 0, 0, 0])  # no fraud in the calibration window
        cp = MondrianCP(alpha=0.05).calibrate(prob, labels)
        self.assertEqual(cp.q_[1], 1.0)
        self.assertEqual(cp.stratum_sizes_[1], 0)
        codes = cp.predict_set_codes(np.array([0.001, 0.5, 0.999]))
        self.assertTrue(np.all(np.isin(codes, [SET_FRAUD, SET_BOTH])))
        self.assertAlmostEqual(cp.describe()["min_prob_positive_for_fraud"], 0.0)

    def test_predict_before_calibration_raises(self):
        with self.assertRaises(RuntimeError):
            MondrianCP(alpha=0.1).predict_set_codes([0.1, 0.2])
        with self.assertRaises(RuntimeError):
            StandardSplitCP(alpha=0.1).describe()

    def test_empty_calibration_set_raises(self):
        with self.assertRaises(ValueError):
            StandardSplitCP(alpha=0.1).calibrate([], [])

    def test_describe_reports_strata_and_thresholds(self):
        cp = MondrianCP(alpha=0.1).calibrate(
            np.array([0.1, 0.2, 0.6, 0.9]), np.array([0, 0, 0, 1])
        )
        description = cp.describe()
        self.assertEqual(description["n_cal_normal"], 3)
        self.assertEqual(description["n_cal_fraud"], 1)
        self.assertAlmostEqual(description["min_prob_positive_for_fraud"], 1.0 - cp.q_[1])
        self.assertEqual(description["method"], "mondrian")


if __name__ == "__main__":
    unittest.main()
