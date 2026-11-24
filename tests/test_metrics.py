"""Unit tests for the ranking and triage metrics (standard library + numpy)."""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal import SET_BOTH, SET_EMPTY, SET_FRAUD, SET_NORMAL  # noqa: E402
from src.metrics import (  # noqa: E402
    aggregate_rows,
    average_precision_score,
    conformal_report,
    roc_auc_score,
    write_csv,
)


def brute_force_auc(labels, scores):
    """ROC-AUC as the mean over positive/negative pairs, with ties worth 0.5."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = (positive[:, None] > negative[None, :]).astype(float)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return comparisons.mean()


class TestRankingMetrics(unittest.TestCase):
    def test_average_precision_of_perfect_ranking_is_one(self):
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        self.assertAlmostEqual(average_precision_score(labels, scores), 1.0)

    def test_average_precision_of_hand_computable_case(self):
        # positives at ranks 1 and 3 -> AP = (1/2) * (1/1 + 2/3)
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        expected = 0.5 * (1.0 + 2.0 / 3.0)
        self.assertAlmostEqual(average_precision_score(labels, scores), expected)

    def test_average_precision_without_positives_is_nan(self):
        self.assertTrue(math.isnan(average_precision_score([0, 0], [0.1, 0.2])))

    def test_average_precision_prefers_better_ranking(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        good = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.7, 0.05, 0.3])
        bad = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.15, 0.3, 0.05])
        self.assertGreater(
            average_precision_score(labels, good), average_precision_score(labels, bad)
        )

    def test_roc_auc_matches_brute_force(self):
        rng = np.random.default_rng(3)
        labels = (rng.random(300) < 0.2).astype(int)
        scores = np.clip(rng.beta(2.0, 5.0, size=300) + 0.4 * labels, 0.0, 1.0)
        self.assertAlmostEqual(roc_auc_score(labels, scores), brute_force_auc(labels, scores))

    def test_roc_auc_counts_ties_as_one_half(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(roc_auc_score(labels, scores), 0.5)

    def test_roc_auc_rejects_degenerate_labels(self):
        self.assertTrue(math.isnan(roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3])))


class TestConformalReport(unittest.TestCase):
    def setUp(self):
        # 8 transactions: 6 legitimate, 2 fraud.
        self.labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        self.codes = np.array(
            [
                SET_NORMAL,  # legitimate, auto-approved
                SET_NORMAL,  # legitimate, auto-approved
                SET_BOTH,    # legitimate, reviewed
                SET_EMPTY,   # legitimate, empty set
                SET_FRAUD,   # legitimate, wrongly auto-declined
                SET_NORMAL,  # legitimate, auto-approved
                SET_FRAUD,   # fraud, auto-declined
                SET_BOTH,    # fraud, reviewed
            ]
        )

    def test_counts_and_rates(self):
        report = conformal_report(self.codes, self.labels)
        self.assertEqual(report["n_test"], 8)
        self.assertEqual(report["n_fraud"], 2)
        self.assertEqual(report["n_normal"], 6)
        self.assertAlmostEqual(report["triage_rate"], 2 / 8)
        self.assertEqual(report["review_queue_size"], 2)
        self.assertAlmostEqual(report["empty_rate"], 1 / 8)
        self.assertAlmostEqual(report["confident_fraud_rate"], 2 / 8)
        self.assertAlmostEqual(report["confident_normal_rate"], 3 / 8)
        # set sizes: 3 singletons {0} + 2 pairs {0,1} + 1 empty + 2 singletons {1}
        self.assertAlmostEqual(report["mean_set_size"], 9 / 8)

    def test_class_wise_coverage(self):
        report = conformal_report(self.codes, self.labels)
        # legitimate: covered by {0}, {0}, {0,1}, {0}; excluded by {}, {1}
        self.assertAlmostEqual(report["coverage_normal"], 4 / 6)
        # fraud: covered by {1} and {0,1}
        self.assertAlmostEqual(report["coverage_fraud"], 2 / 2)
        self.assertAlmostEqual(report["coverage"], (4 + 2) / 8)

    def test_singleton_precision_and_recall(self):
        report = conformal_report(self.codes, self.labels)
        # confident fraud: 2 flagged, 1 of them truly fraud
        self.assertAlmostEqual(report["singleton_fraud_precision"], 0.5)
        self.assertAlmostEqual(report["singleton_fraud_recall"], 0.5)
        # confident normal: 3 flagged, all legitimate
        self.assertAlmostEqual(report["singleton_normal_precision"], 1.0)
        self.assertAlmostEqual(report["singleton_normal_recall"], 3 / 6)
        self.assertAlmostEqual(report["fraud_triage_capture"], 0.5)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            conformal_report(self.codes, self.labels[:-1])


class TestAggregationAndCsv(unittest.TestCase):
    def test_aggregate_rows_mean_and_std(self):
        rows = [
            {"dataset": "ulb", "model": "xgboost", "method": "standard", "alpha": 0.1,
             "seed": 42, "coverage_fraud": 0.2, "triage_rate": 0.01},
            {"dataset": "ulb", "model": "xgboost", "method": "standard", "alpha": 0.1,
             "seed": 43, "coverage_fraud": 0.4, "triage_rate": 0.03},
        ]
        summary = aggregate_rows(rows)
        self.assertEqual(len(summary), 1)
        entry = summary[0]
        self.assertAlmostEqual(entry["coverage_fraud_mean"], 0.3)
        self.assertAlmostEqual(entry["coverage_fraud_std"], math.sqrt(0.02))
        self.assertEqual(entry["n_seeds"], 2)
        self.assertNotIn("seed", entry)

    def test_write_csv_roundtrip(self):
        rows = [{"dataset": "ulb", "model": "xgboost", "coverage_fraud": 0.5, "n_test": 7}]
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(
                Path(directory) / "nested" / "metrics.csv",
                rows,
                columns=["dataset", "model", "coverage_fraud", "n_test"],
            )
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                parsed = list(reader)
        self.assertEqual(parsed[0]["coverage_fraud"], "0.500000")
        self.assertEqual(parsed[0]["n_test"], "7")


if __name__ == "__main__":
    unittest.main()
