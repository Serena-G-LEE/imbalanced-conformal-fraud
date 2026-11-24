"""Unit tests for the chronological split logic.

The pandas-dependent loaders are exercised only when pandas is installed; the
split arithmetic itself is pure numpy and always tested.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import SplitDataset, temporal_split_indices  # noqa: E402

try:  # pragma: no cover - depends on the environment
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False


class TestTemporalSplit(unittest.TestCase):
    def test_blocks_are_contiguous_ordered_and_complete(self):
        train, calibration, test = temporal_split_indices(1_000, (0.6, 0.2, 0.2))
        self.assertEqual(len(train), 600)
        self.assertEqual(len(calibration), 200)
        self.assertEqual(len(test), 200)
        self.assertEqual(train[0], 0)
        self.assertEqual(train[-1] + 1, calibration[0])
        self.assertEqual(calibration[-1] + 1, test[0])
        self.assertEqual(test[-1], 999)
        combined = np.concatenate([train, calibration, test])
        np.testing.assert_array_equal(combined, np.arange(1_000))

    def test_no_overlap_and_no_gap(self):
        train, calibration, test = temporal_split_indices(97, (0.5, 0.25, 0.25))
        self.assertEqual(set(train) & set(calibration), set())
        self.assertEqual(set(calibration) & set(test), set())
        self.assertEqual(len(train) + len(calibration) + len(test), 97)

    def test_small_input_still_produces_three_blocks(self):
        train, calibration, test = temporal_split_indices(3, (0.6, 0.2, 0.2))
        for block in (train, calibration, test):
            self.assertGreaterEqual(len(block), 1)

    def test_rejects_invalid_fractions(self):
        with self.assertRaises(ValueError):
            temporal_split_indices(100, (0.5, 0.5))
        with self.assertRaises(ValueError):
            temporal_split_indices(100, (0.5, 0.2, 0.2))  # does not sum to 1
        with self.assertRaises(ValueError):
            temporal_split_indices(100, (0.8, 0.2, 0.0))  # empty test block

    def test_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            temporal_split_indices(2)

    def test_split_dataset_summary(self):
        zeros = np.zeros((2, 3), dtype=np.float32)
        ones = np.ones(2, dtype=np.int64)
        dataset = SplitDataset(
            name="toy",
            X_train=zeros, y_train=np.array([0, 1], dtype=np.int64),
            X_cal=zeros, y_cal=ones,
            X_test=zeros, y_test=np.array([0, 0], dtype=np.int64),
            feature_names=["a", "b", "c"],
        )
        summary = dataset.summary()
        self.assertEqual(summary["n_features"], 3)
        self.assertEqual(summary["train"]["n_fraud"], 1)
        self.assertAlmostEqual(summary["train"]["prevalence"], 0.5)
        self.assertAlmostEqual(summary["test"]["prevalence"], 0.0)


@unittest.skipUnless(HAS_PANDAS, "pandas is not installed")
class TestUlbLoader(unittest.TestCase):
    def test_loader_sorts_by_time_and_splits_chronologically(self):
        import tempfile

        from src.data import load_ulb

        rng = np.random.default_rng(0)
        n = 300
        times = rng.permutation(n)  # deliberately unsorted
        frame = pd.DataFrame(
            {
                "Time": times,
                "V1": rng.normal(size=n),
                "V2": rng.normal(size=n),
                "Amount": rng.uniform(0, 500, size=n),
                "Class": (rng.random(n) < 0.1).astype(int),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "creditcard.csv"
            frame.to_csv(path, index=False)
            dataset = load_ulb(path, fractions=(0.6, 0.2, 0.2))

        self.assertEqual(len(dataset.X_train) + len(dataset.X_cal) + len(dataset.X_test), n)
        self.assertEqual(dataset.X_train.shape[1], 3)
        self.assertIn("Amount", dataset.feature_names)
        ranges = dataset.meta["time_range"]
        self.assertLess(ranges["train"][1], ranges["calibration"][1])
        self.assertLess(ranges["calibration"][1], ranges["test"][1])
        # log1p(Amount) keeps amounts within a tree-friendly range
        self.assertLess(float(np.max(dataset.X_train)), 10.0)


if __name__ == "__main__":
    unittest.main()
