"""Unit tests for the metric output filenames.

Runs for different datasets are written under ``results/<dataset>/`` and then
copied side by side for analysis, so every CSV carries its dataset name.  The
end-to-end test drives ``run_experiment.main`` with a stubbed loader and model,
which keeps the assertion about the *written* paths independent of any dataset
download or booster installation.
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

import run_experiment  # noqa: E402
from src.data import SplitDataset  # noqa: E402
from src.models import ModelPrediction  # noqa: E402

CONFIG_TEMPLATE = """\
dataset: {dataset}
data:
  path: data/ulb/creditcard.csv
  splits: {{train: 0.6, calibration: 0.2, test: 0.2}}
models:
  - name: xgboost
    calibrator: none
    variant: weighted_raw
experiment:
  seeds: [42]
  alpha_grid: [0.05]
  methods: [mondrian]
  output_dir: results
"""


def _fake_dataset(name: str) -> SplitDataset:
    rng = np.random.default_rng(7)

    def block(n_rows: int, n_fraud: int):
        X = rng.normal(size=(n_rows, 3)).astype(np.float32)
        y = np.zeros(n_rows, dtype=np.int64)
        y[:n_fraud] = 1
        return X, y

    X_train, y_train = block(40, 4)
    X_cal, y_cal = block(20, 2)
    X_test, y_test = block(20, 2)
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


class TestMetricFilenames(unittest.TestCase):
    def test_names_are_dataset_prefixed(self):
        names = run_experiment.metric_filenames("ulb")
        self.assertEqual(names["summary"], "ulb_summary_metrics.csv")
        self.assertEqual(names["per_seed"], "ulb_per_seed_metrics.csv")

        names = run_experiment.metric_filenames("ieee_cis")
        self.assertEqual(names["summary"], "ieee_cis_summary_metrics.csv")
        self.assertEqual(names["per_seed"], "ieee_cis_per_seed_metrics.csv")

    def test_datasets_do_not_share_a_filename(self):
        ulb = set(run_experiment.metric_filenames("ulb").values())
        ieee = set(run_experiment.metric_filenames("ieee_cis").values())
        self.assertTrue(ulb.isdisjoint(ieee))

    def test_no_generic_filename_survives(self):
        for dataset in ("ulb", "ieee_cis"):
            for path in run_experiment.metric_filenames(dataset).values():
                self.assertNotIn(path, {"summary_metrics.csv", "per_seed_metrics.csv"})


class TestWrittenOutputs(unittest.TestCase):
    def test_run_writes_dataset_prefixed_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "ulb_stub.yaml"
            config_path.write_text(CONFIG_TEMPLATE.format(dataset="ulb"), encoding="utf-8")
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

            run_root = output_dir / "ulb"
            self.assertTrue((run_root / "ulb_summary_metrics.csv").is_file())
            self.assertTrue((run_root / "ulb_per_seed_metrics.csv").is_file())
            self.assertFalse((run_root / "summary_metrics.csv").exists())
            self.assertFalse((run_root / "per_seed_metrics.csv").exists())

            metadata = (run_root / "run_metadata.json").read_text(encoding="utf-8")
            self.assertIn("ulb_summary_metrics.csv", metadata)
            self.assertIn("ulb_per_seed_metrics.csv", metadata)


if __name__ == "__main__":
    unittest.main()
