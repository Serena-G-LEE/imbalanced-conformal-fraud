#!/usr/bin/env python3
"""Run the class-conditional conformal triage experiment.

Example
-------
    python run_experiment.py --dataset ulb --data-path data/ulb/creditcard.csv

For every model, seed, calibration method and alpha the script reports ranking
quality (PR-AUC, ROC-AUC) plus the conformal triage statistics (class-wise
coverage, review-queue rate, empty-set rate, singleton precision/recall), writes
per-seed and seed-aggregated CSVs, and draws the coverage / review-cost
trade-off figures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.conformal import MondrianCP, StandardSplitCP
from src.data import SplitDataset, load_ieee_cis, load_ulb
from src.metrics import (
    METRIC_COLUMNS,
    aggregate_rows,
    average_precision_score,
    conformal_report,
    roc_auc_score,
    write_csv,
)
from src.models import train_and_predict
from src.utils import ensure_dir, load_config, save_json, set_global_seed, utc_timestamp
from src.visualize import plot_alpha_tradeoff

CP_REGISTRY = {"standard": StandardSplitCP, "mondrian": MondrianCP}

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_ALPHAS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_METHODS = ("standard", "mondrian")

SUMMARY_COLUMNS = (
    "dataset",
    "model",
    "method",
    "alpha",
    "n_seeds",
    "coverage_fraud_mean",
    "coverage_fraud_std",
    "coverage_normal_mean",
    "coverage_normal_std",
    "coverage_mean",
    "coverage_std",
    "triage_rate_mean",
    "triage_rate_std",
    "empty_rate_mean",
    "empty_rate_std",
    "fraud_triage_capture_mean",
    "mean_set_size_mean",
    "singleton_fraud_precision_mean",
    "singleton_fraud_recall_mean",
    "pr_auc_mean",
    "pr_auc_std",
    "roc_auc_mean",
    "roc_auc_std",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Class-conditional conformal prediction for fraud triage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("ulb", "ieee_cis"), default="ulb")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config (default: configs/<dataset>.yaml)")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="ULB: path to creditcard.csv; IEEE-CIS: directory holding the CSVs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="root directory for CSVs and figures (default: from config)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of model names in the config (e.g. --models xgboost)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--no-figures", action="store_true", help="skip PNG generation")
    return parser


def _data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    section = config.get("data", {})
    if not isinstance(section, dict):
        raise ValueError("config key 'data' must be a mapping")
    return section


def _split_fractions(data_config: Dict[str, Any]) -> Sequence[float]:
    splits = data_config.get("splits", {}) or {}
    return (
        float(splits.get("train", 0.60)),
        float(splits.get("calibration", 0.20)),
        float(splits.get("test", 0.20)),
    )


def load_dataset(
    dataset: str,
    config: Dict[str, Any],
    data_path: Path | None = None,
) -> SplitDataset:
    """Build the chronologically split dataset described by ``config``."""
    section = _data_config(config)
    fractions = _split_fractions(section)

    if dataset == "ulb":
        path = data_path or section.get("path")
        if path is None:
            raise SystemExit("ULB dataset path missing: pass --data-path or set data.path")
        return load_ulb(
            path,
            time_column=section.get("time_column", "Time"),
            label_column=section.get("label_column", "Class"),
            fractions=fractions,
            log1p_amount=bool(section.get("log1p_amount", True)),
        )

    directory = data_path or section.get("dir_path", "data/ieee_cis")
    return load_ieee_cis(
        directory,
        transaction_file=section.get("transaction_file", "train_transaction.csv"),
        identity_file=section.get("identity_file", "train_identity.csv"),
        time_column=section.get("time_column", "TransactionDT"),
        label_column=section.get("label_column", "isFraud"),
        fractions=fractions,
        log1p_amount=bool(section.get("log1p_amount", True)),
        fill_value=float(section.get("fill_value", -1.0)),
    )


def _resolve_settings(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    experiment = config.get("experiment", {}) or {}
    seeds = tuple(args.seeds) if args.seeds else tuple(experiment.get("seeds", DEFAULT_SEEDS))
    alphas = tuple(args.alphas) if args.alphas else tuple(experiment.get("alpha_grid", DEFAULT_ALPHAS))
    methods = tuple(experiment.get("methods", DEFAULT_METHODS))
    output_dir = args.output_dir or experiment.get("output_dir", "results")

    model_specs: List[Dict[str, Any]] = list(config.get("models", []) or [])
    if not model_specs:
        raise SystemExit("config has no 'models' entries")
    if args.models:
        wanted = {name.lower() for name in args.models}
        model_specs = [spec for spec in model_specs if str(spec.get("name", "")).lower() in wanted]
        if not model_specs:
            raise SystemExit(f"no configured model matches --models {args.models}")

    for method in methods:
        if method not in CP_REGISTRY:
            raise SystemExit(f"unknown CP method {method!r}; expected {sorted(CP_REGISTRY)}")
    for alpha in alphas:
        if not 0.0 < float(alpha) < 1.0:
            raise SystemExit(f"alpha must lie in (0, 1), got {alpha}")

    return {
        "seeds": [int(seed) for seed in seeds],
        "alphas": [float(alpha) for alpha in alphas],
        "methods": list(methods),
        "output_dir": Path(output_dir),
        "models": model_specs,
    }


def _run_single_seed(
    dataset: SplitDataset,
    model_spec: Dict[str, Any],
    seed: int,
    alphas: Sequence[float],
    methods: Sequence[str],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """Train one model and evaluate every (method, alpha) cell for one seed."""
    name = str(model_spec.get("name", "")).strip().lower()
    prediction = train_and_predict(
        name,
        dataset.X_train,
        dataset.y_train,
        dataset.X_cal,
        dataset.X_test,
        params=model_spec.get("params"),
        scale_pos_weight=model_spec.get("scale_pos_weight"),
        seed=seed,
    )

    pr_auc = average_precision_score(dataset.y_test, prediction.prob_test)
    roc_auc = roc_auc_score(dataset.y_test, prediction.prob_test)

    rows: List[Dict[str, Any]] = []
    for method in methods:
        for alpha in alphas:
            cp = CP_REGISTRY[method](alpha=alpha).calibrate(
                prediction.prob_cal, dataset.y_cal
            )
            codes = cp.predict_set_codes(prediction.prob_test)
            row: Dict[str, Any] = {
                "dataset": dataset_name,
                "model": name,
                "method": method,
                "alpha": float(alpha),
                "seed": int(seed),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
            }
            calibration = cp.describe()
            calibration.pop("method", None)
            calibration.pop("alpha", None)
            row.update(calibration)
            row.update(conformal_report(codes, dataset.y_test))
            rows.append(row)
    return rows


def _print_summary(summary_rows: Sequence[Dict[str, Any]]) -> None:
    if not summary_rows:
        return
    header = (
        f"{'model':<10} {'method':<10} {'alpha':>6} "
        f"{'fraud_cov':>10} {'normal_cov':>11} {'triage':>8} {'empty':>7} {'pr_auc':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{str(row['model']):<10} {str(row['method']):<10} {float(row['alpha']):>6.3f} "
            f"{float(row['coverage_fraud_mean']):>10.4f} "
            f"{float(row['coverage_normal_mean']):>11.4f} "
            f"{float(row['triage_rate_mean']):>8.4f} "
            f"{float(row['empty_rate_mean']):>7.4f} "
            f"{float(row['pr_auc_mean']):>8.4f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config_path = args.config or Path("configs") / f"{args.dataset}.yaml"
    config = load_config(config_path)
    dataset_name = str(config.get("dataset", args.dataset))
    if dataset_name not in {"ulb", "ieee_cis"}:
        raise SystemExit(f"unsupported dataset in config: {dataset_name!r}")

    settings = _resolve_settings(args, config)
    dataset = load_dataset(args.dataset, config, args.data_path)

    print(f"[info] dataset       : {dataset.name}")
    for block in ("train", "calibration", "test"):
        stats = dataset.summary()[block]
        print(
            f"[info] {block:<11}: {stats['n']:>8,} rows, "
            f"{stats['n_fraud']:>6,} fraud ({stats['prevalence']:.4%})"
        )

    run_root = ensure_dir(settings["output_dir"] / dataset_name)
    figure_dir = ensure_dir(run_root / "figures")

    started = time.time()
    rows: List[Dict[str, Any]] = []
    for model_spec in settings["models"]:
        model_name = str(model_spec.get("name", "")).strip().lower()
        for seed in settings["seeds"]:
            set_global_seed(seed)
            seed_rows = _run_single_seed(
                dataset, model_spec, seed, settings["alphas"], settings["methods"], dataset_name
            )
            rows.extend(seed_rows)
            best = max(seed_rows, key=lambda row: row["pr_auc"])
            print(
                f"[run ] {model_name:<9} seed={seed} "
                f"pr_auc={best['pr_auc']:.4f} roc_auc={best['roc_auc']:.4f}"
            )

    per_seed_path = write_csv(run_root / "per_seed_metrics.csv", rows, METRIC_COLUMNS)
    summary = aggregate_rows(rows)
    summary_path = write_csv(run_root / "summary_metrics.csv", summary, SUMMARY_COLUMNS)

    figures: List[str] = []
    if not args.no_figures:
        for model_spec in settings["models"]:
            model_name = str(model_spec.get("name", "")).strip().lower()
            model_rows = [row for row in summary if row["model"] == model_name]
            if not model_rows:
                continue
            try:
                paths = plot_alpha_tradeoff(
                    model_rows, figure_dir, dataset=dataset_name, model=model_name
                )
            except RuntimeError as error:
                print(f"[warn] skipping figures: {error}")
                break
            figures.extend(str(path) for path in paths)

    metadata = {
        "dataset": dataset_name,
        "config": str(config_path),
        "created_at": utc_timestamp(),
        "elapsed_seconds": round(time.time() - started, 2),
        "seeds": settings["seeds"],
        "alphas": settings["alphas"],
        "methods": settings["methods"],
        "models": [str(spec.get("name")) for spec in settings["models"]],
        "dataset_summary": dataset.summary(),
        "n_runs": len(rows),
        "outputs": {
            "per_seed_metrics": str(per_seed_path),
            "summary_metrics": str(summary_path),
            "figures": figures,
        },
    }
    save_json(run_root / "run_metadata.json", metadata)

    _print_summary(summary)
    print(f"\n[done] per-seed metrics : {per_seed_path}")
    print(f"[done] summary metrics  : {summary_path}")
    for figure in figures:
        print(f"[done] figure           : {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
