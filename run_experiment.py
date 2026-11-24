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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.conformal import MondrianCP, StandardSplitCP
from src.data import DEFAULT_PROB_CAL_FRACTION, SplitDataset, load_ieee_cis, load_ulb
from src.metrics import (
    METRIC_COLUMNS,
    aggregate_rows,
    average_precision_score,
    brier_score,
    conformal_report,
    ece,
    roc_auc_score,
    write_csv,
)
from src.models import CALIBRATORS, train_and_predict
from src.utils import ensure_dir, load_config, save_json, set_global_seed, utc_timestamp
from src.visualize import plot_alpha_tradeoff, plot_alpha_variant_tradeoff

CP_REGISTRY = {"standard": StandardSplitCP, "mondrian": MondrianCP}

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_ALPHAS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_METHODS = ("standard", "mondrian")

SUMMARY_COLUMNS = (
    "dataset",
    "model",
    "variant",
    "calibrator",
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
    "brier_mean",
    "brier_std",
    "ece_mean",
    "ece_std",
    "brier_raw_mean",
    "brier_raw_std",
    "ece_raw_mean",
    "ece_raw_std",
)


def metric_filenames(dataset_name: str) -> Dict[str, str]:
    """Dataset-prefixed names for the two metric CSVs of a run.

    Runs for several datasets are written under ``results/<dataset>/`` but are
    routinely copied side by side (or into one analysis folder), where a bare
    ``summary_metrics.csv`` says nothing about which benchmark it holds.  Every
    file therefore carries its dataset name.
    """
    return {
        "per_seed": f"{dataset_name}_per_seed_metrics.csv",
        "summary": f"{dataset_name}_summary_metrics.csv",
    }


def _calibrator_of(spec: Dict[str, Any]) -> str:
    """Read a model spec's calibrator, normalising aliases and empties."""
    value = spec.get("calibrator", "none")
    key = str(value if value is not None else "none").strip().lower()
    return "none" if key in {"", "identity"} else key


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
                        help="subset of model names or variant labels in the config (e.g. --models xgboost unweighted_platt)")
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
        model_specs = [
            spec
            for spec in model_specs
            if str(spec.get("name", "")).lower() in wanted
            or str(spec.get("variant", "")).lower() in wanted
        ]
        if not model_specs:
            raise SystemExit(f"no configured model matches --models {args.models}")

    for method in methods:
        if method not in CP_REGISTRY:
            raise SystemExit(f"unknown CP method {method!r}; expected {sorted(CP_REGISTRY)}")
    for alpha in alphas:
        if not 0.0 < float(alpha) < 1.0:
            raise SystemExit(f"alpha must lie in (0, 1), got {alpha}")

    for spec in model_specs:
        calibrator = _calibrator_of(spec)
        if calibrator not in CALIBRATORS:
            raise SystemExit(
                f"unknown calibrator {spec.get('calibrator')!r} for model "
                f"{spec.get('name')!r}; expected one of {list(CALIBRATORS)}"
            )

    # Duplicate variant labels would be silently averaged together.
    seen: Dict[tuple, int] = {}
    for index, spec in enumerate(model_specs):
        key = (str(spec.get("name", "")).lower(), str(spec.get("variant", "")))
        if key[1] == "":
            continue
        if key in seen:
            raise SystemExit(
                f"models[{seen[key]}] and models[{index}] share the variant label {key[1]!r}; "
                "give every ablation arm a distinct 'variant'"
            )
        seen[key] = index

    prob_cal_fraction = _resolve_prob_cal_fraction(config, model_specs)

    return {
        "seeds": [int(seed) for seed in seeds],
        "alphas": [float(alpha) for alpha in alphas],
        "methods": list(methods),
        "output_dir": Path(output_dir),
        "models": model_specs,
        "prob_cal_fraction": prob_cal_fraction,
    }


def _resolve_prob_cal_fraction(
    config: Dict[str, Any], model_specs: Sequence[Dict[str, Any]]
) -> Optional[float]:
    """One shared probability-calibration share, or ``None`` when unused.

    Every calibrated arm must fit its booster on the same rows for the ablation
    to be a comparison of weighting and calibration rather than of data volume.
    """
    if not any(_calibrator_of(spec) != "none" for spec in model_specs):
        return None
    default = float(_data_config(config).get("prob_cal_fraction", DEFAULT_PROB_CAL_FRACTION))
    declared = {
        float(spec.get("prob_cal_fraction", default))
        for spec in model_specs
        if _calibrator_of(spec) != "none"
    }
    if len(declared) > 1:
        raise SystemExit(
            "calibrated variants must share a single prob_cal_fraction so they are fit on the "
            f"same rows, got {sorted(declared)}"
        )
    fraction = declared.pop()
    if not 0.0 < fraction < 1.0:
        raise SystemExit(f"prob_cal_fraction must lie in (0, 1), got {fraction}")
    return fraction


def _run_single_seed(
    dataset: SplitDataset,
    model_spec: Dict[str, Any],
    seed: int,
    alphas: Sequence[float],
    methods: Sequence[str],
    dataset_name: str,
    fit_log: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Train one model and evaluate every (method, alpha) cell for one seed."""
    name = str(model_spec.get("name", "")).strip().lower()
    calibrator = _calibrator_of(model_spec)
    prediction = train_and_predict(
        name,
        dataset.X_train,
        dataset.y_train,
        dataset.X_cal,
        dataset.X_test,
        params=model_spec.get("params"),
        scale_pos_weight=model_spec.get("scale_pos_weight"),
        seed=seed,
        calibrator=calibrator,
        X_prob_cal=dataset.X_prob_cal,
        y_prob_cal=dataset.y_prob_cal,
        variant=model_spec.get("variant"),
    )
    if fit_log is not None:
        fit_log.append({"seed": int(seed), **prediction.describe()})

    pr_auc = average_precision_score(dataset.y_test, prediction.prob_test)
    roc_auc = roc_auc_score(dataset.y_test, prediction.prob_test)
    brier = brier_score(dataset.y_test, prediction.prob_test)
    ece_value = ece(dataset.y_test, prediction.prob_test)
    brier_raw = brier_score(dataset.y_test, prediction.raw_prob_test)
    ece_raw = ece(dataset.y_test, prediction.raw_prob_test)

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
                "variant": prediction.variant,
                "calibrator": calibrator,
                "method": method,
                "alpha": float(alpha),
                "seed": int(seed),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "brier": float(brier),
                "ece": float(ece_value),
                "brier_raw": float(brier_raw),
                "ece_raw": float(ece_raw),
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
        f"{'model':<10} {'variant':<20} {'method':<10} {'alpha':>6} "
        f"{'fraud_cov':>10} {'normal_cov':>11} {'triage':>8} {'empty':>7} {'pr_auc':>8} "
        f"{'brier':>8} {'ece':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{str(row['model']):<10} {str(row.get('variant') or '-'):<20} "
            f"{str(row['method']):<10} {float(row['alpha']):>6.3f} "
            f"{float(row['coverage_fraud_mean']):>10.4f} "
            f"{float(row['coverage_normal_mean']):>11.4f} "
            f"{float(row['triage_rate_mean']):>8.4f} "
            f"{float(row['empty_rate_mean']):>7.4f} "
            f"{float(row['pr_auc_mean']):>8.4f} "
            f"{float(row['brier_mean']):>8.4f} "
            f"{float(row['ece_mean']):>8.4f}"
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

    if settings["prob_cal_fraction"] is not None:
        # Carve the probability-calibration block out of the training window
        # before any booster is fit; the conformal calibration block stays
        # untouched and only supplies quantiles.
        dataset = dataset.with_prob_cal_split(settings["prob_cal_fraction"])

    print(f"[info] dataset       : {dataset.name}")
    for block in ("train", "prob_cal", "calibration", "test"):
        stats = dataset.summary().get(block)
        if not stats:
            continue
        print(
            f"[info] {block:<11}: {stats['n']:>8,} rows, "
            f"{stats['n_fraud']:>6,} fraud ({stats['prevalence']:.4%})"
        )
    if dataset.has_prob_cal:
        print(
            "[info] probability calibration is fit on 'prob_cal' only; "
            "conformal quantiles stay on 'calibration'"
        )

    run_root = ensure_dir(settings["output_dir"] / dataset_name)
    figure_dir = ensure_dir(run_root / "figures")

    started = time.time()
    rows: List[Dict[str, Any]] = []
    fit_log: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for model_spec in settings["models"]:
        model_name = str(model_spec.get("name", "")).strip().lower()
        variant_label = str(model_spec.get("variant") or "")
        for position, seed in enumerate(settings["seeds"]):
            set_global_seed(seed)
            try:
                seed_rows = _run_single_seed(
                    dataset,
                    model_spec,
                    seed,
                    settings["alphas"],
                    settings["methods"],
                    dataset_name,
                    fit_log=fit_log if position == 0 else None,
                )
            except Exception as error:  # noqa: BLE001 - one arm must not kill the sweep
                # A long ablation should not lose the arms that already finished
                # because one estimator could not be imported or trained.
                message = f"{type(error).__name__}: {error}"
                print(f"[warn] skipping {model_name} {variant_label or '-'} seed={seed}: {message}")
                failures.append(
                    {
                        "model": model_name,
                        "variant": variant_label,
                        "seed": int(seed),
                        "error": message,
                    }
                )
                break
            rows.extend(seed_rows)
            best = max(seed_rows, key=lambda row: row["pr_auc"])
            print(
                f"[run ] {model_name:<9} {best['variant']:<20} seed={seed} "
                f"pr_auc={best['pr_auc']:.4f} roc_auc={best['roc_auc']:.4f} "
                f"brier={best['brier']:.5f} ece={best['ece']:.5f}"
            )

    if not rows:
        raise SystemExit(
            "no arm produced results; first failure: "
            + (failures[0]["error"] if failures else "unknown")
        )

    filenames = metric_filenames(dataset_name)
    per_seed_path = write_csv(run_root / filenames["per_seed"], rows, METRIC_COLUMNS)
    summary = aggregate_rows(rows)
    summary_path = write_csv(run_root / filenames["summary"], summary, SUMMARY_COLUMNS)

    figures: List[str] = []
    if not args.no_figures:
        # One figure per model.  With more than one ablation arm the figure has
        # to compare variants, with a single arm the base scheme comparison is
        # still the informative plot.
        plotted: List[str] = []
        for model_spec in settings["models"]:
            model_name = str(model_spec.get("name", "")).strip().lower()
            if model_name in plotted:
                continue
            plotted.append(model_name)
            variants = [
                str(spec.get("variant") or "")
                for spec in settings["models"]
                if str(spec.get("name", "")).strip().lower() == model_name
            ]
            variants = [variant for variant in dict.fromkeys(variants) if variant]
            model_rows = [row for row in summary if row["model"] == model_name]
            if not model_rows:
                continue
            try:
                if len(variants) > 1:
                    primary_method = "mondrian" if "mondrian" in settings["methods"] else settings["methods"][0]
                    paths = plot_alpha_variant_tradeoff(
                        model_rows,
                        figure_dir,
                        dataset=dataset_name,
                        model=model_name,
                        method=primary_method,
                        variants=variants,
                    )
                else:
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
        "variants": [str(spec.get("variant") or "") for spec in settings["models"]],
        "calibrators": [_calibrator_of(spec) for spec in settings["models"]],
        "prob_cal_fraction": settings["prob_cal_fraction"],
        "dataset_summary": dataset.summary(),
        "fits": fit_log,
        "failures": failures,
        "n_runs": len(rows),
        "outputs": {
            "per_seed_metrics": str(per_seed_path),
            "summary_metrics": str(summary_path),
            "figures": figures,
        },
    }
    save_json(run_root / "run_metadata.json", metadata)

    _print_summary(summary)
    if failures:
        print(f"\n[warn] {len(failures)} arm(s) were skipped:")
        for failure in failures:
            print(
                f"       {failure['model']} {failure['variant'] or '-'} "
                f"seed={failure['seed']}: {failure['error']}"
            )
    print(f"\n[done] per-seed metrics : {per_seed_path}")
    print(f"[done] summary metrics  : {summary_path}")
    for figure in figures:
        print(f"[done] figure           : {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
