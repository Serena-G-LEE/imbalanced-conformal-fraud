"""Ranking metrics and triage-cost metrics for the conformal layer.

Everything here is implemented with ``numpy`` and the standard library so that
the numbers do not depend on an installed scikit-learn version and can be unit
tested without heavy dependencies.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .conformal import SET_BOTH, SET_EMPTY, SET_FRAUD, SET_NORMAL, codes_to_membership

__all__ = [
    "METRIC_COLUMNS",
    "aggregate_rows",
    "average_precision_score",
    "brier_score",
    "conformal_report",
    "ece",
    "roc_auc_score",
    "write_csv",
]

#: Column order for the per-seed metrics CSV.
METRIC_COLUMNS: Sequence[str] = (
    "dataset",
    "model",
    "variant",
    "calibrator",
    "method",
    "alpha",
    "seed",
    "pr_auc",
    "roc_auc",
    "brier",
    "ece",
    "brier_raw",
    "ece_raw",
    "prevalence",
    "n_test",
    "n_normal",
    "n_fraud",
    "coverage",
    "coverage_normal",
    "coverage_fraud",
    "triage_rate",
    "empty_rate",
    "confident_fraud_rate",
    "confident_normal_rate",
    "mean_set_size",
    "review_queue_size",
    "fraud_triage_capture",
    "singleton_fraud_precision",
    "singleton_fraud_recall",
    "singleton_normal_precision",
    "singleton_normal_recall",
    "n_cal_normal",
    "n_cal_fraud",
    "q_normal",
    "q_fraud",
    "min_prob_positive_for_fraud",
    "max_prob_positive_for_normal",
)

#: Identifiers that must never be averaged across seeds.
_NON_METRIC_KEYS = {"dataset", "model", "variant", "calibrator", "method", "seed", "params"}

#: Default binning strategy for ``ece``: equal-mass bins.  With 0.172 %
#: positives a uniform 15-bin grid puts every transaction in the first bin, so
#: the score would be uninformative.
_DEFAULT_ECE_BINS = 15
_DEFAULT_ECE_STRATEGY = "quantile"


def _as_pair(labels, scores):
    target = np.asarray(labels).ravel().astype(int)
    values = np.asarray(scores, dtype=float).ravel()
    if target.size != values.size:
        raise ValueError(f"labels and scores differ in length: {target.size} != {values.size}")
    if not np.all(np.isin(target, [0, 1])):
        raise ValueError("labels must be binary (0/1)")
    return target, values


def average_precision_score(labels, scores) -> float:
    """Average precision (area under the precision-recall curve, step rule).

    Implemented as ``sum_i (R_i - R_{i-1}) * P_i`` over the distinct score
    thresholds, which is the same convention as scikit-learn's
    ``average_precision_score``.  Returns ``nan`` when the sample has no
    positive label.
    """
    target, values = _as_pair(labels, scores)
    n_positive = int(target.sum())
    if n_positive == 0:
        return float("nan")

    order = np.argsort(-values, kind="mergesort")
    ordered_target = target[order]
    ordered_scores = values[order]

    true_positive = np.cumsum(ordered_target).astype(float)
    false_positive = np.cumsum(1 - ordered_target).astype(float)

    # Only the last index of each tied score block is a realisable threshold.
    last_of_block = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]), target.size - 1]
    true_positive = true_positive[last_of_block]
    false_positive = false_positive[last_of_block]

    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / n_positive
    increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(increments * precision))


def brier_score(labels, scores) -> float:
    """Mean squared error of the probability forecasts, ``mean((p - y)^2)``.

    The Brier score is a proper scoring rule: it is minimised by the true
    probability, so it drops when a miscalibrated score (e.g. one inflated by
    ``scale_pos_weight``) is replaced by a calibrated one.  Returns ``nan`` for
    an empty sample.
    """
    target, values = _as_pair(labels, scores)
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("probability scores must lie in [0, 1]")
    if target.size == 0:
        return float("nan")
    return float(np.mean((values - target) ** 2))


def _ece_bin_index(values: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        index = np.clip((values * n_bins).astype(int), 0, n_bins - 1)
        return index
    if strategy == "quantile":
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))[1:-1]
        return np.clip(np.searchsorted(edges, values, side="left"), 0, n_bins - 1)
    raise ValueError(f"unknown ECE binning strategy {strategy!r}; expected 'uniform' or 'quantile'")


def ece(
    labels,
    scores,
    n_bins: int = _DEFAULT_ECE_BINS,
    strategy: str = _DEFAULT_ECE_STRATEGY,
) -> float:
    """Expected Calibration Error, ``sum_b (n_b / n) * |acc_b - conf_b|``.

    Bins are either fixed-width over ``[0, 1]`` (``strategy="uniform"``) or
    equal-mass quantile bins of the scores (``strategy="quantile"``, the
    default).  Equal-mass bins are the meaningful choice under extreme
    imbalance: at a 0.172 % fraud rate a uniform grid puts every score into the
    first bin and collapses the metric to the prevalence.

    Returns ``nan`` for an empty sample and raises for scores outside ``[0, 1]``.
    """
    target, values = _as_pair(labels, scores)
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("probability scores must lie in [0, 1]")
    if target.size == 0:
        return float("nan")
    bins = int(n_bins)
    if bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    index = _ece_bin_index(values, bins, str(strategy).strip().lower())
    total = 0.0
    n = float(target.size)
    for bin_id in np.unique(index):
        mask = index == bin_id
        share = float(np.sum(mask)) / n
        confidence = float(np.mean(values[mask]))
        accuracy = float(np.mean(target[mask]))
        total += share * abs(accuracy - confidence)
    return float(total)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties receiving the average rank."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), ordered.size - 1]
    average = (starts + ends + 2.0) / 2.0  # (first_rank + last_rank) / 2
    ranks_sorted = np.repeat(average, ends - starts + 1)
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc_score(labels, scores) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties counted as one half)."""
    target, values = _as_pair(labels, scores)
    n_positive = int(target.sum())
    n_negative = target.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = _average_ranks(values)
    u_statistic = ranks[target == 1].sum() - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def conformal_report(codes, labels) -> Dict[str, float]:
    """Coverage and triage statistics for one set of prediction-set codes.

    ``coverage_*`` is the fraction of samples whose true label is inside the
    prediction set, overall and per class.  ``triage_rate`` is the share of
    transactions routed to manual review (``{0, 1}``); ``empty_rate`` the share
    with no plausible label.  ``fraud_triage_capture`` is the fraction of true
    frauds that land in the review queue rather than being auto-declined.
    """
    membership = codes_to_membership(codes)
    target = np.asarray(labels).ravel().astype(int)
    if membership.shape[0] != target.size:
        raise ValueError(
            f"codes and labels differ in length: {membership.shape[0]} != {target.size}"
        )
    if not np.all(np.isin(target, [0, 1])):
        raise ValueError("labels must be binary (0/1)")

    set_codes = np.asarray(codes).ravel().astype(int)
    n = target.size
    covered = membership[np.arange(n), target]
    positive = target == 1
    negative = ~positive
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())

    flagged_fraud = set_codes == SET_FRAUD
    confident_normal = set_codes == SET_NORMAL

    report: Dict[str, float] = {
        "n_test": int(n),
        "n_normal": int(n_negative),
        "n_fraud": int(n_positive),
        "prevalence": n_positive / n if n else float("nan"),
        "coverage": float(np.mean(covered)) if n else float("nan"),
        "coverage_normal": float(np.mean(covered[negative])) if n_negative else float("nan"),
        "coverage_fraud": float(np.mean(covered[positive])) if n_positive else float("nan"),
        "triage_rate": float(np.mean(set_codes == SET_BOTH)) if n else float("nan"),
        "empty_rate": float(np.mean(set_codes == SET_EMPTY)) if n else float("nan"),
        "confident_fraud_rate": float(np.mean(flagged_fraud)) if n else float("nan"),
        "confident_normal_rate": float(np.mean(confident_normal)) if n else float("nan"),
        "mean_set_size": float(np.mean(membership.sum(axis=1))) if n else float("nan"),
        "review_queue_size": int(np.sum(set_codes == SET_BOTH)),
        "fraud_triage_capture": (
            float(np.mean(set_codes[positive] == SET_BOTH)) if n_positive else float("nan")
        ),
        "singleton_fraud_precision": (
            float(np.mean(positive[flagged_fraud])) if flagged_fraud.any() else float("nan")
        ),
        "singleton_fraud_recall": (
            float(np.sum(flagged_fraud & positive) / n_positive) if n_positive else float("nan")
        ),
        "singleton_normal_precision": (
            float(np.mean(negative[confident_normal])) if confident_normal.any() else float("nan")
        ),
        "singleton_normal_recall": (
            float(np.sum(confident_normal & negative) / n_negative) if n_negative else float("nan")
        ),
    }
    return report


def _format_cell(value):
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return f"{number:.6f}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path: str | Path, rows: Iterable[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> Path:
    """Write ``rows`` as CSV with a fixed column order and 6-decimal floats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if columns is None:
        columns = list(METRIC_COLUMNS)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(columns)).writeheader()
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_cell(row.get(column, "")) for column in columns})
    return path


def _numeric_metric_keys(rows: Sequence[Dict[str, object]], group_keys: Sequence[str]) -> List[str]:
    excluded = set(group_keys) | _NON_METRIC_KEYS
    keys: List[str] = []
    for key, value in rows[0].items():
        if key in excluded or isinstance(value, str):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            keys.append(key)
    return keys


def aggregate_rows(
    rows: Sequence[Dict[str, object]],
    group_keys: Sequence[str] = ("dataset", "model", "variant", "method", "alpha"),
) -> List[Dict[str, object]]:
    """Mean and standard deviation across seeds for every numeric metric.

    Rows are grouped by ``group_keys`` (in first-seen order) and each numeric
    metric becomes ``<metric>_mean`` / ``<metric>_std``.  The standard
    deviation uses ``ddof=1`` and is ``nan`` for a single seed.
    """
    if not rows:
        return []
    group_keys = tuple(group_keys)
    metric_keys = _numeric_metric_keys(rows, group_keys)

    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_keys)
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, object]] = []
    for key, members in groups.items():
        entry: Dict[str, object] = dict(zip(group_keys, key))
        entry["n_seeds"] = len({member.get("seed") for member in members})
        for metric in metric_keys:
            values = [
                float(member[metric])
                for member in members
                if isinstance(member.get(metric), (int, float, np.integer, np.floating))
                and not isinstance(member.get(metric), bool)
                and not math.isnan(float(member[metric]))
            ]
            if not values:
                entry[f"{metric}_mean"] = float("nan")
                entry[f"{metric}_std"] = float("nan")
                continue
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        summary.append(entry)

    summary.sort(
        key=lambda entry: tuple(
            (entry.get(column) is None, str(type(entry.get(column))), entry.get(column))
            for column in group_keys
        )
    )
    return summary
