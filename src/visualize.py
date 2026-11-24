"""Plots for the alpha trade-off between fraud coverage and review cost."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

__all__ = ["plot_alpha_tradeoff", "plot_alpha_variant_tradeoff"]

_METHOD_LABELS = {"standard": "Standard split CP", "mondrian": "Mondrian (class-conditional) CP"}
_METHOD_COLORS = {"standard": "#c0392b", "mondrian": "#1f6fb2"}

#: Colour cycle for ablation variants (kept stable so figures stay comparable
#: across reruns).
_VARIANT_COLORS = (
    "#1f6fb2",
    "#c0392b",
    "#2e8b57",
    "#8e44ad",
    "#d68910",
    "#117a65",
    "#7f8c8d",
)
_VARIANT_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def _variant_linestyle(variant: str) -> str:
    """Dash the calibrated arms so an unchanged curve stays visible.

    ``*_platt`` and ``*_isotonic`` are drawn dashed / dotted: when a calibrated
    arm sits exactly on top of its raw twin (the structural invariance of
    Mondrian CP under a strictly monotone map) the overlap is still readable.
    """
    name = variant.lower()
    if name.endswith("_platt"):
        return "--"
    if name.endswith("_isotonic"):
        return ":"
    return "-"


def _import_pyplot():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for figures; install it with "
            "`pip install -r requirements.txt` or run with --no-figures"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _series(summary_rows: Sequence[Dict[str, object]], method: str, metric: str):
    points = sorted(
        (row for row in summary_rows if row.get("method") == method),
        key=lambda row: float(row["alpha"]),
    )
    alphas = [float(row["alpha"]) for row in points]
    means = [float(row[f"{metric}_mean"]) for row in points]
    stds = []
    for row in points:
        value = row.get(f"{metric}_std")
        stds.append(float(value) if isinstance(value, (int, float)) else 0.0)
    return alphas, means, stds


def _variant_series(
    summary_rows: Sequence[Dict[str, object]], method: str, variant: str, metric: str
):
    """``(alphas, means, stds)`` for one ``(method, variant)`` cell."""
    points = sorted(
        (
            row
            for row in summary_rows
            if row.get("method") == method and row.get("variant") == variant
        ),
        key=lambda row: float(row["alpha"]),
    )
    alphas = [float(row["alpha"]) for row in points]
    means = [float(row[f"{metric}_mean"]) for row in points]
    stds = []
    for row in points:
        value = row.get(f"{metric}_std")
        stds.append(float(value) if isinstance(value, (int, float)) else 0.0)
    return alphas, means, stds


def plot_alpha_variant_tradeoff(
    summary_rows: Sequence[Dict[str, object]],
    output_dir: str | Path,
    *,
    dataset: str,
    model: str,
    method: str = "mondrian",
    variants: Sequence[str] | None = None,
    prefix: str = "",
) -> List[Path]:
    """Compare ablation variants on fraud coverage and manual-review rate.

    One line per ``variant`` (``weighted_raw``, ``unweighted_raw``,
    ``unweighted_platt``, ``unweighted_isotonic`` ...) at a fixed calibration
    scheme, so the figure isolates what re-weighting and probability
    calibration change: left panel is empirical fraud coverage against the
    nominal ``1 - alpha`` line, right panel is the share of transactions in
    ``{0, 1}``.  Error bars are the standard deviation across seeds.

    The file is named ``<dataset>_<model>_ablation_tradeoff.png``.
    """
    plt = _import_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if variants is None:
        variants = sorted(
            {
                str(row.get("variant"))
                for row in summary_rows
                if row.get("method") == method and row.get("variant") is not None
            }
        )
    variants = [str(variant) for variant in variants]
    if not variants:
        return []

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    coverage_axis, triage_axis = axes

    for position, variant in enumerate(variants):
        color = _VARIANT_COLORS[position % len(_VARIANT_COLORS)]
        marker = _VARIANT_MARKERS[position % len(_VARIANT_MARKERS)]
        linestyle = _variant_linestyle(variant)

        alphas, means, stds = _variant_series(summary_rows, method, variant, "coverage_fraud")
        if alphas:
            coverage_axis.errorbar(
                alphas, means, yerr=stds, marker=marker, markersize=5, capsize=3,
                linewidth=1.6, color=color, linestyle=linestyle, label=variant,
            )
        alphas, means, stds = _variant_series(summary_rows, method, variant, "triage_rate")
        if alphas:
            triage_axis.errorbar(
                alphas, means, yerr=stds, marker=marker, markersize=5, capsize=3,
                linewidth=1.6, color=color, linestyle=linestyle, label=variant,
            )

    all_alphas = sorted({
        float(row["alpha"])
        for row in summary_rows
        if row.get("method") == method
    })
    if all_alphas:
        coverage_axis.plot(
            all_alphas, [1.0 - alpha for alpha in all_alphas],
            linestyle="--", linewidth=1.1, color="black", label="nominal 1 - alpha",
        )

    coverage_axis.set_title("Fraud coverage (class-conditional)")
    coverage_axis.set_xlabel("alpha (target miscoverage)")
    coverage_axis.set_ylabel("empirical fraud coverage")
    coverage_axis.set_ylim(0.0, 1.02)

    triage_axis.set_title("Manual review queue")
    triage_axis.set_xlabel("alpha (target miscoverage)")
    triage_axis.set_ylabel("share of transactions in {0,1}")
    triage_axis.set_ylim(bottom=0.0)

    for axis in axes:
        if all_alphas:
            axis.set_xscale("log")
            axis.set_xticks(all_alphas)
            axis.set_xticklabels([f"{alpha:g}" for alpha in all_alphas])
            axis.minorticks_off()
        axis.grid(alpha=0.25, linestyle=":")
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        f"{dataset}  |  {model}  |  {_METHOD_LABELS.get(method, method)}  |  calibration ablation",
        fontsize=10,
    )
    figure.tight_layout()

    path = output_dir / f"{prefix}{dataset}_{model}_ablation_tradeoff.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [path]


def plot_alpha_tradeoff(
    summary_rows: Sequence[Dict[str, object]],
    output_dir: str | Path,
    *,
    dataset: str,
    model: str,
    methods: Sequence[str] = ("standard", "mondrian"),
    prefix: str = "",
) -> List[Path]:
    """Plot fraud coverage and manual-review rate against ``alpha``.

    Left panel: class-conditional fraud coverage with the nominal ``1 - alpha``
    reference line.  Right panel: share of transactions routed to the manual
    review queue (``{0, 1}``).  Error bars are the standard deviation across
    seeds.  One PNG is written per model and returned as a one-element list.
    """
    plt = _import_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    coverage_axis, triage_axis = axes

    for method in methods:
        label = _METHOD_LABELS.get(method, method)
        color = _METHOD_COLORS.get(method)

        alphas, means, stds = _series(summary_rows, method, "coverage_fraud")
        if alphas:
            coverage_axis.errorbar(
                alphas, means, yerr=stds, marker="o", markersize=5, capsize=3,
                linewidth=1.6, color=color, label=label,
            )
        alphas, means, stds = _series(summary_rows, method, "triage_rate")
        if alphas:
            triage_axis.errorbar(
                alphas, means, yerr=stds, marker="s", markersize=5, capsize=3,
                linewidth=1.6, color=color, label=label,
            )

    all_alphas = sorted({float(row["alpha"]) for row in summary_rows})
    if all_alphas:
        coverage_axis.plot(
            all_alphas, [1.0 - alpha for alpha in all_alphas],
            linestyle="--", linewidth=1.1, color="black", label="nominal 1 - alpha",
        )

    coverage_axis.set_title("Fraud coverage (class-conditional)")
    coverage_axis.set_xlabel("alpha (target miscoverage)")
    coverage_axis.set_ylabel("empirical fraud coverage")
    coverage_axis.set_ylim(0.0, 1.02)

    triage_axis.set_title("Manual review queue")
    triage_axis.set_xlabel("alpha (target miscoverage)")
    triage_axis.set_ylabel("share of transactions in {0,1}")
    triage_axis.set_ylim(bottom=0.0)

    for axis in axes:
        if all_alphas:
            axis.set_xscale("log")
            axis.set_xticks(all_alphas)
            axis.set_xticklabels([f"{alpha:g}" for alpha in all_alphas])
            axis.minorticks_off()
        axis.grid(alpha=0.25, linestyle=":")
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle(f"{dataset}  |  {model}", fontsize=11)
    figure.tight_layout()

    path = output_dir / f"{prefix}{dataset}_{model}_alpha_tradeoff.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [path]
