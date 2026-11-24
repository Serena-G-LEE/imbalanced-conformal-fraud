"""Gradient-boosted fraud models behind one train/predict interface.

The models are trained on the earliest block only; the calibration and test
blocks are never seen during fitting.  Those out-of-sample probabilities are
what the conformal layer consumes, which is what keeps the coverage guarantee
valid.

Post-hoc probability calibration
--------------------------------
A booster trained with ``scale_pos_weight = negatives / positives`` optimises a
re-weighted likelihood, so its output is *not* ``P(y = 1 | x)``: the positives
are over-weighted by that ratio, which inflates every score.  Conformal
prediction only needs a score with the right *order*, but any operational
reading of the probability (Brier score, ECE, expected review cost) needs the
scale to be honest.  Two monotone calibrators are therefore provided:

``platt``
    A one-dimensional logistic regression on the log-odds,
    ``p_cal = sigmoid(A * logit(p) + B)``, fit with Platt's smoothed targets
    ``t+ = (N+ + 1) / (N+ + 2)``, ``t- = 1 / (N- + 2)`` so that a calibration
    block with very few positives does not produce an infinite slope.  ``A`` is
    kept positive: a decreasing map would invert the ranking and destroy the
    meaning of the conformal score.

``isotonic``
    Monotone piecewise-linear regression (pool-adjacent-violators) of the label
    on the raw score, clipped outside the observed score range
    (``out_of_bounds="clip"`` in the scikit-learn sense).

Both are fit on a ``prob_cal`` block carved out of the *training* window
(``src.data.SplitDataset.with_prob_cal_split``) and are therefore independent of
both the model fit and the conformal calibration scores.  Because both maps are
non-decreasing, they never change the ranking of the raw scores; the ablation
they enable is about probability quality (Brier, ECE) and about whether the
score *scale* is what drives the review-queue size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "CALIBRATORS",
    "ModelPrediction",
    "SUPPORTED_MODELS",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "default_variant",
    "fit_calibrator",
    "resolve_scale_pos_weight",
    "train_and_predict",
]

#: Model identifiers accepted by ``train_and_predict``.
SUPPORTED_MODELS = ("xgboost", "lightgbm")

#: Calibrator identifiers accepted by ``train_and_predict``.
CALIBRATORS = ("none", "platt", "isotonic")

#: Probability clip used before taking log-odds, so ``logit`` stays finite.
_EPS = 1e-6

_XGBOOST_DEFAULTS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "verbosity": 0,
}

_LIGHTGBM_DEFAULTS: Dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 400,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_lambda": 1.0,
    "verbose": -1,
}


@dataclass
class ModelPrediction:
    """Out-of-sample fraud probabilities for the calibration and test blocks.

    ``prob_cal`` / ``prob_test`` are the probabilities the conformal layer
    consumes (after calibration when one is requested); the ``raw_*`` fields
    keep the booster output so the ablation can report both.
    """

    model_name: str
    prob_cal: np.ndarray
    prob_test: np.ndarray
    raw_prob_cal: Optional[np.ndarray] = None
    raw_prob_test: Optional[np.ndarray] = None
    prob_prob_cal: Optional[np.ndarray] = None
    raw_prob_prob_cal: Optional[np.ndarray] = None
    calibrator: str = "none"
    variant: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    scale_pos_weight: Optional[float] = None
    train_positive_rate: float = float("nan")
    calibrator_model: Any = None
    model: Any = None

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator != "none"

    def describe(self) -> Dict[str, Any]:
        """JSON-friendly summary of what was fit, for run metadata."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "variant": self.variant,
            "calibrator": self.calibrator,
            "scale_pos_weight": self.scale_pos_weight,
            "train_positive_rate": self.train_positive_rate,
            "n_prob_cal": None if self.raw_prob_prob_cal is None else int(self.raw_prob_prob_cal.size),
        }
        if self.calibrator_model is not None and hasattr(self.calibrator_model, "to_dict"):
            payload["calibrator_fit"] = self.calibrator_model.to_dict()
        return payload


def resolve_scale_pos_weight(y_train, spec) -> Optional[float]:
    """Turn a config value into a concrete ``scale_pos_weight``.

    ``"auto"`` uses the training-set negative/positive ratio (the usual choice
    for tree models on heavily imbalanced data); ``None``/``False``/``0``
    disables re-weighting; a number is passed through unchanged.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"", "none", "false", "off", "0"}:
            return None
        if key != "auto":
            return float(spec)
        target = np.asarray(y_train).ravel().astype(int)
        positives = float(np.sum(target == 1))
        negatives = float(np.sum(target == 0))
        if positives == 0:
            raise ValueError("cannot compute scale_pos_weight='auto' without positive samples")
        return negatives / positives
    value = float(spec)
    if value <= 0.0:
        return None
    return value


def default_variant(model_name: str, scale_pos_weight: Optional[float], calibrator: str) -> str:
    """Conventional ablation label, e.g. ``unweighted_platt``.

    Used only when a config does not spell out ``variant`` itself.
    """
    weight = "weighted" if scale_pos_weight is not None and float(scale_pos_weight) != 1.0 else "unweighted"
    tail = "raw" if calibrator == "none" else calibrator
    return f"{weight}_{tail}"


# ---------------------------------------------------------------------------
# calibrators
# ---------------------------------------------------------------------------


def _logit(prob, eps: float = _EPS) -> np.ndarray:
    """``log(p / (1 - p))`` with the probabilities clipped away from 0 and 1."""
    values = np.clip(np.asarray(prob, dtype=float).ravel(), eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _check_calibration_inputs(probs, labels) -> tuple:
    values = np.asarray(probs, dtype=float).ravel()
    target = np.asarray(labels).ravel().astype(int)
    if values.size != target.size:
        raise ValueError(
            f"calibration scores and labels differ in length: {values.size} != {target.size}"
        )
    if values.size == 0:
        raise ValueError("calibration set is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("calibration scores must lie in [0, 1]")
    if not np.all(np.isin(target, [0, 1])):
        raise ValueError("calibration labels must be binary (0/1)")
    return values, target


class _IdentityCalibrator:
    """``calibrator: none`` — probabilities pass through untouched."""

    name = "none"

    def fit(self, probs, labels) -> "_IdentityCalibrator":
        self.n_fit_ = int(np.asarray(probs).size)
        return self

    def predict(self, probs) -> np.ndarray:
        return np.clip(np.asarray(probs, dtype=float).ravel(), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "none"}


class PlattCalibrator:
    """Platt scaling on the log-odds: ``sigmoid(A * logit(p) + B)``.

    Fit by Newton's method on the smoothed-target logistic loss (Platt, 1999).
    The prior offset ``log((N- + 1) / (N+ + 1))`` is used as the starting point,
    which is what makes the procedure stable when positives are rare.
    """

    name = "platt"

    def __init__(self, max_iter: int = 100, tol: float = 1e-10, ridge: float = 1e-10) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.coef_: float = 1.0
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.monotone_fallback_: bool = False
        self.n_fit_: int = 0

    def fit(self, probs, labels) -> "PlattCalibrator":
        values, target = _check_calibration_inputs(probs, labels)
        n_positive = int(np.sum(target == 1))
        n_negative = int(target.size - n_positive)
        if n_positive == 0 or n_negative == 0:
            raise ValueError(
                "Platt scaling needs both classes in the calibration block; "
                f"got {n_positive} positives and {n_negative} negatives"
            )

        z = _logit(values)
        # Platt's smoothed targets keep the fit finite for tiny positive counts.
        smoothed = np.where(target == 1, (n_positive + 1.0) / (n_positive + 2.0), 1.0 / (n_negative + 2.0))
        prior_offset = math.log((n_negative + 1.0) / (n_positive + 1.0))

        slope, intercept = 1.0, prior_offset

        def objective(a: float, b: float) -> float:
            p = _sigmoid(a * z + b)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return float(-np.sum(smoothed * np.log(p) + (1.0 - smoothed) * np.log(1.0 - p)))

        current = objective(slope, intercept)
        for iteration in range(1, self.max_iter + 1):
            p = _sigmoid(slope * z + intercept)
            residual = p - smoothed
            weight = p * (1.0 - p)
            grad_slope = float(np.dot(z, residual))
            grad_intercept = float(np.sum(residual))
            h_ss = float(np.dot(z * z, weight)) + self.ridge
            h_si = float(np.dot(z, weight))
            h_ii = float(np.sum(weight)) + self.ridge
            determinant = h_ss * h_ii - h_si * h_si
            if determinant <= 0.0 or not np.isfinite(determinant):
                break
            step_slope = (h_ii * grad_slope - h_si * grad_intercept) / determinant
            step_intercept = (h_ss * grad_intercept - h_si * grad_slope) / determinant
            if not (np.isfinite(step_slope) and np.isfinite(step_intercept)):
                break
            if max(abs(grad_slope), abs(grad_intercept)) < self.tol:
                self.n_iter_ = iteration - 1
                break

            # Backtracking: a full Newton step can overshoot on near-separable data.
            scale = 1.0
            for _ in range(30):
                candidate_slope = slope - scale * step_slope
                candidate_intercept = intercept - scale * step_intercept
                candidate = objective(candidate_slope, candidate_intercept)
                if np.isfinite(candidate) and candidate <= current:
                    slope, intercept, current = candidate_slope, candidate_intercept, candidate
                    break
                scale *= 0.5
            else:  # pragma: no cover - numerical corner
                break
            if scale * max(abs(step_slope), abs(step_intercept)) < self.tol:
                self.n_iter_ = iteration
                break
        else:  # pragma: no cover - loop exhaustion
            self.n_iter_ = self.max_iter

        if not (np.isfinite(slope) and np.isfinite(intercept)) or slope <= 0.0:
            # A non-increasing map would invert the ranking; fall back to the
            # prior-shift-only map and record that it happened.
            slope, intercept = 1.0, prior_offset
            self.monotone_fallback_ = True

        self.coef_ = float(slope)
        self.intercept_ = float(intercept)
        self.n_fit_ = int(target.size)
        return self

    def predict(self, probs) -> np.ndarray:
        z = _logit(probs)
        return np.clip(_sigmoid(self.coef_ * z + self.intercept_), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "platt",
            "coef": self.coef_,
            "intercept": self.intercept_,
            "n_iter": self.n_iter_,
            "n_fit": self.n_fit_,
            "monotone_fallback": self.monotone_fallback_,
        }


class IsotonicCalibrator:
    """Isotonic regression of the label on the raw score (PAV), clipped at the ends.

    Duplicated scores are pooled first, the pool-adjacent-violators algorithm
    makes the fit non-decreasing, and predictions interpolate linearly between
    the fitted points, clipped outside the observed score range — the same
    behaviour as scikit-learn's ``IsotonicRegression(out_of_bounds="clip")``,
    without the dependency.
    """

    name = "isotonic"

    def __init__(self) -> None:
        self.x_: np.ndarray = np.empty(0, dtype=float)
        self.y_: np.ndarray = np.empty(0, dtype=float)
        self.n_fit_: int = 0

    def fit(self, probs, labels) -> "IsotonicCalibrator":
        values, target = _check_calibration_inputs(probs, labels)

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_target = target[order].astype(float)

        unique, starts, counts = np.unique(
            sorted_values, return_index=True, return_counts=True
        )
        sums = np.add.reduceat(sorted_target, starts)

        means = [float(sums[i] / counts[i]) for i in range(unique.size)]
        weights = [float(counts[i]) for i in range(unique.size)]
        # Pool adjacent violators (stack form).  Each open block remembers the
        # range of unique scores it covers, so the fitted value can be written
        # back onto every score after the merging has settled.
        blocks: list = []  # [start_index, end_index, weighted_sum, weight]
        for position in range(unique.size):
            blocks.append(
                [position, position, float(sums[position]), float(counts[position])]
            )
            while len(blocks) > 1:
                left = blocks[-2]
                right = blocks[-1]
                if left[2] / left[3] <= right[2] / right[3]:
                    break
                blocks.pop()
                left = blocks.pop()
                blocks.append(
                    [left[0], right[1], left[2] + right[2], left[3] + right[3]]
                )

        fitted = np.empty(unique.size, dtype=float)
        for start, end, total, weight in blocks:
            fitted[start : end + 1] = total / weight

        self.x_ = np.asarray(unique, dtype=float)
        self.y_ = np.clip(fitted, 0.0, 1.0)
        self.n_fit_ = int(target.size)
        return self

    def predict(self, probs) -> np.ndarray:
        if self.x_.size == 0:
            raise RuntimeError("call fit() before predict()")
        values = np.clip(np.asarray(probs, dtype=float).ravel(), 0.0, 1.0)
        if self.x_.size == 1:
            return np.full(values.shape, float(self.y_[0]))
        # np.interp already clips outside [x_[0], x_[-1]] to the end values.
        return np.clip(np.interp(values, self.x_, self.y_), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "isotonic",
            "n_knots": int(self.x_.size),
            "n_fit": self.n_fit_,
            "score_range": [float(self.x_[0]), float(self.x_[-1])] if self.x_.size else [],
        }


def fit_calibrator(name: str, probs, labels):
    """Fit the calibrator named ``name`` on ``(probs, labels)``."""
    key = str(name).strip().lower()
    if key in {"", "none", "identity"}:
        return _IdentityCalibrator().fit(probs, labels)
    if key == "platt":
        return PlattCalibrator().fit(probs, labels)
    if key == "isotonic":
        return IsotonicCalibrator().fit(probs, labels)
    raise ValueError(f"unsupported calibrator {name!r}; expected one of {CALIBRATORS}")


# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------


def _check_inputs(X_train, y_train, X_cal, X_test) -> None:
    train = np.asarray(X_train)
    cal = np.asarray(X_cal)
    test = np.asarray(X_test)
    if train.ndim != 2 or cal.ndim != 2 or test.ndim != 2:
        raise ValueError("feature matrices must be 2-D")
    if train.shape[1] != cal.shape[1] or train.shape[1] != test.shape[1]:
        raise ValueError(
            "feature count differs between splits: "
            f"{train.shape[1]}, {cal.shape[1]}, {test.shape[1]}"
        )
    if len(y_train) != train.shape[0] or len(y_train) == 0:
        raise ValueError("y_train does not match X_train")


def _check_prob_cal_block(X_train, X_prob_cal, y_prob_cal, X_cal, calibrator: str) -> None:
    """Validate the probability-calibration block, including the leakage guard."""
    if X_prob_cal is None or y_prob_cal is None:
        if calibrator != "none":
            raise ValueError(
                f"calibrator={calibrator!r} requires a separate prob_cal block; build one with "
                "SplitDataset.with_prob_cal_split() instead of reusing X_cal"
            )
        return

    prob_cal = np.asarray(X_prob_cal)
    if prob_cal.ndim != 2:
        raise ValueError("X_prob_cal must be 2-D")
    if prob_cal.shape[1] != np.asarray(X_train).shape[1]:
        raise ValueError(
            f"X_prob_cal has {prob_cal.shape[1]} features, X_train has {np.asarray(X_train).shape[1]}"
        )
    target = np.asarray(y_prob_cal).ravel()
    if target.size != prob_cal.shape[0]:
        raise ValueError("y_prob_cal does not match X_prob_cal")
    if target.size == 0:
        raise ValueError("prob_cal block is empty")

    # The whole point of a separate block is that it is not the conformal
    # calibration set.  Catch the wiring mistake before anything is fit: an
    # identical matrix means the caller passed the same rows twice.
    cal = np.asarray(X_cal)
    if cal.shape == prob_cal.shape and np.array_equal(prob_cal, cal):
        raise ValueError(
            "X_prob_cal is identical to X_cal: fitting the calibrator on the conformal "
            "calibration set would break exchangeability and leak information"
        )


def train_and_predict(
    model_name: str,
    X_train,
    y_train,
    X_cal,
    X_test,
    *,
    params: Optional[Dict[str, Any]] = None,
    scale_pos_weight: Any = None,
    seed: int = 42,
    num_threads: Optional[int] = None,
    calibrator: str = "none",
    X_prob_cal=None,
    y_prob_cal=None,
    variant: Optional[str] = None,
) -> ModelPrediction:
    """Fit ``model_name`` on the training block and score calibration/test.

    Parameters
    ----------
    model_name:
        ``"xgboost"`` or ``"lightgbm"``.
    params:
        Estimator keyword arguments merged over the defaults in this module.
    scale_pos_weight:
        ``"auto"`` (ratio of negatives to positives in the training block), a
        positive number, or ``None`` to train without re-weighting.
    seed:
        Random seed for the booster.
    num_threads:
        Threads for the booster; ``None`` lets the library decide.
    calibrator:
        ``"none"``, ``"platt"`` or ``"isotonic"``.  A non-``"none"`` value
        requires ``X_prob_cal`` / ``y_prob_cal`` — a held-out slice of the
        *training* window, never the conformal calibration block.
    variant:
        Label recorded on the result; defaults to ``weighted_raw``,
        ``unweighted_platt`` ... derived from the two settings above.

    Returns
    -------
    ModelPrediction
        With ``prob_cal`` and ``prob_test`` holding the (calibrated)
        ``P(y = 1 | x)`` that the conformal layer consumes, and ``raw_*``
        fields holding the uncalibrated booster output.
    """
    _check_inputs(X_train, y_train, X_cal, X_test)
    name = str(model_name).strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model_name!r}; expected one of {SUPPORTED_MODELS}")
    calibrator_key = str(calibrator or "none").strip().lower()
    if calibrator_key in {"", "identity"}:
        calibrator_key = "none"
    if calibrator_key not in CALIBRATORS:
        raise ValueError(f"unsupported calibrator {calibrator!r}; expected one of {CALIBRATORS}")
    _check_prob_cal_block(X_train, X_prob_cal, y_prob_cal, X_cal, calibrator_key)

    target = np.asarray(y_train).ravel().astype(int)
    weight = resolve_scale_pos_weight(target, scale_pos_weight)
    effective_params: Dict[str, Any] = {}

    if name == "xgboost":
        from xgboost import XGBClassifier

        effective_params = dict(_XGBOOST_DEFAULTS)
        effective_params.update(params or {})
        if weight is not None:
            effective_params["scale_pos_weight"] = float(weight)
        effective_params.setdefault("n_jobs", num_threads if num_threads is not None else -1)
        effective_params.setdefault("random_state", seed)
        estimator = XGBClassifier(**effective_params)
    else:
        from lightgbm import LGBMClassifier

        effective_params = dict(_LIGHTGBM_DEFAULTS)
        effective_params.update(params or {})
        if weight is not None:
            effective_params["scale_pos_weight"] = float(weight)
        effective_params.setdefault("n_jobs", num_threads if num_threads is not None else -1)
        effective_params.setdefault("random_state", seed)
        estimator = LGBMClassifier(**effective_params)

    estimator.fit(np.asarray(X_train), target)

    def score(matrix) -> np.ndarray:
        if matrix is None:
            return None  # type: ignore[return-value]
        return np.clip(
            np.asarray(estimator.predict_proba(np.asarray(matrix)))[:, 1].astype(np.float64),
            0.0,
            1.0,
        )

    raw_prob_prob_cal = score(X_prob_cal)
    raw_prob_cal = score(X_cal)
    raw_prob_test = score(X_test)

    calibrator_model = None
    prob_prob_cal: Optional[np.ndarray] = None
    if calibrator_key != "none":
        calibrator_model = fit_calibrator(calibrator_key, raw_prob_prob_cal, y_prob_cal)
        prob_prob_cal = calibrator_model.predict(raw_prob_prob_cal)
        prob_cal = calibrator_model.predict(raw_prob_cal)
        prob_test = calibrator_model.predict(raw_prob_test)
    else:
        prob_prob_cal = raw_prob_prob_cal
        prob_cal = raw_prob_cal
        prob_test = raw_prob_test

    label = variant or default_variant(name, weight, calibrator_key)

    return ModelPrediction(
        model_name=name,
        prob_cal=np.asarray(prob_cal, dtype=np.float64),
        prob_test=np.asarray(prob_test, dtype=np.float64),
        raw_prob_cal=np.asarray(raw_prob_cal, dtype=np.float64),
        raw_prob_test=np.asarray(raw_prob_test, dtype=np.float64),
        prob_prob_cal=None if prob_prob_cal is None else np.asarray(prob_prob_cal, dtype=np.float64),
        raw_prob_prob_cal=(
            None if raw_prob_prob_cal is None else np.asarray(raw_prob_prob_cal, dtype=np.float64)
        ),
        calibrator=calibrator_key,
        variant=str(label),
        params=effective_params,
        scale_pos_weight=weight,
        train_positive_rate=float(np.mean(target == 1)),
        calibrator_model=calibrator_model,
        model=estimator,
    )
