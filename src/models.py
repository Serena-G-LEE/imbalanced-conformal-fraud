"""Gradient-boosted fraud models behind one train/predict interface.

The models are trained on the earliest block only; the calibration and test
blocks are never seen during fitting.  Those out-of-sample probabilities are
what the conformal layer consumes, which is what keeps the coverage guarantee
valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["ModelPrediction", "SUPPORTED_MODELS", "resolve_scale_pos_weight", "train_and_predict"]

#: Model identifiers accepted by ``train_and_predict``.
SUPPORTED_MODELS = ("xgboost", "lightgbm")

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
    """Out-of-sample fraud probabilities for the calibration and test blocks."""

    model_name: str
    prob_cal: np.ndarray
    prob_test: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)
    scale_pos_weight: Optional[float] = None
    train_positive_rate: float = float("nan")
    model: Any = None


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

    Returns
    -------
    ModelPrediction
        With ``prob_cal`` and ``prob_test`` holding ``P(y = 1 | x)``.
    """
    _check_inputs(X_train, y_train, X_cal, X_test)
    name = str(model_name).strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model_name!r}; expected one of {SUPPORTED_MODELS}")

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

    prob_cal = np.asarray(estimator.predict_proba(np.asarray(X_cal)))[:, 1].astype(np.float64)
    prob_test = np.asarray(estimator.predict_proba(np.asarray(X_test)))[:, 1].astype(np.float64)

    return ModelPrediction(
        model_name=name,
        prob_cal=np.clip(prob_cal, 0.0, 1.0),
        prob_test=np.clip(prob_test, 0.0, 1.0),
        params=effective_params,
        scale_pos_weight=weight,
        train_positive_rate=float(np.mean(target == 1)),
        model=estimator,
    )
