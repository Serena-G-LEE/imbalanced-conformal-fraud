"""Split conformal prediction for binary fraud triage.

Two calibration schemes are provided.

``StandardSplitCP``
    One nonconformity threshold is calibrated on the pooled calibration set.
    This yields the classical marginal guarantee
    ``P(Y in C(X)) >= 1 - alpha`` but makes no statement about coverage inside a
    class.  Under extreme imbalance the pooled quantile is dominated by the
    majority (legitimate) class, so fraud coverage can fall far below the
    nominal level even though marginal coverage looks healthy.

``MondrianCP``
    Class-conditional (Mondrian) calibration: a separate threshold is
    calibrated inside each label stratum.  This restores the per-class
    guarantee ``P(Y in C(X) | Y = y) >= 1 - alpha`` for every ``y``, keeping
    fraud coverage at the nominal level at the price of a larger ambiguous
    set (the manual review queue).

Nonconformity score
-------------------
``s(x, y) = 1 - p_hat(y | x)``

with ``p_hat(1 | x)`` the model probability of fraud and
``p_hat(0 | x) = 1 - p_hat(1 | x)``.

Calibration
-----------
For a stratum of ``n`` calibration scores, the finite-sample corrected
quantile is the ``k``-th smallest score with

``k = ceil((n + 1) * (1 - alpha))``

and the threshold is clipped to ``1.0`` when ``k > n`` (which can only happen
for very small strata).  An empty stratum (e.g. no fraud in the calibration
window) falls back to ``1.0``, i.e. the label is never excluded and every
transaction is at least routed to review; this is reported through
``describe()`` so the situation is visible rather than silently ignored.

Prediction sets
---------------
``C(x) = { y in {0, 1} : s(x, y) <= q_y }``

which yields exactly four outcomes:

===========  =========================  =======================
code         set                        operational action
===========  =========================  =======================
``SET_NORMAL``  ``{0}``                 auto-approve
``SET_FRAUD``   ``{1}``                 auto-decline / block
``SET_BOTH``    ``{0, 1}``              manual review (triage)
``SET_EMPTY``   ``{}``                  neither label plausible
===========  =========================  =======================

Prediction sets are held internally as a boolean membership matrix of shape
``(n_samples, 2)`` whose column index is the class label; ``codes_*`` helpers
convert between that matrix and the integer codes above.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

__all__ = [
    "SET_NORMAL",
    "SET_FRAUD",
    "SET_BOTH",
    "SET_EMPTY",
    "SET_NAMES",
    "conformal_quantile",
    "nonconformity_scores",
    "codes_from_membership",
    "codes_to_membership",
    "StandardSplitCP",
    "MondrianCP",
]

#: Prediction set codes, ordered by increasing ambiguity.
SET_NORMAL = 0  # {0}
SET_FRAUD = 1  # {1}
SET_BOTH = 2  # {0, 1} -> manual review
SET_EMPTY = 3  # {}     -> no label plausible

SET_NAMES: Dict[int, str] = {
    SET_NORMAL: "{0}",
    SET_FRAUD: "{1}",
    SET_BOTH: "{0,1}",
    SET_EMPTY: "{}",
}

LABELS = (0, 1)


def _as_1d(values, name: str, dtype=float) -> np.ndarray:
    """Return ``values`` as a C-contiguous 1-D array of the requested dtype."""
    array = np.asarray(values, dtype=dtype).ravel()
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _check_alpha(alpha: float) -> float:
    value = float(alpha)
    if not (0.0 < value < 1.0) or math.isnan(value):
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    return value


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample corrected ``(1 - alpha)`` quantile of nonconformity scores.

    Parameters
    ----------
    scores:
        Nonconformity scores of a calibration stratum.  An empty stratum
        returns ``1.0`` (the label is never excluded).
    alpha:
        Target miscoverage rate in ``(0, 1)``.

    Returns
    -------
    float
        ``k``-th smallest score with ``k = ceil((n + 1) * (1 - alpha))``,
        clipped to ``1.0`` when ``k > n``.

    Notes
    -----
    Ties are resolved by the observed order statistic (no randomisation), so
    the returned threshold is never smaller than the exact split-conformal
    quantile.  This keeps the coverage guarantee valid and makes every run
    deterministic.
    """
    alpha = _check_alpha(alpha)
    array = np.asarray(scores, dtype=float).ravel()
    n = array.size
    if n == 0:
        return 1.0
    if not np.all(np.isfinite(array)):
        raise ValueError("nonconformity scores must be finite")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return 1.0
    rank = max(rank, 1)
    return float(np.partition(array, rank - 1)[rank - 1])


def nonconformity_scores(prob_positive, labels) -> np.ndarray:
    """``1 - p_hat(y | x)`` for the observed label of every sample."""
    prob = np.asarray(prob_positive, dtype=float).ravel()
    target = np.asarray(labels).ravel().astype(int)
    if prob.size != target.size:
        raise ValueError(
            f"prob_positive and labels differ in length: {prob.size} != {target.size}"
        )
    if not np.all(np.isin(target, LABELS)):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all((prob >= 0.0) & (prob <= 1.0)):
        raise ValueError("prob_positive must lie in [0, 1]")
    return np.where(target == 1, 1.0 - prob, prob)


def codes_from_membership(membership) -> np.ndarray:
    """Convert a boolean ``(n, 2)`` membership matrix into prediction set codes."""
    matrix = np.asarray(membership, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("membership must have shape (n_samples, 2)")
    codes = np.full(matrix.shape[0], SET_EMPTY, dtype=np.int64)
    include_normal = matrix[:, 0]
    include_fraud = matrix[:, 1]
    codes[include_normal & ~include_fraud] = SET_NORMAL
    codes[~include_normal & include_fraud] = SET_FRAUD
    codes[include_normal & include_fraud] = SET_BOTH
    return codes


def codes_to_membership(codes) -> np.ndarray:
    """Convert prediction set codes into a boolean ``(n, 2)`` membership matrix."""
    array = np.asarray(codes).ravel().astype(int)
    unknown = set(np.unique(array).tolist()) - set(SET_NAMES)
    if unknown:
        raise ValueError(f"unknown prediction set codes: {sorted(unknown)}")
    membership = np.zeros((array.size, 2), dtype=bool)
    membership[:, 0] = (array == SET_NORMAL) | (array == SET_BOTH)
    membership[:, 1] = (array == SET_FRAUD) | (array == SET_BOTH)
    return membership


class _BaseSplitCP:
    """Shared calibration and prediction-set logic for two-label split CP."""

    name = "base"

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = _check_alpha(alpha)
        self.q_: Optional[np.ndarray] = None
        self.stratum_sizes_: Optional[np.ndarray] = None

    # -- calibration ----------------------------------------------------
    def _calibration_strata(self, prob: np.ndarray, target: np.ndarray) -> Dict[int, np.ndarray]:
        scores = nonconformity_scores(prob, target)
        return {label: scores[target == label] for label in LABELS}

    def calibrate(self, prob_positive, labels) -> "_BaseSplitCP":
        """Fit the score thresholds on a held-out calibration set."""
        prob = np.asarray(prob_positive, dtype=float).ravel()
        target = np.asarray(labels).ravel().astype(int)
        if prob.size != target.size:
            raise ValueError(
                f"prob_positive and labels differ in length: {prob.size} != {target.size}"
            )
        if prob.size == 0:
            raise ValueError("calibration set is empty")
        strata = self._calibration_strata(prob, target)
        self.q_ = np.array(
            [conformal_quantile(strata[label], self.alpha) for label in LABELS],
            dtype=float,
        )
        self.stratum_sizes_ = np.array([strata[label].size for label in LABELS], dtype=int)
        return self

    @property
    def is_calibrated(self) -> bool:
        return self.q_ is not None

    def _require_calibration(self) -> None:
        if self.q_ is None or self.stratum_sizes_ is None:
            raise RuntimeError("call calibrate() before predicting")

    # -- prediction -----------------------------------------------------
    def predict_set_matrix(self, prob_positive) -> np.ndarray:
        """Boolean membership matrix ``(n, 2)`` for the prediction sets."""
        self._require_calibration()
        prob = np.asarray(prob_positive, dtype=float).ravel()
        if not np.all((prob >= 0.0) & (prob <= 1.0)):
            raise ValueError("prob_positive must lie in [0, 1]")
        membership = np.empty((prob.size, 2), dtype=bool)
        # score for y = 0 is p, score for y = 1 is 1 - p
        membership[:, 0] = prob <= self.q_[0]
        membership[:, 1] = (1.0 - prob) <= self.q_[1]
        return membership

    def predict_set_codes(self, prob_positive) -> np.ndarray:
        """Prediction set code per sample (see ``SET_*`` constants)."""
        return codes_from_membership(self.predict_set_matrix(prob_positive))

    def set_sizes(self, prob_positive) -> np.ndarray:
        """Cardinality of each prediction set."""
        return self.predict_set_matrix(prob_positive).sum(axis=1)

    # -- reporting ------------------------------------------------------
    def describe(self) -> Dict[str, float]:
        """Human readable calibration summary, suitable for a CSV row."""
        self._require_calibration()
        return {
            "method": self.name,
            "alpha": self.alpha,
            "n_cal_normal": int(self.stratum_sizes_[0]),
            "n_cal_fraud": int(self.stratum_sizes_[1]),
            "q_normal": float(self.q_[0]),
            "q_fraud": float(self.q_[1]),
            "min_prob_positive_for_fraud": float(max(0.0, 1.0 - self.q_[1])),
            "max_prob_positive_for_normal": float(min(1.0, self.q_[0])),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        status = "uncalibrated" if self.q_ is None else f"q={np.round(self.q_, 4).tolist()}"
        return f"{type(self).__name__}(alpha={self.alpha}, {status})"


class StandardSplitCP(_BaseSplitCP):
    """Split conformal prediction with one threshold pooled over both classes."""

    name = "standard"

    def _calibration_strata(self, prob: np.ndarray, target: np.ndarray) -> Dict[int, np.ndarray]:
        scores = nonconformity_scores(prob, target)
        return {label: scores for label in LABELS}


class MondrianCP(_BaseSplitCP):
    """Class-conditional (Mondrian) split conformal prediction."""

    name = "mondrian"
