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

``AdaptiveMondrianCP``
    Class-conditional *adaptive* conformal inference (ACI, Gibbs & Candes
    2021): the two thresholds are still the class-conditional quantiles, but the
    miscoverage level driving each of them moves online.  At every step the
    realised label updates only its own class' level, so the levels track the
    drift that breaks exchangeability across a chronological test stream
    instead of staying at the value calibrated once inside the calibration
    window.  The level update uses labels strictly after the prediction set has
    been formed, which is the causal order of a deployed fraud system: the
    chargeback for transaction ``t`` arrives after the decision on ``t``.

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
    "AdaptiveMondrianCP",
    "AdaptiveStreamResult",
    "predict_adaptive_mondrian",
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

#: Smallest / largest adaptive level that is still a usable quantile level.
#: ACI projects the level back onto ``[0, 1]``; the quantile itself needs a
#: level strictly inside ``(0, 1)``, so the projection keeps a numerical margin.
ADAPTIVE_ALPHA_FLOOR = 1e-4
ADAPTIVE_ALPHA_CEIL = 1.0 - 1e-4


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


def _check_gamma(gamma: float) -> float:
    value = float(gamma)
    if not (0.0 < value < 1.0) or math.isnan(value):
        raise ValueError(
            f"ACI step size gamma must lie strictly between 0 and 1, got {gamma!r}"
        )
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
        #: Calibration nonconformity scores per label, kept so an adaptive
        #: scheme can query the quantile at a level that moves online.
        self.scores_: Optional[Dict[int, np.ndarray]] = None

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
        self.scores_ = strata
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

    def predict_set_codes(self, prob_positive, labels=None) -> np.ndarray:
        """Prediction set code per sample (see ``SET_*`` constants).

        ``labels`` is accepted for interface symmetry with the adaptive scheme
        (which consumes the realised label stream); the static schemes ignore
        it, because their thresholds do not move after calibration.
        """
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


class _OrderStatisticQuantile:
    """Cheap repeated lookup of the finite-sample conformal quantile.

    ACI queries the same calibration stratum many times, at a level that
    changes by at most ``gamma`` per step.  Sorting the stratum once and
    resolving the rank

    ``k = ceil((n + 1)(1 - alpha))``    (``q = 1.0`` when ``k > n``)

    on demand turns every query into an array index, and caching by rank means
    a run whose level stays on the same rank does no work at all.  The rule is
    exactly the one in :func:`conformal_quantile`, which a unit test pins.
    """

    __slots__ = ("values", "n", "_by_rank")

    def __init__(self, scores) -> None:
        values = np.asarray(scores, dtype=float).ravel()
        if values.size and not np.all(np.isfinite(values)):
            raise ValueError("nonconformity scores must be finite")
        self.values = np.sort(values)
        self.n = int(values.size)
        self._by_rank: Dict[int, float] = {}

    def at(self, alpha: float) -> float:
        if self.n == 0:
            return 1.0
        rank = int(math.ceil((self.n + 1) * (1.0 - float(alpha))))
        if rank > self.n:
            return 1.0
        rank = max(rank, 1)
        value = self._by_rank.get(rank)
        if value is None:
            value = float(self.values[rank - 1])
            self._by_rank[rank] = value
        return value


class AdaptiveStreamResult:
    """Output of :meth:`AdaptiveMondrianCP.predict_stream`.

    ``codes`` is what the metric layer consumes; the remaining fields expose
    the online state so a figure or a test can check that the level moved for
    the class that produced the error and only for that class.
    """

    __slots__ = ("codes", "labels", "alpha_levels", "alpha_final", "errors")

    def __init__(self, codes, labels, alpha_levels, alpha_final, errors) -> None:
        self.codes = np.asarray(codes, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.alpha_levels = np.asarray(alpha_levels, dtype=float)  # (n, 2) used at step t
        self.alpha_final = np.asarray(alpha_final, dtype=float)  # (2,)
        self.errors = np.asarray(errors, dtype=float)  # (n,)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AdaptiveStreamResult(n={self.codes.size}, "
            f"alpha_final={np.round(self.alpha_final, 5).tolist()})"
        )


class AdaptiveMondrianCP(MondrianCP):
    """Class-conditional adaptive conformal inference (ACI).

    Gibbs & Candes (2021) replace the fixed miscoverage level with a sequence
    that is updated from the observed errors,

    ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)``,
    ``err_t = 1{Y_t not in C_t(X_t)}``,

    so that the long-run error rate converges to ``alpha`` even when
    exchangeability fails.  The class-conditional variant keeps one level per
    label, ``alpha_{t,0}`` and ``alpha_{t,1}``, and moves only the level of the
    class whose label was just observed:

    ``alpha_{t+1, c} = alpha_{t, c} + gamma * (alpha - err_t)``  for ``c = Y_t``,
    ``alpha_{t+1, c'} = alpha_{t, c'}``                          for ``c' != Y_t``.

    Only the observed class may move because only its error is observed: a
    miscoverage of the fraud quantile says nothing about how the legitimate
    quantile should be relaxed, and moving both on every event would let the
    majority class (which supplies almost every label) erase the correction the
    minority class needs.

    Causality
    ---------
    The update uses ``Y_t`` strictly *after* ``C_t`` has been formed, so no
    future label enters any earlier prediction set.  In production the label of
    transaction ``t`` (a chargeback) arrives weeks after the decision, which is
    exactly this order with a delay; the delay is only assumed to be finite and
    ordered, not to be zero.

    Levels are projected onto ``[1e-4, 1 - 1e-4]`` (``ADAPTIVE_ALPHA_FLOOR`` /
    ``ADAPTIVE_ALPHA_CEIL``) so the quantile query stays defined; those two
    ends mean "include the class for every transaction" and "keep only the
    most confident transactions" respectively.
    """

    name = "adaptive_mondrian"

    def __init__(self, alpha: float = 0.05, gamma: float = 0.005) -> None:
        super().__init__(alpha=alpha)
        self.gamma = _check_gamma(gamma)
        self.quantiles_: Optional[Dict[int, _OrderStatisticQuantile]] = None
        self.alpha_levels_: Optional[np.ndarray] = None
        self.stream_: Optional[AdaptiveStreamResult] = None

    # -- calibration ----------------------------------------------------
    def calibrate(self, prob_positive, labels) -> "AdaptiveMondrianCP":
        """Fit the class-conditional quantile sources and reset the levels."""
        super().calibrate(prob_positive, labels)
        assert self.scores_ is not None  # set by the base calibrate
        self.quantiles_ = {
            label: _OrderStatisticQuantile(self.scores_[label]) for label in LABELS
        }
        self.alpha_levels_ = np.full(2, self.alpha, dtype=float)
        self.stream_ = None
        return self

    # -- prediction -----------------------------------------------------
    def predict_stream(
        self,
        prob_positive,
        labels,
        *,
        update: bool = True,
    ) -> AdaptiveStreamResult:
        """Run the causal online loop over one chronologically ordered stream.

        Parameters
        ----------
        prob_positive:
            Model probability of fraud for every transaction, in time order.
        labels:
            Realised label of every transaction.  ``labels[t]`` is consumed
            only after ``C_t`` has been emitted.
        update:
            ``False`` freezes the levels at ``alpha`` (a useful control: the
            result then equals static Mondrian CP and the code path is
            verified to have no hidden adaptation).
        """
        if self.quantiles_ is None or self.alpha_levels_ is None:
            raise RuntimeError("call calibrate() before predicting")
        prob = _as_1d(prob_positive, "prob_positive")
        target = _as_1d(labels, "labels", dtype=int)
        if prob.size != target.size:
            raise ValueError(
                f"prob_positive and labels differ in length: {prob.size} != {target.size}"
            )
        if not np.all((prob >= 0.0) & (prob <= 1.0)):
            raise ValueError("prob_positive must lie in [0, 1]")
        if not np.all(np.isin(target, LABELS)):
            raise ValueError("labels must contain only 0 and 1")

        levels = np.array(self.alpha_levels_, dtype=float, copy=True)
        n = prob.size
        codes = np.empty(n, dtype=np.int64)
        history = np.empty((n, 2), dtype=float)
        errors = np.zeros(n, dtype=float)

        q0_source = self.quantiles_[0]
        q1_source = self.quantiles_[1]
        goal = self.alpha
        gamma = self.gamma

        for t in range(n):
            history[t] = levels
            q_normal = q0_source.at(levels[0])
            q_fraud = q1_source.at(levels[1])
            p = float(prob[t])
            in_normal = p <= q_normal
            in_fraud = (1.0 - p) <= q_fraud
            if in_normal and in_fraud:
                codes[t] = SET_BOTH
            elif in_normal:
                codes[t] = SET_NORMAL
            elif in_fraud:
                codes[t] = SET_FRAUD
            else:
                codes[t] = SET_EMPTY

            label = int(target[t])
            covered = in_normal if label == 0 else in_fraud
            error = 0.0 if covered else 1.0
            errors[t] = error
            if update:
                # Only the class just observed moves; the other keeps its level.
                levels[label] = np.clip(
                    levels[label] + gamma * (goal - error),
                    ADAPTIVE_ALPHA_FLOOR,
                    ADAPTIVE_ALPHA_CEIL,
                )

        self.alpha_levels_ = levels
        self.stream_ = AdaptiveStreamResult(codes, target, history, levels, errors)
        return self.stream_

    def predict_set_codes(self, prob_positive, labels=None) -> np.ndarray:
        """Prediction set codes of the causal stream (``labels`` required)."""
        if labels is None:
            raise ValueError(
                "adaptive_mondrian needs the realised label stream: pass labels to "
                "predict_set_codes() or use predict_stream()"
            )
        return self.predict_stream(prob_positive, labels).codes

    # -- reporting ------------------------------------------------------
    def describe(self) -> Dict[str, float]:
        """Calibration summary plus the state the online update reached."""
        description = super().describe()
        levels = self.alpha_levels_ if self.alpha_levels_ is not None else np.full(2, self.alpha)
        description.update(
            {
                "aci_gamma": float(self.gamma),
                "alpha_final_normal": float(levels[0]),
                "alpha_final_fraud": float(levels[1]),
            }
        )
        return description


def predict_adaptive_mondrian(
    prob_cal,
    labels_cal,
    prob_test,
    labels_test,
    *,
    alpha: float = 0.05,
    gamma: float = 0.005,
    update: bool = True,
) -> AdaptiveStreamResult:
    """Calibrate on one block and run class-conditional ACI over a stream.

    Convenience wrapper around :class:`AdaptiveMondrianCP` for scripts and
    tests::

        result = predict_adaptive_mondrian(
            prob_cal, y_cal, prob_test, y_test, alpha=0.05, gamma=0.005
        )
        result.codes          # prediction set code per test transaction
        result.alpha_final    # miscoverage level each class ended on

    ``gamma`` is the ACI step size: each step moves the level of the observed
    class by at most ``gamma``, so ``1 / gamma`` steps is the resolution at
    which the scheme can follow a shift.
    """
    cp = AdaptiveMondrianCP(alpha=alpha, gamma=gamma).calibrate(prob_cal, labels_cal)
    return cp.predict_stream(prob_test, labels_test, update=update)
