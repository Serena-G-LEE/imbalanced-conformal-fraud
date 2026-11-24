"""Dataset loading and leakage-free temporal splits.

Both supported datasets are time-ordered event logs, so a random split would
leak future information into the calibration and test windows.  Every loader
therefore sorts by the dataset's own timestamp and cuts the data
chronologically into ``train / calibration / test`` blocks.  All fitted
preprocessing (frequency maps, sentinel fill values) is estimated on the
training block only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_FRACTIONS",
    "SplitDataset",
    "load_ieee_cis",
    "load_ulb",
    "temporal_split_indices",
]

#: ``train / calibration / test`` proportions of the time-ordered data.
DEFAULT_FRACTIONS: Tuple[float, float, float] = (0.60, 0.20, 0.20)


@dataclass
class SplitDataset:
    """A chronologically split binary-classification dataset.

    Feature matrices are ``float32`` numpy arrays; the calibration block is
    kept separate because conformal quantiles must not be estimated on data
    the model was trained on.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        """Counts, prevalence and split sizes, for logging and run metadata."""
        def block(y: np.ndarray) -> Dict[str, float]:
            positives = float(np.sum(y == 1))
            return {
                "n": int(y.size),
                "n_fraud": int(positives),
                "prevalence": positives / max(y.size, 1),
            }

        return {
            "dataset": self.name,
            "n_features": len(self.feature_names),
            "train": block(self.y_train),
            "calibration": block(self.y_cal),
            "test": block(self.y_test),
            **self.meta,
        }


def _validate_fractions(fractions: Sequence[float]) -> Tuple[float, float, float]:
    values = tuple(float(f) for f in fractions)
    if len(values) != 3:
        raise ValueError("fractions must contain exactly three values (train, calibration, test)")
    if any(v <= 0.0 for v in values):
        raise ValueError(f"all split fractions must be positive, got {values}")
    if not np.isclose(sum(values), 1.0, atol=1e-6):
        raise ValueError(f"split fractions must sum to 1.0, got {sum(values):.6f}")
    return values  # type: ignore[return-value]


def temporal_split_indices(
    n_samples: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological index blocks ``(train, calibration, test)``.

    ``train`` holds the oldest rows, ``test`` the newest, so the evaluation
    window is strictly in the future of everything used for fitting.
    """
    train_fraction, cal_fraction, _ = _validate_fractions(fractions)
    n = int(n_samples)
    if n < 3:
        raise ValueError(f"need at least 3 samples to build three splits, got {n}")

    n_train = max(int(np.floor(train_fraction * n)), 1)
    n_cal = max(int(np.floor(cal_fraction * n)), 1)
    n_test = n - n_train - n_cal
    if n_test < 1:
        # Shrink the training block just enough to leave a test block.
        overflow = 1 - n_test
        n_train = max(n_train - overflow, 1)
        n_test = n - n_train - n_cal
        if n_test < 1:
            raise ValueError(f"cannot build three non-empty splits from {n} samples")

    train = np.arange(0, n_train)
    calibration = np.arange(n_train, n_train + n_cal)
    test = np.arange(n_train + n_cal, n)
    return train, calibration, test


def _to_float32(matrix: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return np.ascontiguousarray(array)


def load_ulb(
    path: str | Path,
    *,
    time_column: str = "Time",
    label_column: str = "Class",
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    log1p_amount: bool = True,
    drop_columns: Optional[Sequence[str]] = None,
    name: str = "ulb",
) -> SplitDataset:
    """Load the ULB credit-card fraud dataset (``creditcard.csv``).

    The file holds 284,807 European card transactions from September 2013, of
    which 492 (0.172 %) are fraud.  ``V1..V28`` are anonymised PCA components,
    ``Time`` is the seconds elapsed since the first transaction and ``Amount``
    is the raw transaction amount.

    Parameters
    ----------
    path:
        Path to ``creditcard.csv`` (Kaggle: "Credit Card Fraud Detection").
    fractions:
        Ordered ``(train, calibration, test)`` proportions after sorting by
        ``Time``.
    log1p_amount:
        Replace ``Amount`` by ``log1p(Amount)``, which makes the feature scale
        comparable to the PCA components.  This is a fixed transform, not a
        fitted statistic, so it cannot leak information across splits.
    """
    import pandas as pd  # imported lazily: tests do not need pandas

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"ULB data file not found: {path}. Download 'creditcard.csv' from the "
            "Kaggle 'Credit Card Fraud Detection' dataset (mlg-ulb) and place it there."
        )

    frame = pd.read_csv(path)
    missing = {time_column, label_column} - set(frame.columns)
    if missing:
        raise KeyError(f"missing required columns in {path.name}: {sorted(missing)}")

    frame = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)

    target = frame[label_column].to_numpy()
    if not np.all(np.isin(np.unique(target), [0, 1])):
        raise ValueError(f"{label_column} must be binary (0/1)")

    excluded = {time_column, label_column, *(drop_columns or ())}
    feature_names = [column for column in frame.columns if column not in excluded]

    features = frame[feature_names].astype(np.float64)
    if log1p_amount and "Amount" in features.columns:
        features["Amount"] = np.log1p(features["Amount"].clip(lower=0.0))
    matrix = features.to_numpy(dtype=np.float32)

    train_idx, cal_idx, test_idx = temporal_split_indices(len(frame), fractions)
    times = frame[time_column].to_numpy()

    return SplitDataset(
        name=name,
        X_train=_to_float32(matrix[train_idx]),
        y_train=target[train_idx].astype(np.int64),
        X_cal=_to_float32(matrix[cal_idx]),
        y_cal=target[cal_idx].astype(np.int64),
        X_test=_to_float32(matrix[test_idx]),
        y_test=target[test_idx].astype(np.int64),
        feature_names=feature_names,
        meta={
            "source": str(path),
            "split": "temporal",
            "time_column": time_column,
            "fractions": list(_validate_fractions(fractions)),
            "time_range": {
                "train": [float(np.min(times[train_idx])), float(np.max(times[train_idx]))],
                "calibration": [float(np.min(times[cal_idx])), float(np.max(times[cal_idx]))],
                "test": [float(np.min(times[test_idx])), float(np.max(times[test_idx]))],
            },
        },
    )


def load_ieee_cis(
    dir_path: str | Path,
    *,
    transaction_file: str = "train_transaction.csv",
    identity_file: str = "train_identity.csv",
    time_column: str = "TransactionDT",
    label_column: str = "isFraud",
    id_column: str = "TransactionID",
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    log1p_amount: bool = True,
    fill_value: float = -1.0,
    drop_columns: Optional[Sequence[str]] = None,
    name: str = "ieee_cis",
) -> SplitDataset:
    """Load the IEEE-CIS Fraud Detection dataset (transaction + identity).

    ``train_transaction.csv`` and ``train_identity.csv`` are merged on
    ``TransactionID`` with a left join (identity information is optional per
    transaction), sorted by ``TransactionDT`` and split chronologically.

    Preprocessing, all fitted on the training block only:

    * object columns (``ProductCD``, ``card4``, ``id_31`` ...) are frequency
      encoded, i.e. replaced by their relative frequency in the *training*
      block; categories first seen later map to ``0``.
    * numeric columns are cast to ``float32`` and missing values (which are
      structural for identity/Vesta features) are filled with ``fill_value``.
    * ``TransactionAmt`` is optionally ``log1p`` transformed.
    """
    import pandas as pd  # imported lazily: tests do not need pandas

    base = Path(dir_path)
    transaction_path = base / transaction_file
    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"IEEE-CIS transaction file not found: {transaction_path}. Download "
            "'train_transaction.csv' and 'train_identity.csv' from the Kaggle "
            "'IEEE-CIS Fraud Detection' competition into that directory."
        )

    frame = pd.read_csv(transaction_path, low_memory=False)
    missing = {time_column, label_column, id_column} - set(frame.columns)
    if missing:
        raise KeyError(f"missing required columns in {transaction_file}: {sorted(missing)}")

    identity_path = base / identity_file
    identity_joined = identity_path.is_file()
    if identity_joined:
        identity = pd.read_csv(identity_path, low_memory=False)
        frame = frame.merge(identity, on=id_column, how="left", suffixes=("", "_identity"))
    else:
        warnings.warn(
            f"{identity_path} not found; continuing with transaction features only",
            RuntimeWarning,
            stacklevel=2,
        )

    frame = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)

    target = frame[label_column].to_numpy()
    if not np.all(np.isin(np.unique(target), [0, 1])):
        raise ValueError(f"{label_column} must be binary (0/1)")

    excluded = {time_column, label_column, id_column, *(drop_columns or ())}
    feature_names = [column for column in frame.columns if column not in excluded]

    train_idx, cal_idx, test_idx = temporal_split_indices(len(frame), fractions)

    categorical = [
        column
        for column in feature_names
        if frame[column].dtype == object or str(frame[column].dtype).startswith("category")
    ]
    numeric = [column for column in feature_names if column not in categorical]

    # Frequency maps are estimated on the training block only to avoid leakage.
    train_frame = frame.iloc[train_idx]
    frequency_maps = {
        column: train_frame[column].value_counts(normalize=True, dropna=True)
        for column in categorical
    }

    def encode(index: np.ndarray) -> np.ndarray:
        block = frame.iloc[index]
        columns: List[np.ndarray] = []
        for column in numeric:
            values = pd.to_numeric(block[column], errors="coerce").to_numpy(dtype=np.float64)
            values = np.where(np.isfinite(values), values, fill_value)
            if log1p_amount and column == "TransactionAmt":
                values = np.log1p(np.clip(values, 0.0, None))
            columns.append(values)
        for column in categorical:
            mapped = block[column].map(frequency_maps[column])
            columns.append(mapped.fillna(0.0).to_numpy(dtype=np.float64))
        matrix = np.column_stack(columns) if columns else np.empty((len(block), 0))
        return _to_float32(matrix, fill_value=fill_value)

    encoded_names = numeric + [f"{column}__freq" for column in categorical]
    times = frame[time_column].to_numpy()

    return SplitDataset(
        name=name,
        X_train=encode(train_idx),
        y_train=target[train_idx].astype(np.int64),
        X_cal=encode(cal_idx),
        y_cal=target[cal_idx].astype(np.int64),
        X_test=encode(test_idx),
        y_test=target[test_idx].astype(np.int64),
        feature_names=encoded_names,
        meta={
            "source": str(base),
            "split": "temporal",
            "time_column": time_column,
            "fractions": list(_validate_fractions(fractions)),
            "identity_joined": identity_joined,
            "n_numeric_features": len(numeric),
            "n_categorical_features": len(categorical),
            "categorical_encoding": "train-frequency",
            "time_range": {
                "train": [float(np.min(times[train_idx])), float(np.max(times[train_idx]))],
                "calibration": [float(np.min(times[cal_idx])), float(np.max(times[cal_idx]))],
                "test": [float(np.min(times[test_idx])), float(np.max(times[test_idx]))],
            },
        },
    )
