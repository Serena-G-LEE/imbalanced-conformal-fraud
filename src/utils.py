"""Small shared helpers: seeding, config loading, output directories."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

__all__ = ["ensure_dir", "load_config", "save_json", "set_global_seed", "utc_timestamp"]


def set_global_seed(seed: int) -> int:
    """Seed ``random``, ``numpy`` and ``PYTHONHASHSEED`` for one run."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def load_config(path: str | Path) -> Dict[str, Any]:
    """Read a YAML experiment config."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyYAML is required to read configs/*.yaml; install dependencies with "
            "`pip install -r requirements.txt`"
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config at {config_path} must be a mapping")
    return config


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Write ``payload`` as pretty-printed UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return path


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp for run metadata."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
