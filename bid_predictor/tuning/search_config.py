"""Utilities for loading hyperparameter search configurations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

DEFAULT_SEARCH_CONFIG: Dict[str, Dict[str, Any]] = {
    "catboost": {
        "iterations": [200, 400],
        "depth": [6, 8],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [3.0, 5.0],
    },
    "transform": {
        "impute_median": {},
        "impute_value": {},
        "outlier": {},
        "bins": {},
    },
}


def load_search_config(path: Optional[str]) -> Mapping[str, Any]:
    """Load the search configuration file if provided."""

    if path is None:
        return deepcopy(DEFAULT_SEARCH_CONFIG)

    search_path = Path(path)
    if not search_path.is_file():
        raise FileNotFoundError(f"Search configuration file not found: {search_path}")

    with search_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    catboost_cfg = config.get("catboost", {})
    transform_cfg = config.get("transform", {})

    transform_cfg.setdefault("impute_value", {})
    transform_cfg.setdefault("impute_median", {})
    transform_cfg.setdefault("outlier", {})
    transform_cfg.setdefault("bins", {})

    return {
        "catboost": catboost_cfg,
        "transform": transform_cfg,
    }
