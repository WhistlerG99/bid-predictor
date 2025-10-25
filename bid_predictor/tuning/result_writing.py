"""Helpers for writing tuning outputs to disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_results_csv(path: str | Path, results: pd.DataFrame) -> Path:
    """Persist the evaluated tuning combinations as a CSV file."""

    output_path = _ensure_parent(Path(path))
    results.to_csv(output_path, index=False)
    return output_path


def write_best_feature_config(path: str | Path, feature_config: Mapping[str, Any]) -> Path:
    """Persist the feature configuration for the best score as YAML."""

    output_path = _ensure_parent(Path(path))
    features_yaml = {
        "features": {name: dict(values) for name, values in feature_config["feature_metadata"].items()},
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(features_yaml, handle, sort_keys=True)
    return output_path


def write_best_catboost_params(path: str | Path, params: Mapping[str, Any]) -> Path:
    """Persist the best CatBoost hyperparameters explored by the tuner."""

    output_path = _ensure_parent(Path(path))
    payload = {"catboost": dict(params)}
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    return output_path


def write_best_result_json(path: str | Path, payload: Any) -> Path:
    """Persist the best result summary as a JSON document."""

    output_path = _ensure_parent(Path(path))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path
