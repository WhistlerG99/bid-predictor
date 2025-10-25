"""Helpers for adjusting feature metadata during hyperparameter tuning."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, MutableMapping, Tuple

from bid_predictor.feature_config import _ensure_groupby_keys


def split_combination(
    combination: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Split a flattened grid combination into model and transform sections."""

    cat_params: Dict[str, Any] = {}
    transform_params: Dict[str, Dict[str, Any]] = {
        "impute_value": {},
        "impute_median": {},
        "outlier": {},
        "bins": {},
    }

    for key, value in combination.items():
        if key == "__noop__":
            continue
        prefix, rest = key.split("__", 1)
        if prefix == "catboost":
            cat_params[rest] = value
        elif prefix == "transform":
            section, feature = rest.split("__", 1)
            transform_params[section][feature] = value
        else:
            raise ValueError(f"Unrecognized parameter key: {key}")

    return cat_params, transform_params


def clone_feature_metadata(
    feature_metadata: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Deep-copy the feature metadata mapping."""

    return {name: dict(values) for name, values in feature_metadata.items()}


def rebuild_feature_config(
    feature_metadata: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Reconstruct a feature configuration dictionary from metadata."""

    pre_features = [
        name for name, values in feature_metadata.items() if not values.get("derived", False)
    ]
    pre_features = _ensure_groupby_keys(pre_features)

    selected_features = [
        name for name, values in feature_metadata.items() if values.get("include_in_model", False)
    ]
    categorical_features = [
        name
        for name, values in feature_metadata.items()
        if values.get("include_in_model", False) and values.get("categorical", False)
    ]
    impute_value = [
        (name, values.get("impute_value"))
        for name, values in feature_metadata.items()
        if values.get("include_in_model", False) and values.get("impute_value") is not None
    ]
    impute_median = [
        name
        for name, values in feature_metadata.items()
        if values.get("include_in_model", False) and values.get("impute_median", False)
    ]
    outlier = [
        (name, values.get("outlier"))
        for name, values in feature_metadata.items()
        if values.get("include_in_model", False) and values.get("outlier") is not None
    ]
    bins = [
        (name, values.get("bins"))
        for name, values in feature_metadata.items()
        if values.get("include_in_model", False)
        and values.get("categorical", False)
        and values.get("bins") is not None
    ]

    return {
        "pre_features": pre_features,
        "features": selected_features,
        "cat_features": categorical_features,
        "feature_metadata": feature_metadata,
        "impute_value": impute_value,
        "impute_median": impute_median,
        "outlier": outlier,
        "bins": bins,
    }


def apply_transform_overrides(
    metadata: MutableMapping[str, MutableMapping[str, Any]],
    overrides: Mapping[str, Dict[str, Any]],
) -> None:
    """Mutate feature metadata with the provided transformation overrides."""

    for section, feature_map in overrides.items():
        for feature, value in feature_map.items():
            if feature not in metadata:
                raise KeyError(f"Feature '{feature}' not found in feature configuration")
            if section == "impute_median":
                metadata[feature][section] = bool(value)
            else:
                metadata[feature][section] = value


def summarize_transform_params(overrides: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten the transformation overrides for reporting purposes."""

    summary: Dict[str, Any] = {}
    for section, feature_map in overrides.items():
        for feature, value in feature_map.items():
            key = f"{section}.{feature}"
            if isinstance(value, (dict, list)):
                summary[key] = json.dumps(value, sort_keys=True)
            else:
                summary[key] = value
    return summary
