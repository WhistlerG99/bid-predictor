import os
import json
import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

_FEATURE_CONFIG_ENV = "BID_PREDICTOR_FEATURE_CONFIG"
_GROUPBY_KEY_FEATURES = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
    "snapshot_num",
]

_FEATURE_FIELDS = {
    "categorical": False,
    "include_in_model": True,
    "derived": False,
    "impute_value": None,
    "impute_median": False,
    "outlier": None,
    "bins": None,
}

_FEATURE_BOOLEAN_FIELDS = {
    "categorical",
    "include_in_model",
    "derived",
    "impute_median",
}


def _parse_feature_spec(values):
    if values is None:
        raise KeyError("Missing 'features' section in feature config YAML")

    if isinstance(values, dict):
        items = values.items()
    elif isinstance(values, (list, tuple)):
        # Allow simple lists for backwards compatibility; treat as empty metadata
        items = ((value, {}) for value in values)
    else:
        raise TypeError(
            "'features' section in feature config must be a mapping or list, "
            f"got {type(values)!r}"
        )

    parsed = []
    seen = set()
    for name, metadata in items:
        name = str(name)
        if name in seen:
            continue
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Feature '{name}' metadata must be a mapping, got {type(metadata)!r}"
            )

        normalized = {
            field: metadata.get(field, default) if field not in _FEATURE_BOOLEAN_FIELDS else bool(metadata.get(field, default))
            for field, default in _FEATURE_FIELDS.items()
        }
        parsed.append((name, normalized))
        seen.add(name)

    return parsed


def _ensure_groupby_keys(pre_features):
    missing = [name for name in _GROUPBY_KEY_FEATURES if name not in pre_features]
    if not missing:
        return pre_features
    # Preserve original ordering and append any required keys that were missing.
    return pre_features + missing


def _resolve_feature_config_path(config_path=None):
    if config_path is not None:
        return Path(config_path)
    env_path = os.environ.get(_FEATURE_CONFIG_ENV)
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "feature_config.yaml"


@lru_cache(maxsize=None)
def load_feature_config(config_path=None):
    if config_path is not None:
        config_path = str(Path(config_path))
    path = _resolve_feature_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature config YAML not found at: {path}")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    feature_entries = _parse_feature_spec(config.get("features"))

    metadata = {name: values.copy() for name, values in feature_entries}

    pre_features = [name for name, values in feature_entries if not values["derived"]]
    pre_features = _ensure_groupby_keys(pre_features)

    selected_features = [
        name for name, values in feature_entries if values["include_in_model"]
    ]
    categorical_features = [
        name
        for name, values in feature_entries
        if values["include_in_model"] and values["categorical"]
    ]

    impute_value = [
        (name, values["impute_value"])
        for name, values in feature_entries
        if values["include_in_model"] and values["impute_value"]
    ]

    impute_median = [
        name
        for name, values in feature_entries
        if values["include_in_model"] and values["impute_median"]
    ]

    outlier = [
        (name,values["outlier"])
        for name, values in feature_entries
        if values["include_in_model"] and (values["outlier"] is not None)
    ]

    bins = [
        (name,values["bins"])
        for name, values in feature_entries
        if values["include_in_model"] and values["categorical"] and (values["bins"] is not None)
    ]
    
    return {
        "pre_features": pre_features,
        "features": selected_features,
        "cat_features": categorical_features,
        "feature_metadata": metadata,
        "impute_value": impute_value,
        "impute_median": impute_median,
        "outlier": outlier,
        "bins": bins,
    }


def _format_bins(bins):
    if not bins:
        return None
    if "interval" in bins:
        return {
            "type": "interval",
            "min": bins.get("min"),
            "max": bins.get("max"),
            "interval": bins.get("interval"),
        }
    if "nsteps" in bins:
        return {
            "type": "nsteps",
            "min": bins.get("min"),
            "max": bins.get("max"),
            "nsteps": bins.get("nsteps"),
        }
    return bins


def _format_outlier(outlier):
    if not outlier:
        return None
    return {key: outlier[key] for key in ("min", "max") if key in outlier}


def summarize_feature_transformations(feature_config):
    """Summarize model features and their transformations for MLflow logging.

    Parameters
    ----------
    feature_config : Mapping[str, Any]
        Loaded feature configuration dictionary.

    Returns
    -------
    list[dict]
        A deterministic list of feature metadata dictionaries describing the
        transformations applied to every feature that participates in the
        model.
    """

    metadata = feature_config.get("feature_metadata", {})
    summary = []
    for feature_name in sorted(metadata):
        values = metadata[feature_name]
        if not values.get("include_in_model", False):
            continue

        transformations = []
        impute_value = values.get("impute_value")
        impute_median = values.get("impute_median", False)

        if impute_value is not None:
            transformations.append({"type": "impute_value", "value": impute_value})
            transformations.append({"type": "missing_indicator"})
        elif impute_median:
            transformations.append({"type": "impute_median"})
            transformations.append({"type": "missing_indicator"})

        outlier = _format_outlier(values.get("outlier"))
        if outlier:
            transformations.append({"type": "outlier_capper", "params": outlier})

        bins = _format_bins(values.get("bins"))
        if bins:
            transformations.append({"type": "discretiser", "params": bins})

        if values.get("categorical", False):
            transformations.append({"type": "catboost_categorical"})

        if not transformations:
            transformations.append({"type": "passthrough"})

        summary.append(
            {
                "feature": feature_name,
                "derived": bool(values.get("derived", False)),
                "categorical": bool(values.get("categorical", False)),
                "impute_value": impute_value,
                "impute_median": bool(impute_median),
                "outlier": outlier,
                "bins": bins,
                "transformations": transformations,
            }
        )

    return summary


def feature_config_fingerprint(feature_summary):
    """Create a stable fingerprint for a feature configuration summary."""

    encoded = json.dumps(feature_summary, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def feature_summary_markdown(feature_summary):
    """Render a markdown table describing the feature summary."""

    if not feature_summary:
        return "No features are configured for this run."

    header = [
        "| Feature | Derived | Categorical | Transformations |",
        "| --- | --- | --- | --- |",
    ]
    rows = []
    for item in feature_summary:
        transforms = []
        for transform in item["transformations"]:
            if transform["type"] in {"impute_value", "outlier_capper", "discretiser"}:
                detail = transform.copy()
                detail_type = detail.pop("type")
                transforms.append(f"**{detail_type}**: {json.dumps(detail, sort_keys=True)}")
            elif transform["type"] == "impute_median":
                transforms.append("**impute_median**")
            elif transform["type"] == "missing_indicator":
                transforms.append("missing_indicator")
            elif transform["type"] == "catboost_categorical":
                transforms.append("catboost_categorical")
            else:
                transforms.append(transform["type"])
        rows.append(
            "| {feature} | {derived} | {categorical} | {transforms} |".format(
                feature=item["feature"],
                derived="✅" if item["derived"] else "",
                categorical="✅" if item["categorical"] else "",
                transforms="<br>".join(transforms),
            )
        )

    return "\n".join(header + rows)


_DEFAULT_FEATURE_CONFIG = load_feature_config()
# pre_features = _DEFAULT_FEATURE_CONFIG["pre_features"]
# features = _DEFAULT_FEATURE_CONFIG["features"]
# cat_features = _DEFAULT_FEATURE_CONFIG["cat_features"]
# feature_metadata = _DEFAULT_FEATURE_CONFIG["feature_metadata"]
