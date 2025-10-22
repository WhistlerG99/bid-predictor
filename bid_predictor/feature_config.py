import os
import yaml
from functools import lru_cache
from pathlib import Path

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


_DEFAULT_FEATURE_CONFIG = load_feature_config()
# pre_features = _DEFAULT_FEATURE_CONFIG["pre_features"]
# features = _DEFAULT_FEATURE_CONFIG["features"]
# cat_features = _DEFAULT_FEATURE_CONFIG["cat_features"]
# feature_metadata = _DEFAULT_FEATURE_CONFIG["feature_metadata"]
