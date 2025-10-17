import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .tracking import MlflowCallback
from .utils import detect_execution_environment, get_output_dir


_FEATURE_CONFIG_ENV = "BID_PREDICTOR_FEATURE_CONFIG"
_GROUPBY_KEY_FEATURES = (
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
    "snapshot_num",
)

_FEATURE_BOOLEAN_FIELDS = {
    "categorical": False,
    "include_in_model": True,
    "derived": False,
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

        normalized = {field: bool(metadata.get(field, default)) for field, default in _FEATURE_BOOLEAN_FIELDS.items()}
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

    metadata = {
        name: values.copy()
        for name, values in feature_entries
    }

    pre_features = [name for name, values in feature_entries if not values["derived"]]
    pre_features = _ensure_groupby_keys(pre_features)

    selected_features = [name for name, values in feature_entries if values["include_in_model"]]
    categorical_features = [
        name
        for name, values in feature_entries
        if values["include_in_model"] and values["categorical"]
    ]

    return {
        "pre_features": pre_features,
        "features": selected_features,
        "cat_features": categorical_features,
        "feature_metadata": metadata,
    }


_DEFAULT_FEATURE_CONFIG = load_feature_config()
pre_features = _DEFAULT_FEATURE_CONFIG["pre_features"]
features = _DEFAULT_FEATURE_CONFIG["features"]
cat_features = _DEFAULT_FEATURE_CONFIG["cat_features"]
feature_metadata = _DEFAULT_FEATURE_CONFIG["feature_metadata"]


def add_days_b4_depart(data):
    required = {"departure_timestamp", "current_timestamp"}
    if not required.issubset(data.columns):
        return data
    data["days_before_departure"] = (
        data.departure_timestamp - data.current_timestamp
    ).apply(lambda y: y.total_seconds()) / (60 * 60 * 24)
    return data


def add_group_features(data):
    if "usd_base_amount" not in data.columns:
        return data
    if any(key not in data.columns for key in _GROUPBY_KEY_FEATURES):
        return data
    num_offers_col_name = "num_offers"
    usd_base_amount_max_name = "usd_base_amount_max"
    data = data.drop(
        columns=[num_offers_col_name, usd_base_amount_max_name], errors="ignore"
    )
    data = data.merge(
        data.groupby(
            [
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
                "snapshot_num",
            ],
            observed=True,
        )
        .size()
        .rename(num_offers_col_name)
        .reset_index(),
        on=[
            "carrier_code",
            "flight_number",
            "travel_date",
            "upgrade_type",
            "snapshot_num",
        ],
    )
    data = data.merge(
        data.groupby(
            [
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
                "snapshot_num",
            ],
            observed=True,
        )["usd_base_amount"]
        .max()
        .rename(usd_base_amount_max_name)
        .reset_index(),
        on=[
            "carrier_code",
            "flight_number",
            "travel_date",
            "upgrade_type",
            "snapshot_num",
        ],
    )
    return data


def bin_features(data):
    if "usd_base_amount" in data.columns:
        amount_bins = [-float("inf")] + list(range(100, 1701, 100)) + [float("inf")]
        amount_labels = (
            ["<" + str(amount_bins[1])]
            + [
                f"{amount_bins[i]}-{amount_bins[i+1]}"
                for i in range(1, len(amount_bins) - 2)
            ]
            + [">" + str(amount_bins[-2])]
        )

        data["usd_base_amount_grp"] = pd.cut(
            data["usd_base_amount"], bins=amount_bins, labels=amount_labels
        )

    if "offer_time" in data.columns:
        offer_time_bins = list(range(0, 31, 1)) + [float("inf")]
        offer_time_labels = list(range(1, 32, 1))
        data["offer_time_grp"] = pd.cut(
            data["offer_time"],
            bins=offer_time_bins,
            labels=offer_time_labels,
        )

    if "item_count" in data.columns:
        item_count_threshold = 4
        data["item_count_grp"] = data["item_count"].apply(
            lambda x: x if x <= item_count_threshold else item_count_threshold + 1
        )
        data["item_count_grp"] = pd.Categorical(data["item_count_grp"])

    if "num_offers" in data.columns:
        num_offers_threshold = 15
        data["num_offers_grp"] = data["num_offers"].apply(
            lambda x: x if x <= num_offers_threshold else num_offers_threshold + 1
        )
        data["num_offers_grp"] = pd.Categorical(data["num_offers_grp"])

    if "seats_available" in data.columns:
        seats_available_low_threshold = -1
        seats_available_high_threshold = 30
        data["seats_available_grp"] = data["seats_available"].apply(
            lambda x: (
                (
                    x
                    if x < seats_available_high_threshold or np.isnan(x)
                    else seats_available_high_threshold
                )
                if x > seats_available_low_threshold or np.isnan(x)
                else seats_available_low_threshold
            )
        )
        data["seats_available_grp"] = pd.Categorical(
            data.fillna({"seats_available_grp": -2}).seats_available_grp.astype(int)
        )
    return data


def quantiles_vectorized(vals, qs=(0.25, 0.50, 0.75)):
    vals = np.asarray(vals)
    n = vals.size
    data = np.full((len(qs), n), np.nan, dtype=float)
    if n <= 1:
        return data

    # Stable sort to keep deterministic ranks with ties
    order = np.argsort(vals, kind="mergesort")
    S = vals[order]  # sorted values
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)  # rank of each original element

    m = n - 1
    for qi, q in enumerate(qs):
        h = q * (m - 1)  # NumPy's 'linear' method position
        j = int(np.floor(h))
        gamma = h - j
        j2 = min(j + 1, m - 1)  # clamp upper neighbor inside [0, m-1]

        # Map indices from T (length m) back to S (length n), accounting for the removed item
        idx_a = j + (j >= ranks)
        idx_b = j2 + (j2 >= ranks)

        data[qi, :] = (1.0 - gamma) * S[idx_a] + gamma * S[idx_b]

    return data  # shape (len(qs), n)


def quantiles_group(group, col):
    arr = group[col].to_numpy()
    q25, q50, q75 = quantiles_vectorized(arr, (0.25, 0.50, 0.75))
    return pd.DataFrame(
        {
            "q25_excl_self": q25,
            "median_excl_self": q50,
            "q75_excl_self": q75,
        },
        index=group.index,
    )


def add_quantiles(data):
    if "usd_base_amount" not in data.columns:
        return data
    if any(key not in data.columns for key in _GROUPBY_KEY_FEATURES):
        return data
    data[["usd_base_amount_25%", "usd_base_amount_50%", "usd_base_amount_75%"]] = (
        data.groupby(
            [
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
                "snapshot_num",
            ],
            group_keys=False,
            observed=True,
            sort=False,
        ).apply(quantiles_group, col="usd_base_amount", include_groups=False)
    )
    return data


# --- Wrappers around your existing funcs so they can be used in pipelines ---
def add_days_b4_depart_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return add_days_b4_depart(X)
    else:
        return X


def add_group_features_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return add_group_features(X)
    else:
        return X


def bin_features_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return bin_features(X)
    else:
        return X


def add_quantiles_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return add_quantiles(X)
    else:
        return X


class ColumnReducer(BaseEstimator, TransformerMixin):
    """Selects the configured columns from a pandas DataFrame."""

    def __init__(self, selected_features):
        self.selected_features = list(selected_features)
        self._existing_features = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._existing_features = [
                feature for feature in self.selected_features if feature in X.columns
            ]
        else:
            self._existing_features = list(self.selected_features)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        features = self._existing_features
        if features is None:
            features = [feature for feature in self.selected_features if feature in X.columns]
        else:
            features = [feature for feature in features if feature in X.columns]

        return X.reindex(columns=features)


# FunctionTransformer allows arbitrary pandas-based functions

add_days_b4_depart_transformer = FunctionTransformer(add_days_b4_depart_wrapper)
group_features_transformer = FunctionTransformer(add_group_features_wrapper)
bin_features_transformer = FunctionTransformer(bin_features_wrapper)
quantiles_transformer = FunctionTransformer(add_quantiles_wrapper)


# ---- 1) Minimal routing-aware wrapper
class CBC(BaseEstimator, ClassifierMixin):
    def __init__(self, *, cat_features, **cb_params):
        if detect_execution_environment()[0] == "sagemaker_job":
            train_dir = get_output_dir()
            cb_params["train_dir"] = train_dir
            cb_params["allow_writing_files"] = True
            self._callbacks = None
        else:
            self._callbacks = [MlflowCallback()]
        self.cb_params = cb_params
        self.cat_features = list(cat_features)
        self._cb = None

    # sklearn will route eval_set here if we request it on the instance
    def fit(self, X, y=None, eval_set=None, **fit_kwargs):
        self._cb = CatBoostClassifier(**self.cb_params)
        active_cat_features = [
            feature for feature in self.cat_features if feature in getattr(X, "columns", [])
        ]
        # eval_set supports (X_val, y_val) tuples or Pool objects
        self._cb.fit(
            X,
            y,
            eval_set=eval_set,
            cat_features=active_cat_features,
            callbacks=self._callbacks,
            **fit_kwargs,
        )
        return self

    def __sklearn_is_fitted__(self):
        if hasattr(self, "_cb") and self._cb is not None:
            return self._cb.is_fitted()

    # pass-through predict/predict_proba
    def predict(self, X):
        return self._cb.predict(X)

    def predict_proba(self, X):
        return self._cb.predict_proba(X)

    def get_feature_importance(self, *args, **kwargs):
        return self._cb.get_feature_importance(*args, **kwargs)

    # make params grid-searchable
    def get_params(self, deep=True):
        params = self.cb_params.copy()
        params["cat_features"] = self.cat_features
        return params

    def set_params(self, **params):
        if "cat_features" in params:
            self.cat_features = list(params.pop("cat_features"))
        self.cb_params.update(params)
        return self


def build_pipeline(feature_config=None, **kw):
    if feature_config is None:
        feature_config = _DEFAULT_FEATURE_CONFIG

    selected_features = feature_config["features"]
    categorical_features = feature_config["cat_features"]

    reduce_features_transformer = ColumnReducer(selected_features)

    # CatBoostClassifier integrates with sklearn API
    clf = CBC(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        cat_features=categorical_features,
        **kw,
    ).set_fit_request(eval_set=True)

    pipeline = Pipeline(
        steps=[
            ("depart", add_days_b4_depart_transformer),
            ("group", group_features_transformer),
            ("bin", bin_features_transformer),
            ("loo", quantiles_transformer),
            ("reduce", reduce_features_transformer),
            ("clf", clf),
        ],
        transform_input=["eval_set"],
    )
    return pipeline
