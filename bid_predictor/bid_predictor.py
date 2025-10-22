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
from .transform import (
    ArbitraryOutlierCapperCustom,
    ArbitraryDiscretiserCustom,
    ArbitraryNumberImputerCustom,
    AddMissingIndicatorCustom,
    MeanMedianImputerCustom,
)
from .tracking import MlflowCallback
from .utils import detect_execution_environment, get_output_dir


_FEATURE_CONFIG_ENV = "BID_PREDICTOR_FEATURE_CONFIG"
_GROUPBY_KEY_FEATURES = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
    "snapshot_num",
]

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

        normalized = {
            field: bool(metadata.get(field, default))
            for field, default in _FEATURE_BOOLEAN_FIELDS.items()
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


def add_flight_code(data):
    required = {"carrier_code", "flight_number"}
    if not required.issubset(data.columns):
        return data
    data["flight_code"] = (
        data["carrier_code"].astype(str) + data["flight_number"].astype(str)
    ).astype("category")
    return data


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
            _GROUPBY_KEY_FEATURES,
            observed=True,
        )
        .size()
        .rename(num_offers_col_name)
        .reset_index(),
        on=_GROUPBY_KEY_FEATURES,
    )
    data = data.merge(
        data.groupby(
            _GROUPBY_KEY_FEATURES,
            observed=True,
        )["usd_base_amount"]
        .max()
        .rename(usd_base_amount_max_name)
        .reset_index(),
        on=_GROUPBY_KEY_FEATURES,
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
            _GROUPBY_KEY_FEATURES,
            group_keys=False,
            observed=True,
            sort=False,
        ).apply(quantiles_group, col="usd_base_amount", include_groups=False)
    )
    return data


# --- Wrappers around your existing funcs so they can be used in pipelines ---
def add_flight_code_wrapper(X):
    if isinstance(X, pd.DataFrame):
        return add_flight_code(X)
    else:
        return X

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
            self._existing_features += [
                feature + "_na"
                for feature in self.selected_features
                if feature + "_na" in X.columns
            ]
        else:
            self._existing_features = list(self.selected_features)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        features = self._existing_features
        if features is None:
            features = [
                feature for feature in self.selected_features if feature in X.columns
            ]
        else:
            features = [feature for feature in features if feature in X.columns]

        return X.reindex(columns=features)


# FunctionTransformer allows arbitrary pandas-based functions
add_flight_code_transformer = FunctionTransformer(add_flight_code_wrapper)
add_days_b4_depart_transformer = FunctionTransformer(add_days_b4_depart_wrapper)
group_features_transformer = FunctionTransformer(add_group_features_wrapper)
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
            feature
            for feature in self.cat_features
            if feature in getattr(X, "columns", [])
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

    indicator = AddMissingIndicatorCustom(
        variables=[
            "seats_available",
            "multiplier_fare_class",
            "multiplier_loyalty",
            "multiplier_success_history",
            "multiplier_payment_type",
        ]
    )
    multi_imputer = ArbitraryNumberImputerCustom(
        imputer_dict={
            "multiplier_fare_class": 1,
            "multiplier_loyalty": 1,
            "multiplier_success_history": 1,
            "multiplier_payment_type": 1,
        }
    )
    seat_imputer = MeanMedianImputerCustom(
        variables=["seats_available"], imputation_method="median"
    )
    outliers = ArbitraryOutlierCapperCustom(
        max_capping_dict={"item_count": 5, "num_offers": 16},
    )

    offer_time_bins = [-float("inf")] + list(range(0, 31, 1)) + [float("inf")]

    amount_bins = list(range(0, 1701, 25)) + [float("inf")]
    amount_cats = {
        i: f"<{y}" if x == 0 else f">{x}" if y == float("inf") else f"{x}-{y}"
        for i, (x, y) in enumerate(list(zip(amount_bins[:-1], amount_bins[1:])))
    }
    seat_bins = [-float("inf")] + list(range(-1, 30, 1)) + [float("inf")]
    seat_cats = {i: i - 1 for i in range(len(seat_bins) - 1)}

    day_b4_bins = (
        [-float("inf")] + [i / 24 for i in range(0, 5 * 24 + 1, 1)] + [float("inf")]
    )
    day_b4_cats = {i: (i - 1) / 24 for i in range(len(day_b4_bins) - 1)}

    discrete = ArbitraryDiscretiserCustom(
        binning_dict={
            "offer_time": offer_time_bins,
            "usd_base_amount": amount_bins,
            "seats_available": seat_bins,
            "days_before_departure": day_b4_bins,
        },
        bin_names_dict={
            "usd_base_amount": amount_cats,
            "seats_available": seat_cats,
            "days_before_departure": day_b4_cats,
        },
    )

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
            ("flight_code", add_flight_code_transformer),
            ("depart", add_days_b4_depart_transformer),
            ("group", group_features_transformer),
            ("indicator", indicator),
            ("multi_imputer", multi_imputer),
            ("seat_imputer", seat_imputer),
            ("outliers", outliers),
            ("quantile", quantiles_transformer),
            ("discrete", discrete),
            ("reduce", reduce_features_transformer),
            ("clf", clf),
        ],
        transform_input=["eval_set"],
    )
    return pipeline
