import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import get_output_dir, detect_execution_environment
from .tracking import MlflowCallback

pre_features = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "departure_timestamp",
    "item_count",
    "usd_base_amount",
    "seats_available",
    "fare_class",
    "offer_time",
    "multiplier_fare_class",
    "multiplier_loyalty",
    "multiplier_success_history",
    "multiplier_payment_type",
    "upgrade_type",
    "current_timestamp",
    "snapshot_num",
]

features = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "item_count_grp",
    "usd_base_amount_grp",
    "usd_base_amount_25%",
    "usd_base_amount_50%",
    "usd_base_amount_75%",
    "usd_base_amount_max",
    "num_offers_grp",
    "seats_available_grp",
    "fare_class",
    "offer_time_grp",
    "days_before_departure",
    "multiplier_fare_class",
    "multiplier_loyalty",
    "multiplier_success_history",
    "multiplier_payment_type",
]

cat_features = [
    "carrier_code",
    "flight_number",
    "item_count_grp",
    "usd_base_amount_grp",
    "num_offers_grp",
    "seats_available_grp",
    "fare_class",
    "offer_time_grp",
]


def add_days_b4_depart(data):
    data["days_before_departure"] = (
        data.departure_timestamp - data.current_timestamp
    ).apply(lambda y: y.total_seconds()) / (60 * 60 * 24)
    return data


def add_group_features(data):
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
    offer_time_bins = list(range(0, 31, 1)) + [float("inf")]
    offer_time_labels = list(range(1, 32, 1))
    data["offer_time_grp"] = pd.cut(
        data["offer_time"],
        bins=offer_time_bins,
        labels=offer_time_labels,
    )
    item_count_threshold = 4
    data["item_count_grp"] = data["item_count"].apply(
        lambda x: x if x <= item_count_threshold else item_count_threshold + 1
    )
    data["item_count_grp"] = pd.Categorical(data["item_count_grp"])
    num_offers_threshold = 15
    data["num_offers_grp"] = data["num_offers"].apply(
        lambda x: x if x <= num_offers_threshold else num_offers_threshold + 1
    )
    data["num_offers_grp"] = pd.Categorical(data["num_offers_grp"])
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


def reduce_features(X):
    if isinstance(X, pd.DataFrame):
        return X[features]
    else:
        return X


# FunctionTransformer allows arbitrary pandas-based functions

add_days_b4_depart_transformer = FunctionTransformer(add_days_b4_depart_wrapper)
group_features_transformer = FunctionTransformer(add_group_features_wrapper)
bin_features_transformer = FunctionTransformer(bin_features_wrapper)
quantiles_transformer = FunctionTransformer(add_quantiles_wrapper)
reduce_features_transformer = FunctionTransformer(reduce_features)


# ---- 1) Minimal routing-aware wrapper
class CBC(BaseEstimator, ClassifierMixin):
    def __init__(self, **cb_params):
        if detect_execution_environment()[0] == "sagemaker_job":
            train_dir = get_output_dir()
            cb_params["train_dir"] = train_dir
            cb_params["allow_writing_files"] = True
            self._callbacks = None
        else:
            self._callbacks = [MlflowCallback()]
        self.cb_params = cb_params
        self._cb = None

    # sklearn will route eval_set here if we request it on the instance
    def fit(self, X, y=None, eval_set=None, **fit_kwargs):
        self._cb = CatBoostClassifier(**self.cb_params)
        # eval_set supports (X_val, y_val) tuples or Pool objects
        self._cb.fit(
            X,
            y,
            eval_set=eval_set,
            cat_features=cat_features,
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
        return self.cb_params.copy()

    def set_params(self, **params):
        self.cb_params.update(params)
        return self


def build_pipeline(**kw):
    # CatBoostClassifier integrates with sklearn API
    clf = CBC(
        task_type=kw.get("task_type", "CPU"),
        devices=kw.get("devices", "0"),
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        auto_class_weights="Balanced",
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
