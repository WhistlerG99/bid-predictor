import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from .utils import get_output_dir, in_sagemaker
from .tracking import MlflowCallback

pre_features = [
    "carrier_code",
    "flight_number",
    "travel_date",
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


def add_group_features(out):
    num_offers_col_name = "num_offers"
    usd_base_amount_max_name = "usd_base_amount_max"
    out = out.drop(
        columns=[num_offers_col_name, usd_base_amount_max_name], errors="ignore"
    )
    out = out.merge(
        out.groupby(
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
            observed=True,
        )
        .size()
        .rename(num_offers_col_name)
        .reset_index(),
        on=["carrier_code", "flight_number", "travel_date", "upgrade_type"],
    )
    out = out.merge(
        out.groupby(
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
            observed=True,
        )["usd_base_amount"]
        .max()
        .rename(usd_base_amount_max_name)
        .reset_index(),
        on=["carrier_code", "flight_number", "travel_date", "upgrade_type"],
    )
    return out


def bin_features(out):
    amount_bins = [-float("inf")] + list(range(100, 1701, 100)) + [float("inf")]
    amount_labels = (
        ["<" + str(amount_bins[1])]
        + [
            f"{amount_bins[i]}-{amount_bins[i+1]}"
            for i in range(1, len(amount_bins) - 2)
        ]
        + [">" + str(amount_bins[-2])]
    )
    out["usd_base_amount_grp"] = pd.cut(
        out["usd_base_amount"], bins=amount_bins, labels=amount_labels
    )
    offer_time_bins = list(range(0, 31, 1)) + [float("inf")]
    offer_time_labels = list(range(1, 32, 1))
    out["offer_time_grp"] = pd.cut(
        out["offer_time"],
        bins=offer_time_bins,
        labels=offer_time_labels,
    )
    item_count_threshold = 4
    out["item_count_grp"] = out["item_count"].apply(
        lambda x: x if x <= item_count_threshold else item_count_threshold + 1
    )
    out["item_count_grp"] = pd.Categorical(out["item_count_grp"])
    num_offers_threshold = 15
    out["num_offers_grp"] = out["num_offers"].apply(
        lambda x: x if x <= num_offers_threshold else num_offers_threshold + 1
    )
    out["num_offers_grp"] = pd.Categorical(out["num_offers_grp"])
    seats_available_low_threshold = -1
    seats_available_high_threshold = 30
    out["seats_available_grp"] = out["seats_available"].apply(
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
    out["seats_available_grp"] = pd.Categorical(
        out.fillna({"seats_available_grp": -2}).seats_available_grp.astype(int)
    )
    return out


def quantiles_vectorized(vals, qs=(0.25, 0.50, 0.75)):
    vals = np.asarray(vals)
    n = vals.size
    out = np.full((len(qs), n), np.nan, dtype=float)
    if n <= 1:
        return out

    # Stable sort to keep deterministic ranks with ties
    order = np.argsort(vals, kind="mergesort")
    S = vals[order]                           # sorted values
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)               # rank of each original element

    m = n - 1
    for qi, q in enumerate(qs):
        h = q * (m - 1)                       # NumPy's 'linear' method position
        j = int(np.floor(h))
        gamma = h - j
        j2 = min(j + 1, m - 1)                # clamp upper neighbor inside [0, m-1]

        # Map indices from T (length m) back to S (length n), accounting for the removed item
        idx_a = j  + (j  >= ranks)
        idx_b = j2 + (j2 >= ranks)

        out[qi, :] = (1.0 - gamma) * S[idx_a] + gamma * S[idx_b]

    return out  # shape (len(qs), n)


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


def add_quantiles(out):
    out[["usd_base_amount_25%", "usd_base_amount_50%", "usd_base_amount_75%"]] = (
        out.groupby(
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"],
            group_keys=False,
            observed=True,
            sort=False,
        ).apply(quantiles_group, col="usd_base_amount", include_groups=False)
    )
    return out


# --- Wrappers around your existing funcs so they can be used in pipelines ---
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
group_features_transformer = FunctionTransformer(add_group_features_wrapper)
bin_features_transformer = FunctionTransformer(bin_features_wrapper)
quantiles_transformer = FunctionTransformer(add_quantiles_wrapper)
reduce_features_transformer = FunctionTransformer(reduce_features)


# ---- 1) Minimal routing-aware wrapper
class CBC(BaseEstimator, ClassifierMixin):
    def __init__(self, **cb_params):
        if in_sagemaker():
            train_dir = get_output_dir()
            cb_params["train_dir"] = train_dir
            cb_params["allow_writing_files"] = True
            cb_params["verbose"] = 1
            self._callbacks = None
        else:
            self._callbacks = [MlflowCallback()]
        self.cb_params = cb_params
        self._cb = None

    # sklearn will route eval_set here if we request it on the instance
    def fit(self, X, y=None, eval_set=None, **fit_kwargs):
        self._cb = CatBoostClassifier(
            **self.cb_params
        )
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
            ("group", group_features_transformer),
            ("bin", bin_features_transformer),
            ("loo", quantiles_transformer),
            ("reduce", reduce_features_transformer),
            ("clf", clf),
        ],
        transform_input=["eval_set"],
    )
    return pipeline
