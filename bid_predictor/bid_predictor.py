import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from .transform import (
    ArbitraryOutlierCapperCustom,
    ArbitraryDiscretiserCustom,
    ArbitraryNumberImputerCustom,
    AddMissingIndicatorCustom,
    MeanMedianImputerCustom,
    add_flight_code_transformer,
    add_days_b4_depart_transformer,
    group_features_transformer,
    quantiles_transformer,
    ColumnReducer,
)
from .tracking import MlflowCallback
from .utils import detect_execution_environment, get_output_dir
from .feature_config import _DEFAULT_FEATURE_CONFIG


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
