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

    steps = [
        ("flight_code", add_flight_code_transformer),
        ("depart", add_days_b4_depart_transformer),
        ("group", group_features_transformer),
    ]

    if "impute_value" in feature_config or "impute_median" in feature_config:
        variables = []
        if "impute_value" in feature_config:
            for feature, value in feature_config.get("impute_value", []):
                variables.append(feature)
        if "impute_median" in feature_config:
            for feature in feature_config.get("impute_median", []):
                variables.append(feature)

        if variables:
            steps.append(
                (
                    "indicator",
                    AddMissingIndicatorCustom(
                        variables=variables,
                    ),
                )
            )

    if "impute_value" in feature_config:
        imputer_dict={}
        for feature, value in feature_config.get("impute_value", []):
            imputer_dict[feature] = value

        if imputer_dict:
            steps.append(
                (
                    "value_imputer",
                    ArbitraryNumberImputerCustom(
                        imputer_dict=imputer_dict,
                    ),
                )
            )
    if "impute_median" in feature_config:
        variables = []
        for feature in feature_config.get("impute_median", []):
            variables.append(feature)
        
        if variables:
            steps.append(
                (
                    "median_imputer",
                    MeanMedianImputerCustom(
                        variables=variables,
                        imputation_method="median",
                    ),
                )
            )

    if "outlier" in feature_config:
        min_capping_dict, max_capping_dict = {}, {}
        for feature, outlier in feature_config.get("outlier",[]):
            if "max" in outlier:
                max_capping_dict[feature] = outlier["max"]
            if "min" in outlier:
                min_capping_dict[feature] = outlier["min"]

        outlier_args = {}
        if min_capping_dict:
            outlier_args["min_capping_dict"] = min_capping_dict
        if max_capping_dict:
            outlier_args["max_capping_dict"] = max_capping_dict

        if outlier_args:
            steps.append(
                (
                    "outliers",
                    ArbitraryOutlierCapperCustom(
                        **outlier_args,
                    ),
                )
            )
    
    steps.append(("quantile", quantiles_transformer))

    if "bins" in feature_config:
        binning_dict = {}
        for feature, bins in feature_config.get("bins", []):
            if "interval" in bins:
                binning_dict[feature] = (
                    [-float("inf")]
                    + list(range(bins["min"], bins["max"] + 1, bins["interval"]))
                    + [float("inf")]
                )
            elif "nsteps" in bins:
                binning_dict[feature] = (
                    [-float("inf")]
                    + np.linspace(bins["min"], bins["max"], bins["nsteps"]).tolist()
                    + [float("inf")]
                )
            else:
                raise ValueError(f"Invalid binning specification for feature {feature}")

        if binning_dict:
            steps.append(
                (
                    "discrete",
                    ArbitraryDiscretiserCustom(
                        binning_dict=binning_dict,
                        return_boundaries=True,
                    ),
                )
            )

    reduce_features_transformer = ColumnReducer(selected_features)
    steps.append(("reduce", reduce_features_transformer))

    # CatBoostClassifier integrates with sklearn API
    clf = CBC(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        cat_features=categorical_features,
        **kw,
    ).set_fit_request(eval_set=True)
    steps.append(("clf", clf))

    pipeline = Pipeline(
        steps=steps,
        transform_input=["eval_set"],
    )
    return pipeline
