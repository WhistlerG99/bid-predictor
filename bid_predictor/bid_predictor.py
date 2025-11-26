import copy
import inspect
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, ClassVar, Mapping, MutableMapping

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import yaml
from .transform import (
    ArbitraryOutlierCapperCustom,
    ArbitraryDiscretiserCustom,
    ArbitraryNumberImputerCustom,
    AddMissingIndicatorCustom,
    MeanMedianImputerCustom,
    add_flight_code_transformer,
    add_days_b4_depart_transformer,
    add_bid_rank_transformer,
    group_features_transformer,
    quantiles_transformer,
    ColumnReducer,
)
from .tracking import MlflowCallback
from .utils import detect_execution_environment, get_output_dir
from .feature_config import _DEFAULT_FEATURE_CONFIG


def _introspect_catboost_defaults() -> dict[str, Any]:
    signature = inspect.signature(CatBoostClassifier.__init__)
    defaults: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        defaults[name] = None if param.default is inspect._empty else param.default
    return defaults


_CATBOOST_PARAM_DEFAULTS = _introspect_catboost_defaults()


class FeatureConfiguredPipeline(Pipeline):
    """Pipeline that persists feature configuration metadata."""

    feature_config: ClassVar[Mapping[str, Any] | None] = None
    catboost_params: ClassVar[Mapping[str, Any] | None] = None

    def __init__(
        self,
        steps,
        *,
        feature_config: Mapping[str, Any] | None = None,
        transform_input=None,
        **kwargs,
    ) -> None:
        """Initialize the pipeline and persist the supplied feature metadata.

        Parameters
        ----------
        steps:
            The ordered list of pipeline steps, forwarded to ``Pipeline``.
        feature_config:
            Raw feature configuration dictionary that should be stored on the
            fitted estimator for later inspection or serialization.
        transform_input:
            Additional scikit-learn ``Pipeline`` keyword argument specifying
            which inputs to pass through to intermediate transformers.
        **kwargs:
            Remaining keyword arguments forwarded to ``Pipeline``.
        """
        self.feature_config: Mapping[str, Any] | None = feature_config
        self.feature_config_: Mapping[str, Any] | None = None
        self.catboost_params: Mapping[str, Any] | None = None
        self.catboost_params_: Mapping[str, Any] | None = None
        super().__init__(steps, transform_input=transform_input, **kwargs)
        if feature_config is not None:
            self._assign_feature_config(feature_config)

    def _transform(self, X):
        return self[:-1].transform(X)

    def _transform_and_predict_proba(self, X):
        X_tf = self._transform(X)
        probs = self[-1].predict_proba(X_tf)
        return probs, X_tf
    
    def transform_and_predict_proba(self, X):
        X_tf = self[:1].transform(X)
        probs = self[1:].predict_proba(X_tf)
        return probs, X_tf    

    def _assign_feature_config(self, feature_config: Mapping[str, Any]) -> None:
        """Store a defensive copy of the feature configuration on the class.

        The copy ensures that downstream mutations to ``feature_config`` do not
        affect the persisted metadata attached to the pipeline instance.
        """
        cloned = copy.deepcopy(feature_config)
        type(self).feature_config = cloned
        self.feature_config_ = cloned

    def _assign_catboost_params(self, catboost_params: Mapping[str, Any]) -> None:
        cloned = copy.deepcopy(catboost_params)
        type(self).catboost_params = cloned
        self.catboost_params = cloned
        self.catboost_params_ = cloned

    @classmethod
    def dump_feature_config(cls, output_path: str | Path) -> Path:
        """Serialize the stored feature configuration to a YAML file.

        Parameters
        ----------
        output_path:
            Destination path for the emitted YAML file.

        Returns
        -------
        pathlib.Path
            The resolved path of the written YAML file.

        Raises
        ------
        ValueError
            If no feature configuration has been stored on the pipeline.
        """

        if cls.feature_config is None:
            raise ValueError(
                "FeatureConfiguredPipeline.feature_config is not set; "
                "nothing to serialize. Fit or build the pipeline with a "
                "feature configuration before dumping."
            )

        feature_metadata = cls.feature_config.get("feature_metadata")
        if not isinstance(feature_metadata, MappingABC):
            raise ValueError(
                "Feature configuration is missing 'feature_metadata' and "
                "cannot be serialized to YAML."
            )

        features_section: dict[str, Any] = {}
        for feature_name, metadata in feature_metadata.items():
            if not isinstance(metadata, MappingABC):
                raise ValueError(
                    f"Feature '{feature_name}' metadata must be a mapping to "
                    "serialize to YAML."
                )
            features_section[feature_name] = cls._prepare_for_yaml(metadata)

        payload = {"features": features_section}

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

        return destination

    @staticmethod
    def _prepare_for_yaml(value: Any) -> Any:
        """Convert mappings and tuples to YAML-friendly built-in structures."""
        if isinstance(value, MappingABC):
            return {
                key: FeatureConfiguredPipeline._prepare_for_yaml(val)
                for key, val in value.items()
                if val is not None
            }
        if isinstance(value, tuple):
            return [FeatureConfiguredPipeline._prepare_for_yaml(item) for item in value]
        if isinstance(value, list):
            return [FeatureConfiguredPipeline._prepare_for_yaml(item) for item in value]
        return copy.deepcopy(value)

    @classmethod
    def dump_catboost_params(cls, output_path: str | Path) -> Path:
        """Serialize stored CatBoost parameters that diverge from defaults."""

        if cls.catboost_params is None:
            raise ValueError(
                "FeatureConfiguredPipeline.catboost_params is not set; "
                "nothing to serialize. Build the pipeline with CatBoost "
                "parameters before dumping."
            )

        if not isinstance(cls.catboost_params, MappingABC):
            raise ValueError(
                "CatBoost parameters must be a mapping in order to be serialized."
            )

        sentinel = object()
        payload_params: dict[str, Any] = {}
        for key, value in cls.catboost_params.items():
            default_value = _CATBOOST_PARAM_DEFAULTS.get(key, sentinel)
            if (default_value is not sentinel and default_value == value) or key=="monotone_constraints":
                continue
            payload_params[key] = cls._prepare_for_yaml(value)

        payload = {"catboost": payload_params}

        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

        return destination

    def __getstate__(self) -> MutableMapping[str, Any]:
        """Include feature configuration metadata in pickled state."""
        state = super().__getstate__()
        state["_feature_config"] = self.feature_config_
        state["feature_config"] = self.feature_config
        state["_catboost_params"] = self.catboost_params_
        state["catboost_params"] = self.catboost_params
        return state

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        """Restore persisted feature configuration during unpickling."""
        feature_config = state.pop("_feature_config", None)
        original_config = state.pop("feature_config", None)
        catboost_params = state.pop("_catboost_params", None)
        original_catboost = state.pop("catboost_params", None)
        super().__setstate__(state)
        self.feature_config = original_config
        self.feature_config_ = feature_config
        self.catboost_params = original_catboost
        self.catboost_params_ = catboost_params
        if feature_config is not None:
            type(self).feature_config = feature_config
        if catboost_params is not None:
            type(self).catboost_params = catboost_params


# ---- 1) Minimal routing-aware wrapper
class CBC(BaseEstimator, ClassifierMixin):
    def __init__(self, *, cat_features, **cb_params):
        """Initialize the CatBoost wrapper with environment-aware settings."""
        if detect_execution_environment()[0] in (
            "sagemaker_notebook",
            "sagemaker_job",
            "sagemaker_terminal",
        ):
            train_dir = get_output_dir()
            cb_params["train_dir"] = train_dir
            cb_params["allow_writing_files"] = True
            self._callbacks = None
        else:
            self._callbacks = [MlflowCallback()]
        self.cb_params = cb_params
        self.cat_features = cat_features
        self._cb = None
        self.is_fitted_ = False

    # sklearn will route eval_set here if we request it on the instance
    def fit(self, X, y=None, eval_set=None, **fit_kwargs):
        """Fit the wrapped ``CatBoostClassifier`` and mirror fitted attrs."""
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
        self.is_fitted_ = True
        for attr in ("classes_", "n_classes_", "feature_names_in_", "n_features_in_"):
            if hasattr(self._cb, attr):
                setattr(self, attr, getattr(self._cb, attr))
        return self

    def __sklearn_is_fitted__(self):
        """Comply with scikit-learn's ``check_is_fitted`` protocol."""
        return bool(getattr(self, "_cb", None) is not None and self._cb.is_fitted())

    # pass-through predict/predict_proba
    def predict(self, X):
        """Predict class labels using the underlying CatBoost model."""
        check_is_fitted(self, attributes=["is_fitted_"])
        return self._cb.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities using the CatBoost estimator."""
        check_is_fitted(self, attributes=["is_fitted_"])
        return self._cb.predict_proba(X)

    def get_feature_importance(self, *args, **kwargs):
        """Expose CatBoost's ``get_feature_importance`` helper."""
        return self._cb.get_feature_importance(*args, **kwargs)

    # make params grid-searchable
    def get_params(self, deep=True):
        """Return constructor parameters for scikit-learn compatibility."""
        params = self.cb_params.copy()
        params["cat_features"] = self.cat_features
        return params

    def set_params(self, **params):
        """Update CatBoost parameters while preserving scikit-learn semantics."""
        if "cat_features" in params:
            self.cat_features = params.pop("cat_features")
        self.cb_params.update(params)
        self._cb = None
        self.is_fitted_ = False
        return self


def build_pipeline(feature_config=None, **kw):
    """Construct the preprocessing and CatBoost pipeline from feature metadata.

    The routine wires together the custom feature-engine transformers defined
    in :mod:`bid_predictor.transform`, injects missing-value indicators,
    outlier cappers, and discretisers based on the ``feature_config`` payload,
    and finally appends the :class:`CBC` estimator with the supplied CatBoost
    parameters.
    """
    if feature_config is None:
        feature_config = _DEFAULT_FEATURE_CONFIG

    selected_features = feature_config["features"]
    categorical_features = feature_config["cat_features"]

    pipeline_feat_add = Pipeline(
        steps=[
            ("flight_code", add_flight_code_transformer),
            ("depart", add_days_b4_depart_transformer),
            ("bid_rank", add_bid_rank_transformer),
            ("group", group_features_transformer),
            ("quantile", quantiles_transformer),
        ],
        transform_input=["eval_set"],
    )

    steps_ft = []
    if "impute_value" in feature_config or "impute_median" in feature_config:
        variables = []
        if "impute_value" in feature_config:
            for feature, value in feature_config.get("impute_value", []):
                variables.append(feature)
        if "impute_median" in feature_config:
            for feature in feature_config.get("impute_median", []):
                variables.append(feature)

        if variables:
            steps_ft.append(
                (
                    "indicator",
                    AddMissingIndicatorCustom(
                        variables=variables,
                    ),
                )
            )

    if "impute_value" in feature_config:
        imputer_dict = {}
        for feature, value in feature_config.get("impute_value", []):
            imputer_dict[feature] = value

        if imputer_dict:
            steps_ft.append(
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
            steps_ft.append(
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
        for feature, outlier in feature_config.get("outlier", []):
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
            steps_ft.append(
                (
                    "outliers",
                    ArbitraryOutlierCapperCustom(
                        **outlier_args,
                    ),
                )
            )

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
            steps_ft.append(
                (
                    "discrete",
                    ArbitraryDiscretiserCustom(
                        binning_dict=binning_dict,
                        return_boundaries=True,
                    ),
                )
            )

    pipeline_tf = Pipeline(steps=steps_ft, transform_input=["eval_set"])

    reduce_features_transformer = ColumnReducer(selected_features)
    steps= [
        ("feature_add", pipeline_feat_add),
        ("feature_transform", pipeline_tf),
        ("reduce", reduce_features_transformer)
    ]

    # CatBoostClassifier integrates with sklearn API
    clf = CBC(
        cat_features=categorical_features,
        **kw,
    ).set_fit_request(eval_set=True)
    steps.append(("clf", clf))

    pipeline = FeatureConfiguredPipeline(
        steps=steps,
        transform_input=["eval_set"],
        feature_config=feature_config,
    )
    pipeline._assign_catboost_params(clf.cb_params)
    return pipeline
