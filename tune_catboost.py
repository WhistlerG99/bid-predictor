"""Hyperparameter tuning utility for the CatBoost bid predictor pipeline.

The script evaluates cross-validated combinations of CatBoost hyperparameters
and feature transformation options (imputation, binning, outlier capping) that
are derived from the feature configuration metadata. Users can provide a YAML
or JSON search configuration to explore custom grids for both the model and the
preprocessing steps.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - mlflow is optional during import
    mlflow = None

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
from sklearn import set_config
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.feature_config import _ensure_groupby_keys, load_feature_config
from bid_predictor.utils import detect_execution_environment
from train import prepare_features


set_config(enable_metadata_routing=True)

warnings.filterwarnings(
    "ignore",
    message=(
        "This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments "
        "before using other methods such as transform, predict, etc. This will raise an "
        "error in 1.8 instead of the current warning."
    ),
    category=FutureWarning,
)


_DEFAULT_SEARCH_CONFIG = {
    "catboost": {
        "iterations": [200, 400],
        "depth": [6, 8],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [3.0, 5.0],
    },
    "transform": {
        "impute_median": {},
        "impute_value": {},
        "outlier": {},
        "bins": {},
    },
}


def _patch_mlflow_metric_logging() -> None:
    if mlflow is None:
        return

    log_metric = getattr(mlflow, "log_metric", None)
    if log_metric is None:
        return

    if getattr(log_metric, "_bid_predictor_wrapped", False):
        return

    permission_state = {"warned": False}

    def safe_log_metric(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return log_metric(*args, **kwargs)
        except PermissionError as exc:  # pragma: no cover - environment specific
            if not permission_state["warned"]:
                permission_state["warned"] = True
                print(
                    "Warning: Skipping MLflow metric logging due to permission error. "
                    f"Details: {exc}",
                    flush=True,
                )
            return None

    safe_log_metric._bid_predictor_wrapped = True  # type: ignore[attr-defined]
    mlflow.log_metric = safe_log_metric  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validated hyperparameter tuning for the CatBoost bid predictor. "
            "Use --search-config to provide a YAML/JSON parameter grid that "
            "includes CatBoost options and feature transformation overrides."
        )
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help=(
            "Path to the training parquet dataset. If omitted, the script will "
            "use the same environment-aware defaults as train.py."
        ),
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help="Path to the feature configuration YAML (defaults to package config).",
    )
    parser.add_argument(
        "--search-config",
        type=str,
        default=None,
        help=(
            "YAML/JSON file describing the hyperparameter grid. The file should "
            "define 'catboost' and optional 'transform' sections."
        ),
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3).",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Scikit-learn scoring metric to optimize (default: roc_auc).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-validation shuffling and CatBoost (default: 42).",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="CPU",
        help="CatBoost task type (CPU or GPU).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Device identifiers for CatBoost when using GPU execution.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=(
            "Reproduce the smaller testing split from train.py (cutoff 2023-08-01). "
            "By default the full training range before 2025-05-01 is used."
        ),
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Optional hard limit on the number of parameter combinations to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write a CSV summary of all evaluated configurations.",
    )
    parser.add_argument(
        "--best-config-out",
        type=str,
        default=None,
        help=(
            "Optional path to write the feature configuration (YAML) for the best "
            "observed score."
        ),
    )
    return parser.parse_args()


def _ensure_list(values: Any) -> List[Any]:
    if isinstance(values, (list, tuple, np.ndarray)):
        return list(values)
    return [values]


def load_search_config(path: Optional[str]) -> Mapping[str, Any]:
    if path is None:
        return deepcopy(_DEFAULT_SEARCH_CONFIG)

    search_path = Path(path)
    if not search_path.is_file():
        raise FileNotFoundError(f"Search configuration file not found: {search_path}")

    with search_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    catboost_cfg = config.get("catboost", {})
    transform_cfg = config.get("transform", {})

    # Normalize shape even if user omits sections.
    transform_cfg.setdefault("impute_value", {})
    transform_cfg.setdefault("impute_median", {})
    transform_cfg.setdefault("outlier", {})
    transform_cfg.setdefault("bins", {})

    return {
        "catboost": catboost_cfg,
        "transform": transform_cfg,
    }


def build_parameter_grid(search_cfg: Mapping[str, Any]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}

    for param, values in search_cfg.get("catboost", {}).items():
        values_list = _ensure_list(values)
        if not values_list:
            continue
        grid[f"catboost__{param}"] = values_list

    transform_cfg: Mapping[str, Mapping[str, Iterable[Any]]] = search_cfg.get(
        "transform", {}
    )
    for section in ("impute_value", "impute_median", "outlier", "bins"):
        section_cfg = transform_cfg.get(section, {}) or {}
        for feature, values in section_cfg.items():
            values_list = _ensure_list(values)
            if not values_list:
                continue
            grid[f"transform__{section}__{feature}"] = values_list

    if not grid:
        grid["__noop__"] = [None]

    return grid


def split_combination(
    combination: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
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


def _clone_feature_metadata(
    feature_metadata: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    return {name: dict(values) for name, values in feature_metadata.items()}


def _rebuild_feature_config(
    feature_metadata: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
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


def _apply_transform_overrides(
    metadata: MutableMapping[str, MutableMapping[str, Any]],
    overrides: Mapping[str, Dict[str, Any]],
) -> None:
    for section, feature_map in overrides.items():
        for feature, value in feature_map.items():
            if feature not in metadata:
                raise KeyError(f"Feature '{feature}' not found in feature configuration")
            if section == "impute_median":
                metadata[feature][section] = bool(value)
            else:
                metadata[feature][section] = value


def resolve_train_file(arg_value: Optional[str]) -> str:
    if arg_value:
        return arg_value

    env, _ = detect_execution_environment()
    if env == "sagemaker_job":
        return os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    if env in {"sagemaker_notebook", "sagemaker_terminal"}:
        bucket = os.environ.get("S3_BUCKET_DATA")
        if not bucket:
            raise RuntimeError(
                "S3_BUCKET_DATA environment variable must be set for SageMaker environments"
            )
        return bucket + "/data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
    return "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"


def load_training_data(train_path: str) -> pd.DataFrame:
    dataset = ds.dataset(train_path, format="parquet")
    table = dataset.to_table()
    data = table.to_pandas()

    for col in ["carrier_code", "flight_number", "fare_class"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    data = data.rename(columns={"current_available_seats": "seats_available"}, errors="ignore")
    return data


def summarize_transform_params(overrides: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for section, feature_map in overrides.items():
        for feature, value in feature_map.items():
            key = f"{section}.{feature}"
            if isinstance(value, (dict, list)):
                summary[key] = json.dumps(value, sort_keys=True)
            else:
                summary[key] = value
    return summary


def _split_indices(
    X: Any, y: Any, indices: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Any, Any, Any, Any]:
    train_idx, val_idx = indices

    if hasattr(X, "iloc"):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
    else:
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]

    if hasattr(y, "iloc"):
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
    else:
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

    return X_train_fold, X_val_fold, y_train_fold, y_val_fold


def _call_prediction_interface(model, method: str, X_val):
    """Invoke the requested prediction interface on a fitted estimator."""

    if hasattr(model, method):
        predictor = getattr(model, method)
        return predictor(X_val)

    if hasattr(model, "steps"):
        Xt = X_val
        for _, transformer in model.steps[:-1]:
            if transformer in (None, "passthrough"):
                continue
            if hasattr(transformer, "transform"):
                Xt = transformer.transform(Xt)
            elif callable(transformer):
                Xt = transformer(Xt)
            else:
                raise AttributeError(
                    "Pipeline step does not expose a transform method"
                )

        final_estimator = model.steps[-1][1]
        predictor = getattr(final_estimator, method)
        return predictor(Xt)

    predictor = getattr(model, method)
    return predictor(X_val)


def _score_fold(model, scorer, X_val_fold, y_val_fold) -> float:
    """Evaluate a fitted estimator on the validation fold.

    The helper mirrors scikit-learn's scorer logic but bypasses the internal
    response-method validation that incorrectly flags the CatBoost pipeline as a
    regressor. We rely on the scorer's factory arguments to determine the
    expected prediction interface.
    """

    factory_args = getattr(scorer, "_factory_args", {})
    needs_proba = factory_args.get("needs_proba", False)
    needs_threshold = factory_args.get("needs_threshold", False)

    if needs_proba:
        try:
            y_pred = _call_prediction_interface(model, "predict_proba", X_val_fold)
        except AttributeError as exc:
            raise ValueError(
                "Scorer requires predict_proba but estimator lacks it"
            ) from exc
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    elif needs_threshold:
        try:
            y_pred = _call_prediction_interface(model, "predict_proba", X_val_fold)
            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
        except AttributeError:
            try:
                y_pred = _call_prediction_interface(
                    model, "decision_function", X_val_fold
                )
            except AttributeError:
                y_pred = _call_prediction_interface(model, "predict", X_val_fold)
    else:
        y_pred = _call_prediction_interface(model, "predict", X_val_fold)

    return float(
        scorer._sign * scorer._score_func(y_val_fold, y_pred, **scorer._kwargs)
    )


def _cross_validate_with_eval(
    estimator, X: Any, y: Any, cv: StratifiedKFold, scoring: str
) -> List[float]:
    scorer = get_scorer(scoring)
    scores: List[float] = []

    for split in cv.split(X, y):
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = _split_indices(X, y, split)

        model = clone(estimator)
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))
        fold_score = _score_fold(model, scorer, X_val_fold, y_val_fold)
        scores.append(fold_score)

    return scores


def main() -> None:
    args = parse_args()

    _patch_mlflow_metric_logging()

    search_cfg = load_search_config(args.search_config)
    param_grid_dict = build_parameter_grid(search_cfg)
    param_grid = list(ParameterGrid(param_grid_dict))
    total_combinations = len(param_grid)

    if args.max_combinations is not None:
        total_combinations = min(total_combinations, args.max_combinations)

    feature_config = load_feature_config(args.feature_config)
    base_metadata = _clone_feature_metadata(feature_config["feature_metadata"])

    train_file = resolve_train_file(args.train_file)
    data = load_training_data(train_file)

    X_train, _, y_train, _, _ = prepare_features(
        data, feature_config["pre_features"], testing=args.testing
    )

    cv = StratifiedKFold(
        n_splits=args.cv_splits, shuffle=True, random_state=args.random_state
    )

    static_cat_params = {
        "task_type": args.task_type,
        "devices": args.devices,
        "random_seed": args.random_state,
        "logging_level": "Silent",
        "custom_metric": ["AUC"],
    }

    records: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_result: Optional[Dict[str, Any]] = None
    best_feature_config: Optional[Dict[str, Any]] = None

    for idx, combination in enumerate(param_grid):
        if args.max_combinations is not None and idx >= args.max_combinations:
            break

        cat_params, transform_overrides = split_combination(combination)

        metadata = _clone_feature_metadata(base_metadata)
        _apply_transform_overrides(metadata, transform_overrides)
        tuned_config = _rebuild_feature_config(metadata)

        pipeline = build_pipeline(
            feature_config=tuned_config,
            **{**static_cat_params, **cat_params},
        )

        fold_scores = _cross_validate_with_eval(pipeline, X_train, y_train, cv, args.scoring)
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        record: Dict[str, Any] = {
            "mean_score": mean_score,
            "std_score": std_score,
        }
        for param, value in cat_params.items():
            record[f"catboost.{param}"] = value
        record.update(summarize_transform_params(transform_overrides))
        records.append(record)

        print(
            f"[{idx + 1}/{total_combinations}] score={mean_score:.4f} ± {std_score:.4f} "
            f"catboost={cat_params} transforms={transform_overrides}",
            flush=True,
        )

        if mean_score > best_score:
            best_score = mean_score
            best_result = {
                "score": mean_score,
                "std": std_score,
                "catboost": {**static_cat_params, **cat_params},
                "transforms": transform_overrides,
            }
            best_feature_config = tuned_config

    if not records:
        raise RuntimeError("No parameter combinations were evaluated.")

    results_df = pd.DataFrame(records)
    results_df.sort_values("mean_score", ascending=False, inplace=True)

    print("\nTop configurations:")
    print(results_df.head(min(10, len(results_df))).to_string(index=False))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    if args.best_config_out and best_feature_config is not None:
        config_path = Path(args.best_config_out)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        features_yaml = {
            "features": {
                name: dict(values)
                for name, values in best_feature_config["feature_metadata"].items()
            }
        }
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(features_yaml, fh, sort_keys=True)
        print(f"Saved best feature configuration to {config_path}")

    if best_result is not None:
        print("\nBest result:")
        print(json.dumps(best_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
