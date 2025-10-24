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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
import sklearn
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate

from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.feature_config import _ensure_groupby_keys, load_feature_config
from bid_predictor.utils import detect_execution_environment
from train import prepare_features


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


def main() -> None:
    args = parse_args()

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

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=args.scoring,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        score_key = f"test_{args.scoring}"
        if score_key not in scores:
            available = ", ".join(sorted(scores.keys()))
            raise KeyError(
                f"Scoring key '{score_key}' not returned by cross_validate. Available: {available}"
            )

        mean_score = float(np.mean(scores[score_key]))
        std_score = float(np.std(scores[score_key]))

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
    sklearn.set_config(enable_metadata_routing=True)
    main()
