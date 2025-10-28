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
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn import set_config
from skopt import Optimizer

from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.tuning.cross_validation import (
    RandomDateSplitter,
    cross_validate_with_eval,
)
from bid_predictor.data import load_training_data, resolve_train_file
from bid_predictor.feature_config import load_feature_config
from bid_predictor.tuning.feature_tuning import (
    apply_transform_overrides,
    clone_feature_metadata,
    rebuild_feature_config,
    split_combination,
    summarize_transform_params,
)
from bid_predictor.tuning.search_config import load_search_config
from bid_predictor.tuning.search_grid import build_search_space, unwrap_search_value
from bid_predictor.tuning.mlflow_logging import mlflow_run
from bid_predictor.tuning.result_writing import (
    write_best_catboost_params,
    write_best_feature_config,
    write_best_result_json,
    write_results_csv,
)
from bid_predictor.data import prepare_features


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


def _normalize_structure(value: Any) -> Any:
    """Recursively convert numpy/scientific scalars to built-in Python types."""

    value = unwrap_search_value(value)

    if isinstance(value, dict):
        return {key: _normalize_structure(sub_value) for key, sub_value in value.items()}

    if isinstance(value, list):
        return [_normalize_structure(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_normalize_structure(item) for item in value)

    if isinstance(value, set):
        return sorted(_normalize_structure(item) for item in value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def normalize_search_value(value: Any) -> Any:
    """Convert optimizer suggestions into plain Python scalars."""

    return _normalize_structure(value)


def stringify_param_value(value: Any) -> Any:
    """Prepare parameter values for MLflow logging."""
    value = _normalize_structure(value)
    if isinstance(value, (np.bool_,)):
        value = bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


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
        help=(
            "Number of random date-based splits to evaluate (default: 3)."
        ),
    )
    parser.add_argument(
        "--split-sample-size",
        type=int,
        default=None,
        help=(
            "Approximate number of rows to include across the sampled training "
            "and evaluation windows. Defaults to the full dataset."
        ),
    )
    parser.add_argument(
        "--eval-train-ratio",
        type=float,
        default=1.0,
        help=(
            "Target ratio of evaluation rows to training rows when sampling "
            "date windows (default: 1.0)."
        ),
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
        help=(
            "Maximum number of Bayesian optimization iterations to evaluate. "
            "If omitted, a default of 25 iterations is used unless the search "
            "space is constant."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write a CSV summary of all evaluated configurations.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Optional path to write the best configuration summary as JSON.",
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
    parser.add_argument(
        "--best-catboost-out",
        type=str,
        default=None,
        help=(
            "Optional path to write the CatBoost hyperparameters (YAML) corresponding "
            "to the best observed score. Only parameters defined in the search config "
            "are included."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help=(
            "Optional MLflow experiment name to use when logging tuning metrics. "
            "Ignored when MLflow is unavailable."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    search_cfg = load_search_config(args.search_config)
    search_dimensions = build_search_space(search_cfg)
    search_catboost_keys: Sequence[str] = tuple(search_cfg.get("catboost", {}).keys())
    param_names: Sequence[str] = [name for name, _ in search_dimensions]
    dimensions = [dim for _, dim in search_dimensions]

    all_constant = all(getattr(dim, "is_constant", False) for dim in dimensions)

    requested_iterations = args.max_combinations if args.max_combinations is not None else 25
    total_combinations = 1 if all_constant else max(1, requested_iterations)

    n_initial_points = min(total_combinations, max(1, min(10, len(dimensions) * 2)))
    optimizer = Optimizer(
        dimensions,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=n_initial_points,
        initial_point_generator="lhs",
        random_state=args.random_state,
    )

    feature_config = load_feature_config(args.feature_config)
    base_metadata = clone_feature_metadata(feature_config["feature_metadata"])

    train_file = resolve_train_file(args.train_file)
    data = load_training_data(train_file)

    X_train, _, y_train, _, _ = prepare_features(
        data, feature_config["pre_features"], testing=args.testing
    )

    train_indices = X_train.index
    travel_dates = pd.to_datetime(
        data.loc[train_indices, "travel_date"]
    ).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    cv = RandomDateSplitter(
        travel_dates=travel_dates,
        n_splits=args.cv_splits,
        total_size=args.split_sample_size,
        eval_ratio=args.eval_train_ratio,
        random_state=args.random_state,
    )

    static_cat_params = {
        "task_type": args.task_type,
        "devices": args.devices,
        "random_seed": args.random_state,
        "logging_level": "Silent",
        "custom_metric": ["AUC"],
    }

    with mlflow_run(args.mlflow_experiment) as mlflow_logger:
        records: List[Dict[str, Any]] = []
        best_score = float("-inf")
        best_result: Optional[Dict[str, Any]] = None
        best_feature_config: Optional[Dict[str, Any]] = None
        best_catboost_params: Optional[Dict[str, Any]] = None

        if mlflow_logger.enabled:
            mlflow_logger.log_params(
                {
                    "scoring": args.scoring,
                    "cv_splits": args.cv_splits,
                    "task_type": args.task_type,
                    "devices": args.devices,
                    "random_state": args.random_state,
                    "total_iterations": total_combinations,
                    "split_sample_size": args.split_sample_size,
                    "eval_train_ratio": args.eval_train_ratio,
                }
            )

        for idx in range(total_combinations):
            suggestion = optimizer.ask()
            combination = {
                name: normalize_search_value(value)
                for name, value in zip(param_names, suggestion)
            }

            cat_params, transform_overrides = split_combination(combination)
            cat_params = {
                key: normalize_search_value(value) for key, value in cat_params.items()
            }

            metadata = clone_feature_metadata(base_metadata)
            apply_transform_overrides(metadata, transform_overrides)
            tuned_config = rebuild_feature_config(metadata)

            pipeline = build_pipeline(
                feature_config=tuned_config,
                **{**static_cat_params, **cat_params},
            )

            fold_scores = cross_validate_with_eval(
                pipeline, X_train, y_train, cv, args.scoring
            )
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))

            record: Dict[str, Any] = {
                "mean_score": mean_score,
                "std_score": std_score,
            }
            for param, value in cat_params.items():
                record[f"catboost.{param}"] = _normalize_structure(value)
            record.update(summarize_transform_params(transform_overrides))
            records.append(record)

            print(
                f"[{idx + 1}/{total_combinations}] score={mean_score:.4f} ± {std_score:.4f} "
                f"catboost={cat_params} transforms={transform_overrides}",
                flush=True,
            )

            if mlflow_logger.enabled:
                mlflow_logger.log_metric("cv_mean_score", mean_score, step=idx)
                mlflow_logger.log_metric("cv_std_score", std_score, step=idx)

            optimizer.tell(suggestion, -mean_score)

            if mean_score > best_score:
                best_score = mean_score
                best_result = _normalize_structure(
                    {
                        "score": mean_score,
                        "std": std_score,
                        "catboost": {**static_cat_params, **cat_params},
                        "transforms": transform_overrides,
                    }
                )
                best_feature_config = tuned_config
                best_catboost_params = {
                    key: _normalize_structure(cat_params[key])
                    for key in search_catboost_keys
                    if key in cat_params
                }

        if not records:
            raise RuntimeError("No parameter combinations were evaluated.")

        results_df = pd.DataFrame(records)
        results_df.sort_values("mean_score", ascending=False, inplace=True)

        print("\nTop configurations:")
        print(results_df.head(min(10, len(results_df))).to_string(index=False))

        if args.output_csv:
            output_path = write_results_csv(args.output_csv, results_df)
            print(f"Saved results to {output_path}")
            if mlflow_logger.enabled:
                mlflow_logger.log_artifact(str(output_path))

        if args.best_config_out and best_feature_config is not None:
            config_path = write_best_feature_config(args.best_config_out, best_feature_config)
            print(f"Saved best feature configuration to {config_path}")
            if mlflow_logger.enabled:
                mlflow_logger.log_artifact(str(config_path))

        if args.best_catboost_out and best_catboost_params is not None:
            catboost_path = write_best_catboost_params(args.best_catboost_out, best_catboost_params)
            print(f"Saved best CatBoost parameters to {catboost_path}")
            if mlflow_logger.enabled:
                mlflow_logger.log_artifact(str(catboost_path))

        if best_result is not None:
            if mlflow_logger.enabled:
                mlflow_logger.log_metric("best_score", best_result["score"])
                mlflow_logger.log_metric("best_std", best_result["std"])
                flat_params = {
                    f"best.{key}": stringify_param_value(value)
                    for key, value in {
                        **{
                            f"catboost.{param}": val
                            for param, val in best_result["catboost"].items()
                        },
                        **{
                            key: val
                            for key, val in summarize_transform_params(
                                best_result["transforms"]
                            ).items()
                        },
                    }.items()
                }
                mlflow_logger.log_params(flat_params)

            json_ready_result = _normalize_structure(best_result)

            if args.results_json:
                json_path = write_best_result_json(args.results_json, json_ready_result)
                print(f"Saved best result summary to {json_path}")
                if mlflow_logger.enabled:
                    mlflow_logger.log_artifact(str(json_path))

            print("\nBest result:")
            print(json.dumps(json_ready_result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
