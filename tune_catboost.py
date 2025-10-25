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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover - mlflow is optional during import
    mlflow = None

import numpy as np
import pandas as pd
import yaml
from sklearn import set_config
from sklearn.model_selection import StratifiedKFold
from skopt import Optimizer

from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.tuning.cross_validation import cross_validate_with_eval
from bid_predictor.tuning.data_access import load_training_data, resolve_train_file
from bid_predictor.feature_config import load_feature_config
from bid_predictor.tuning.feature_tuning import (
    apply_transform_overrides,
    clone_feature_metadata,
    rebuild_feature_config,
    split_combination,
    summarize_transform_params,
)
from bid_predictor.tuning.search_config import load_search_config
from bid_predictor.tuning.search_grid import build_search_space
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
        "--best-config-out",
        type=str,
        default=None,
        help=(
            "Optional path to write the feature configuration (YAML) for the best "
            "observed score."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _patch_mlflow_metric_logging()

    search_cfg = load_search_config(args.search_config)
    search_dimensions = build_search_space(search_cfg)
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

    def _normalize_value(value: Any) -> Any:
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    for idx in range(total_combinations):
        suggestion = optimizer.ask()
        combination = {
            name: _normalize_value(value)
            for name, value in zip(param_names, suggestion)
        }

        cat_params, transform_overrides = split_combination(combination)
        cat_params = {key: _normalize_value(value) for key, value in cat_params.items()}

        metadata = clone_feature_metadata(base_metadata)
        apply_transform_overrides(metadata, transform_overrides)
        tuned_config = rebuild_feature_config(metadata)

        pipeline = build_pipeline(
            feature_config=tuned_config,
            **{**static_cat_params, **cat_params},
        )

        fold_scores = cross_validate_with_eval(pipeline, X_train, y_train, cv, args.scoring)
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

        optimizer.tell(suggestion, -mean_score)

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
