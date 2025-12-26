import os
import sys
import argparse
import warnings
import inspect
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set

import pandas as pd
import yaml
import mlflow
import pyarrow.dataset as ds
import sklearn
import joblib
from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.feature_config import load_feature_config, _GROUPBY_KEY_FEATURES
from bid_predictor.tracking import (
    log_evaluation_results_parquet,
    log_classification_metrics,
    log_classification_metrics_by_time,
    log_prob_examples,
    log_evaluation_figures,
    log_feature_config_artifacts,
    log_feature_importances,
    log_pipeline_model,
    log_run_parameters,
    start_catboost_mlflow_stream,
)
from bid_predictor.data import load_dataset_cached, prepare_features
from bid_predictor.utils import detect_execution_environment, get_output_dir
from catboost import CatBoostClassifier
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=(
        "This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments "
        "before using other methods such as transform, predict, etc. This will raise an "
        "error in 1.8 instead of the current warning."
    ),
    category=FutureWarning,
)

if detect_execution_environment()[0] in (
    "sagemaker_notebook",
    "sagemaker_job",
    "sagemaker_terminal",
):
    arn = os.environ["MLFLOW_AWS_ARN"]
    mlflow.set_tracking_uri(arn)

DEFAULT_EXP_NAME = "tests"


def _introspect_catboost_defaults() -> Dict[str, Any]:
    signature = inspect.signature(CatBoostClassifier.__init__)
    defaults: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        defaults[name] = None if param.default is inspect._empty else param.default
    return defaults


_CATBOOST_PARAM_DEFAULTS = _introspect_catboost_defaults()
CATBOOST_PARAM_NAMES = tuple(_CATBOOST_PARAM_DEFAULTS.keys())
CATBOOST_FLAG_MAP = {
    name: f"--{name.replace('_', '-')}" for name in CATBOOST_PARAM_NAMES
}


def _parse_catboost_cli_value(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return value


def _register_catboost_arguments(parser: argparse.ArgumentParser) -> None:
    for name in CATBOOST_PARAM_NAMES:
        flag = CATBOOST_FLAG_MAP[name]
        parser.add_argument(
            flag,
            dest=name,
            type=str if name=="devices" else _parse_catboost_cli_value,
            default=_CATBOOST_PARAM_DEFAULTS[name],
        )


def _detect_explicit_flags(argv: Iterable[str]) -> Set[str]:
    explicit: Set[str] = set()
    for name, flag in CATBOOST_FLAG_MAP.items():
        for arg in argv:
            if arg == flag or arg.startswith(f"{flag}="):
                explicit.add(name)
                break
    return explicit


def _load_catboost_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"CatBoost configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if isinstance(payload, Mapping) and "catboost" in payload:
        payload = payload["catboost"]

    if not isinstance(payload, Mapping):
        raise ValueError(
            "CatBoost configuration must be a mapping of parameter names to values"
        )

    return {str(key): value for key, value in payload.items()}


def _merge_catboost_params(
    cli_params: Mapping[str, Any],
    config_params: Mapping[str, Any],
    explicit_flags: Set[str],
) -> Dict[str, Any]:
    merged = dict(cli_params)
    for key, value in config_params.items():
        if key in explicit_flags:
            continue
        merged[key] = value
    return merged


def _persist_model_artifacts(pipeline) -> None:
    """Persist the trained pipeline and inference entry point for SageMaker."""

    if detect_execution_environment()[0] != "sagemaker_job":
        return

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)

    # Copy the "bid_predictor" directory next to pipeline.joblib
    bid_predictor_src = Path(__file__).resolve().with_name("bid_predictor")
    if bid_predictor_src.exists() and bid_predictor_src.is_dir():
        bid_predictor_dst = target_dir / "bid_predictor"

        # If it already exists from a previous run, replace it
        if bid_predictor_dst.exists():
            shutil.rmtree(bid_predictor_dst)

        shutil.copytree(bid_predictor_src, bid_predictor_dst)

    print(f"Saved model to {model_dir}")       


def parse_args():
    p = argparse.ArgumentParser()
    _register_catboost_arguments(p)
    p.set_defaults(
        task_type="CPU",
        devices="0",
        iterations=200,
        depth=6,
        learning_rate=None,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        eval_metric="AUC",
        random_state=42,
    )
    # your own toggles
    p.add_argument("--feature-config", type=str, default=None)
    p.add_argument("--experiment-name", type=str, default=DEFAULT_EXP_NAME)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--testing", action="store_true")
    p.add_argument(
        "--catboost-config",
        type=str,
        default=None,
        help=(
            "Optional YAML file containing CatBoost hyperparameters exported from the "
            "tuning script. Command-line arguments override values from this file."
        ),
    )

    args = p.parse_args()
    setattr(args, "_explicit_flags", _detect_explicit_flags(sys.argv[1:]))
    return args


def train_and_log_model(
    data,
    feature_config,
    args,
):
    run_name = f"run-{pd.Timestamp.now():%Y-%m-%d-%H-%M-%S}"

    cat_features = list(feature_config["cat_features"])
    features = list(feature_config["features"])
    pre_features = feature_config["pre_features"]

    var_args = vars(args)
    testing = var_args.pop("testing")
    experiment_name = var_args.pop("experiment_name", DEFAULT_EXP_NAME)
    run_name = var_args.pop("run_name", run_name)
    explicit_flags = set(var_args.pop("_explicit_flags", set()))
    catboost_config_path = var_args.pop("catboost_config", None)

    X_train, X_test, y_train, y_test, bid_prob_test_results = prepare_features(
        data, pre_features, testing=testing
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        nested=(mlflow.active_run() is not None),
        log_system_metrics=True,
    ) as run:
        catboost_file_params = _load_catboost_config(catboost_config_path)

        if catboost_file_params.get("monotone_constraints") is None and any(
            feature_config.get("monotone_constraints", [])
        ):
            catboost_file_params["monotone_constraints"] = feature_config.get(
                "monotone_constraints", []
            )

        cli_catboost_params = {
            name: var_args.get(name)
            for name in CATBOOST_PARAM_NAMES
            if name in var_args
        }
        merged_catboost_params = _merge_catboost_params(
            cli_catboost_params, catboost_file_params, explicit_flags
        )

        for key, value in merged_catboost_params.items():
            var_args[key] = value

        log_run_parameters(var_args, features, cat_features, len(X_train), len(X_test))
        log_feature_config_artifacts(feature_config)

        catboost_kwargs = var_args.copy()
        catboost_kwargs.pop("feature_config", None)
        catboost_kwargs.pop("cat_features", None)
        catboost_kwargs["train_dir"] = get_output_dir()
        catboost_kwargs["allow_writing_files"] = True

        pipeline = build_pipeline(feature_config=feature_config, **catboost_kwargs)

        if detect_execution_environment()[0] in (
            "sagemaker_notebook",
            "sagemaker_job",
            "sagemaker_terminal",
        ):  # == "sagemaker_job":
            train_dir = pipeline[-1].cb_params.get(
                "train_dir", "/opt/ml/output/catboost"
            )

            # start streaming before fit
            stop_stream = start_catboost_mlflow_stream(train_dir, run.info.run_id)

            try:
                pipeline.fit(X_train, y_train, eval_set=(X_test, y_test))
            finally:
                stop_stream()
        else:
            pipeline.fit(X_train, y_train, eval_set=(X_test, y_test))

        proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        bid_prob_test_results["Acceptance Probability"] = proba

        log_evaluation_results_parquet(bid_prob_test_results.reset_index())
        log_classification_metrics(y_test, y_pred)
        log_classification_metrics_by_time(bid_prob_test_results)
        for carrier in ["AC", "LO"]:
            if carrier in bid_prob_test_results.index.levels[0]:
                log_classification_metrics_by_time(
                    bid_prob_test_results.loc[carrier],
                    figure_path=f"classification_metrics_vs_time_utill_departure_{carrier.lower()}.png",
                )
        log_prob_examples(bid_prob_test_results)
        log_feature_importances(pipeline, X_train, y_train, cat_features)
        log_evaluation_figures(y_test, y_pred, proba)
        log_pipeline_model(pipeline)
        _persist_model_artifacts(pipeline)


def main():
    if detect_execution_environment()[0] == "sagemaker_job":
        train_file = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    elif detect_execution_environment()[0] in (
        "sagemaker_notebook",
        "sagemaker_terminal",
    ):
        train_file = os.environ.get("S3_BUCKET_DATA") + "/data"
        # train_file += "/air_canada_and_lot/bid_data_snapshots_v2.parquet"
        # train_file += "/etihad/bid_and_flight_data_snapshots_etihad_v2.parquet"
        train_file += "/saudia/bid_and_flight_data_snapshots.parquet"
    else:
        train_file = "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
        # train_file = "../bid_data_snapshots_v2.parquet"

    data = load_dataset_cached(train_file)

    args = parse_args()
    feature_config = load_feature_config(args.feature_config)

    train_and_log_model(
        data,
        feature_config,
        args,
    )


if __name__ == "__main__":
    sklearn.set_config(enable_metadata_routing=True)
    main()
