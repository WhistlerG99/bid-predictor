import os
import argparse
import pandas as pd
import mlflow
import pyarrow.dataset as ds
import sklearn
from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.feature_config import load_feature_config
from bid_predictor.tracking import (
    log_classification_metrics,
    log_evaluation_figures,
    log_feature_config_artifacts,
    log_feature_importances,
    log_pipeline_model,
    log_run_parameters,
    start_catboost_mlflow_stream,
)
from bid_predictor.utils import detect_execution_environment
from dotenv import load_dotenv

load_dotenv()
if detect_execution_environment()[0] in (
    "sagemaker_notebook",
    "sagemaker_job",
    "sagemaker_terminal",
):
    arn = os.environ["MLFLOW_AWS_ARN"]
    mlflow.set_tracking_uri(arn)

DEFAULT_EXP_NAME = "tests"


def parse_args():
    p = argparse.ArgumentParser()
    # CatBoost knobs
    p.add_argument("--task-type", type=str, default="CPU")  # "GPU" to use GPU
    p.add_argument("--devices", type=str, default="0")  # "0", "0,1", etc.
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--l2-leaf-reg", type=float, default=3.0)
    # your own toggles
    p.add_argument("--eval-metric", type=str, default="AUC")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--feature-config", type=str, default=None)
    p.add_argument("--experiment-name", type=str, default=DEFAULT_EXP_NAME)
    p.add_argument("--testing", action="store_true")
    return p.parse_args()


def prepare_features(data, pre_features, testing=True):
    data = data[
        data.departure_timestamp - data.current_timestamp < pd.to_timedelta("5d")
    ]

    data.sort_values(["travel_date", "carrier_code", "flight_number"]).reset_index(
        drop=True
    )

    available_pre_features = [
        feature for feature in pre_features if feature in data.columns
    ]
    selection_columns = list(dict.fromkeys(available_pre_features + ["offer_status"]))

    if testing:
        cutoff = "2023-08-01"
        yX_test = data[
            (data.travel_date >= cutoff) & (data.travel_date <= "2023-08-15")
        ][selection_columns]
    else:
        cutoff = "2025-05-01"
        yX_test = data[data.travel_date >= cutoff][selection_columns]
    yX_train = data[data.travel_date < cutoff][selection_columns]

    X_train, X_test = yX_train[available_pre_features], yX_test[available_pre_features]
    y_train = (yX_train["offer_status"] == "TICKETED").astype(int)
    y_test = (yX_test["offer_status"] == "TICKETED").astype(int)
    return X_train, X_test, y_train, y_test


def train_and_log_model(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_config,
    args,
):
    cat_features = list(feature_config["cat_features"])
    features = list(feature_config["features"])

    var_args = vars(args)

    experiment_name = var_args.pop("experiment_name", DEFAULT_EXP_NAME)
    mlflow.set_experiment(experiment_name)
    run_name = f"catboost_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(
        run_name=run_name,
        nested=(mlflow.active_run() is not None),
        log_system_metrics=True,
    ) as run:
        log_run_parameters(var_args, features, cat_features, len(X_train), len(X_test))
        log_feature_config_artifacts(feature_config)

        catboost_kwargs = var_args.copy()
        catboost_kwargs.pop("feature_config", None)

        pipeline = build_pipeline(feature_config=feature_config, **catboost_kwargs)

        if detect_execution_environment()[0] in (
            "sagemaker_notebook",
            "sagemaker_job",
            "sagemaker_terminal",
        ):  # == "sagemaker_job":
            train_dir = pipeline[-1].cb_params.get(
                "train_dir", "/opt/ml/output/catboost"
            )
            # train_dir = os.environ.get("CATBOOST_TRAIN_DIR", "/opt/ml/output/catboost")

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
        log_classification_metrics(y_test, y_pred)
        log_feature_importances(pipeline, X_train, y_train, cat_features)
        log_evaluation_figures(y_test, y_pred, proba)
        log_pipeline_model(pipeline)


def main():
    if detect_execution_environment()[0] == "sagemaker_job":
        train_file = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    elif detect_execution_environment()[0] in (
        "sagemaker_notebook",
        "sagemaker_terminal",
    ):
        train_file = (
            os.environ.get("S3_BUCKET_DATA")
            + "/data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
        )
    else:
        train_file = "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
        # train_file = "../bid_data_snapshots_v2.parquet"

    dataset = ds.dataset(
        train_file, format="parquet"
    )  # auto-detects partitions (Hive-style)
    table = dataset.to_table()  # optionally: .to_table(columns=["col1","col2"])
    data = table.to_pandas()

    # Make these categorical
    for col in ["carrier_code", "flight_number", "fare_class"]:
        data[col] = data[col].astype("category")

    data = data.rename(
        columns={"current_available_seats": "seats_available"}, errors="ignore"
    )

    args = parse_args()
    feature_config = load_feature_config(args.feature_config)
    pre_features = feature_config["pre_features"]
    testing = vars(args).pop("testing")

    X_train, X_test, y_train, y_test = prepare_features(data, pre_features, testing)
    train_and_log_model(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_config,
        args,
    )


if __name__ == "__main__":
    sklearn.set_config(enable_metadata_routing=True)
    main()
