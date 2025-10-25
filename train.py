import os
import argparse
import warnings
import inspect
from ast import literal_eval

import pandas as pd
import mlflow
import pyarrow.dataset as ds
import sklearn
from catboost import CatBoostClassifier

from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.feature_config import load_feature_config, _GROUPBY_KEY_FEATURES
from bid_predictor.tracking import (
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
from bid_predictor.utils import detect_execution_environment
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


_CATBOOST_DEFAULTS = {
    "task_type": "CPU",
    "devices": "0",
    "iterations": 200,
    "depth": 6,
    "learning_rate": None,
    "l2_leaf_reg": 3.0,
    "eval_metric": "AUC",
    "random_state": 42,
}


def _parse_catboost_arg(value):
    if isinstance(value, str) and value.lower() == "none":
        return None
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _catboost_param_names():
    signature = inspect.signature(CatBoostClassifier.__init__)
    return [
        name
        for name in signature.parameters
        if name != "self"
    ]


CATBOOST_PARAM_NAMES = _catboost_param_names()


def parse_args():
    p = argparse.ArgumentParser()
    catboost_group = p.add_argument_group("CatBoost hyperparameters")
    for param_name in CATBOOST_PARAM_NAMES:
        option_name = f"--{param_name.replace('_', '-')}"
        default = _CATBOOST_DEFAULTS.get(param_name, None)
        catboost_group.add_argument(
            option_name,
            dest=param_name,
            type=_parse_catboost_arg,
            default=default,
            help=f"Override CatBoost parameter '{param_name}'.",
        )
    # your own toggles
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
    selection_columns = list(dict.fromkeys(available_pre_features + ["offer_status", "id", "decision_timestamp"]))

    if testing:
        cutoff = "2023-08-01"
        yX_test = data[
            (data.travel_date >= cutoff) & (data.travel_date <= "2023-08-15")
        ][selection_columns]
    else:
        cutoff = "2025-05-01"
        yX_test = data.loc[data.travel_date >= cutoff,selection_columns]
    yX_train = data.loc[data.travel_date < cutoff,selection_columns]

    X_train, X_test = yX_train.loc[:,available_pre_features], yX_test.loc[:,available_pre_features]
    y_train = (yX_train["offer_status"] == "TICKETED").astype(int)
    y_test = (yX_test["offer_status"] == "TICKETED").astype(int)

    yX_test.loc[:,"offer_status"] = "Rejected"
    yX_test.loc[y_test==1,"offer_status"] = "Accepted"

    yX_test = yX_test.set_index(_GROUPBY_KEY_FEATURES)# + ["current_timestamp"])
    # yX_test = yX_test.sort_index()

    yX_test["Bid #"] = (
        yX_test
        .groupby(level=_GROUPBY_KEY_FEATURES[:-1], observed=True)["id"]
        .transform(lambda s: pd.factorize(s)[0] + 1)
        .astype(int)
        .apply(lambda n: f"Bid {n}")
    )

    return X_train, X_test, y_train, y_test, yX_test


def train_and_log_model(
    data,
    feature_config,
    args,
):
    cat_features = list(feature_config["cat_features"])
    features = list(feature_config["features"])
    pre_features = feature_config["pre_features"]

    args_dict = vars(args).copy()
    testing = args_dict.pop("testing")
    experiment_name = args_dict.pop("experiment_name", DEFAULT_EXP_NAME)
    feature_config_path = args_dict.pop("feature_config", None)

    catboost_kwargs = {
        key: value
        for key, value in args_dict.items()
        if key in CATBOOST_PARAM_NAMES and value is not None
    }

    remaining_args = {
        key: value
        for key, value in args_dict.items()
        if key not in CATBOOST_PARAM_NAMES and value is not None
    }
    if feature_config_path is not None:
        remaining_args["feature_config"] = feature_config_path
    remaining_args["testing"] = testing
    remaining_args["experiment_name"] = experiment_name

    X_train, X_test, y_train, y_test, bid_prob_test_results = prepare_features(
        data, pre_features, testing
    )

    mlflow.set_experiment(experiment_name)
    run_name = f"catboost_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(
        run_name=run_name,
        nested=(mlflow.active_run() is not None),
        log_system_metrics=True,
    ) as run:
        log_run_parameters(
            {
                **{k: v for k, v in catboost_kwargs.items() if v is not None},
                **remaining_args,
            },
            features,
            cat_features,
            len(X_train),
            len(X_test),
        )
        log_feature_config_artifacts(feature_config)

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

        bid_prob_test_results["Acceptance Probability"] = proba

        log_classification_metrics(y_test, y_pred)
        log_classification_metrics_by_time(bid_prob_test_results)
        log_prob_examples(bid_prob_test_results)
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

    train_and_log_model(
        data,
        feature_config,
        args,
    )


if __name__ == "__main__":
    sklearn.set_config(enable_metadata_routing=True)
    main()
