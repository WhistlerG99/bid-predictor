import os
import argparse
import pandas as pd
import mlflow
from catboost import Pool
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from bid_predictor.bid_predictor import build_pipeline, load_feature_config
from bid_predictor.tracking import start_catboost_mlflow_stream
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

DEFAULT_EXP_NAME="tests"

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
    return p.parse_args()


def prepare_features(data, pre_features):
    data = data[
        data.departure_timestamp - data.current_timestamp < pd.to_timedelta("5d")
    ]

    data.sort_values(["travel_date", "carrier_code", "flight_number"]).reset_index(
        drop=True
    )

    # data = data.fillna(
    #     {
    #         "multiplier_fare_class": 1.0,
    #         "multiplier_loyalty": 1.0,
    #         "multiplier_success_history": 1.0,
    #         "multiplier_payment_type": 1.0,
    #     }
    # )

    available_pre_features = [feature for feature in pre_features if feature in data.columns]
    selection_columns = list(dict.fromkeys(available_pre_features + ["offer_status"]))

    testing = False
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

    experiment_name = var_args.pop("experiment_name",DEFAULT_EXP_NAME)
    mlflow.set_experiment(experiment_name)
    run_name = f"catboost_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(
        run_name=run_name,
        nested=(mlflow.active_run() is not None),
        log_system_metrics=True,
    ) as run:
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("categorical_features", ",".join(cat_features))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_params(var_args)

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
        mlflow.log_metrics(
            {
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
            }
        )

        X_train_trns = pipeline[:-1].transform(X_train)
        active_cat_features = [
            feature for feature in cat_features if feature in X_train_trns.columns
        ]
        train_pool = Pool(X_train_trns, y_train, cat_features=active_cat_features)
        fi_vals = pipeline[-1].get_feature_importance(train_pool)
        fi = pd.Series(fi_vals, index=X_train_trns.columns).sort_values()
        fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
        fi.plot.barh(ax=ax_fi)
        ax_fi.set_title("CatBoost Feature Importance")
        ax_fi.set_xlabel("Importance")
        ax_fi.grid(zorder=0)
        ax_fi.set_axisbelow(True)
        fig_fi.tight_layout()
        mlflow.log_figure(fig_fi, "feature_importance.png")
        plt.close(fig_fi)

        cm = confusion_matrix(y_test, y_pred)
        cmn = confusion_matrix(y_test, y_pred, normalize="true")

        fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        fig_cm.tight_layout()
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        fig_cmn, ax_cmn = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cmn).plot(ax=ax_cmn)
        ax_cmn.set_title("Confusion Matrix (Normalized)")
        fig_cmn.tight_layout()
        mlflow.log_figure(fig_cmn, "confusion_matrix_normalized.png")
        plt.close(fig_cmn)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(
            y_test, proba, ax=ax_roc, drop_intermediate=True
        )
        ax_roc.set_title("ROC Curve")
        fig_roc.tight_layout()
        mlflow.log_figure(fig_roc, "roc_curve.png")
        plt.close(fig_roc)

        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax_pr)
        ax_pr.set_title("Precision-Recall Curve")
        fig_pr.tight_layout()
        mlflow.log_figure(fig_pr, "precision_recall_curve.png")
        plt.close(fig_pr)

        fig_ap, ax_ap = plt.subplots(1, 2, figsize=(12, 5))
        ax_ap[0].hist(proba[y_test == 1], alpha=0.4, label="Ticketed", bins=100)
        ax_ap[0].hist(proba[y_test == 0], alpha=0.4, label="Expired", bins=100)
        ax_ap[0].grid(zorder=0)
        ax_ap[0].set_axisbelow(True)
        ax_ap[0].legend(loc="best")
        ax_ap[0].set_xlabel("Acceptence Probability")
        ax_ap[0].set_title("Linear Scale")
        ax_ap[1].hist(proba[y_test == 1], alpha=0.4, label="Ticketed", bins=100)
        ax_ap[1].hist(proba[y_test == 0], alpha=0.4, label="Expired", bins=100)
        ax_ap[1].set_yscale("log")
        ax_ap[1].grid(zorder=0)
        ax_ap[1].set_axisbelow(True)
        ax_ap[1].legend(loc="best")
        ax_ap[1].set_xlabel("Acceptence Probability")
        ax_ap[1].set_title("Log Scale")
        fig_ap.tight_layout()
        mlflow.log_figure(fig_ap, "acceptance_probability.png")
        plt.close(fig_ap)

        mlflow.sklearn.log_model(pipeline, "pipeline")


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
        train_file = "../bid_data_snapshots_v2.parquet"

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

    X_train, X_test, y_train, y_test = prepare_features(data, pre_features)
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
