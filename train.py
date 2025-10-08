import argparse
import pandas as pd
import mlflow
from catboost import Pool
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
from bid_predictor.bid_predictor import (
    build_pipeline,
    pre_features,
    cat_features,
    features,
    start_catboost_mlflow_stream,
)
import os
arn = "arn:aws:sagemaker:us-east-1:622055002283:mlflow-tracking-server/tracking-server-5vt5uv9jpcqmxz-c3e5i8l64ccsh3-dev"
mlflow.set_tracking_uri(arn)
# os.environ["MLFLOW_TRACKING_URI"] = arn


def parse_args():
    p = argparse.ArgumentParser()
    # CatBoost knobs
    p.add_argument("--task_type", type=str, default="CPU")   # "GPU" to use GPU
    p.add_argument("--devices", type=str, default="0")       # "0", "0,1", etc.
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--l2_leaf_reg", type=float, default=3.0)
    # your own toggles
    p.add_argument("--eval_metric", type=str, default="AUC")
    p.add_argument("--seed", type=int, default=42)
    # bool flags — use explicit parsing
    p.add_argument("--use_border_count", type=int, default=1) # 1/0 from SageMaker
    return p.parse_args()


def prepare_features(out):
    out.sort_values(["travel_date", "carrier_code", "flight_number"]).reset_index(
        drop=True
    )

    out = out.fillna(
        {
            "multiplier_fare_class": 1.0,
            "multiplier_loyalty": 1.0,
            "multiplier_success_history": 1.0,
            "multiplier_payment_type": 1.0,
        }
    )

    cutoff = "2025-05-01"
    # cutoff = "2023-08-01"
    yX_train = out[out.travel_date < cutoff][pre_features + ["offer_status"]]
    yX_test = out[out.travel_date >= cutoff][pre_features + ["offer_status"]]
    # yX_test = out[(out.travel_date >= cutoff) & (out.travel_date <= "2023-08-15")][
        # pre_features + ["offer_status"]
    # ]

    X_train, X_test = yX_train[pre_features], yX_test[pre_features]
    y_train = (yX_train["offer_status"] == "TICKETED").astype(int)
    y_test = (yX_test["offer_status"] == "TICKETED").astype(int)
    return X_train, X_test, y_train, y_test


def train_and_log_model(X_train, X_test, y_train, y_test, cat_features, features):
    args = parse_args()

    mlflow.set_experiment("catboost-bid-predictor")
    run_name = f"catboost_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=run_name, nested=(mlflow.active_run() is not None), log_system_metrics=True) as run:
        eval_metric = "AUC"
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("categorical_features", ",".join(cat_features))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("eval_metric", eval_metric)
        mlflow.log_params(vars(args))

        pipeline = build_pipeline(
            task_type=args.task_type,
            devices=args.devices,
        )
        # cat = pipeline[-1]  # CatBoostClassifier
        # train_dir = cat._cb.get_param("train_dir")
        train_dir = os.environ.get("CATBOOST_TRAIN_DIR", "/opt/ml/output/catboost")
        # os.makedirs(train_dir, exist_ok=True)

        # start streaming before fit
        stop_stream = start_catboost_mlflow_stream(train_dir, run.info.run_id)

        try:
            pipeline.fit(X_train, y_train, eval_set=(X_test, y_test))
        finally:
            stop_stream()
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
        train_pool = Pool(X_train_trns, y_train, cat_features=cat_features)
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
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    train_file = train_dir + "/bid_data_enriched_new_reduced.csv"

    data = pd.read_csv(
        train_file,
        parse_dates=["travel_date"],
        dtype={
            "carrier_code": "category",
            "flight_number": "category",
            "fare_class": "category",
        },
        low_memory=False,
    )

    X_train, X_test, y_train, y_test = prepare_features(data)
    train_and_log_model(X_train, X_test, y_train, y_test, cat_features, features)


if __name__ == "__main__":
    sklearn.set_config(enable_metadata_routing=True)
    main()
