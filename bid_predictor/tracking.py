import os
import json
import hashlib
import re
import time
import threading
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mlflow
import pandas as pd
import numpy as np
from catboost import Pool
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from .feature_config import _GROUPBY_KEY_FEATURES


class MlflowCallback(object):
    """CatBoost callback to log training and validation metrics to MLflow."""

    def after_iteration(self, info):
        iteration = info.iteration

        train_loss = info.metrics["learn"]["Logloss"][-1]
        mlflow.log_metric("train_loss", train_loss, step=iteration)

        validation_loss = info.metrics["validation"]["Logloss"][-1]
        mlflow.log_metric("validation_loss", validation_loss, step=iteration)

        validation_auc = info.metrics["validation"]["AUC"][-1]
        mlflow.log_metric("validation_auc", validation_auc, step=iteration)

        return True


def _format_bins(bins):
    """Normalize binning metadata into a compact, serializable dictionary."""
    if not bins:
        return None
    if "interval" in bins:
        return {
            "type": "interval",
            "min": bins.get("min"),
            "max": bins.get("max"),
            "interval": bins.get("interval"),
        }
    if "nsteps" in bins:
        return {
            "type": "nsteps",
            "min": bins.get("min"),
            "max": bins.get("max"),
            "nsteps": bins.get("nsteps"),
        }
    return bins


def _format_outlier(outlier):
    """Extract relevant outlier capping thresholds from raw metadata."""
    if not outlier:
        return None
    return {key: outlier[key] for key in ("min", "max") if key in outlier}


def _sanitize_feature_name(name):
    """Convert arbitrary feature names into MLflow-safe parameter keys."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def summarize_feature_transformations(feature_config: Mapping[str, Mapping]):
    """Summarize model features and their transformations for MLflow logging."""

    metadata = feature_config.get("feature_metadata", {})
    summary = []
    for feature_name in sorted(metadata):
        values = metadata[feature_name]
        if not values.get("include_in_model", False):
            continue

        transformations = []
        impute_value = values.get("impute_value")
        impute_median = values.get("impute_median", False)

        if impute_value is not None:
            transformations.append({"type": "impute_value", "value": impute_value})
            transformations.append({"type": "missing_indicator"})
        elif impute_median:
            transformations.append({"type": "impute_median"})
            transformations.append({"type": "missing_indicator"})

        outlier = _format_outlier(values.get("outlier"))
        if outlier:
            transformations.append({"type": "outlier_capper", "params": outlier})

        bins = _format_bins(values.get("bins"))
        if bins:
            transformations.append({"type": "discretiser", "params": bins})

        if values.get("categorical", False):
            transformations.append({"type": "catboost_categorical"})

        if not transformations:
            transformations.append({"type": "passthrough"})

        summary.append(
            {
                "feature": feature_name,
                "derived": bool(values.get("derived", False)),
                "categorical": bool(values.get("categorical", False)),
                "impute_value": impute_value,
                "impute_median": bool(impute_median),
                "outlier": outlier,
                "bins": bins,
                "transformations": transformations,
            }
        )

    return summary


def feature_config_fingerprint(feature_summary: Sequence[Mapping]):
    """Create a stable fingerprint for a feature configuration summary."""

    encoded = json.dumps(feature_summary, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def feature_summary_markdown(feature_summary: Sequence[Mapping]):
    """Render a markdown table describing the feature summary."""

    if not feature_summary:
        return "No features are configured for this run."

    header = [
        "| Feature | Derived | Categorical | Transformations |",
        "| --- | --- | --- | --- |",
    ]
    rows = []
    for item in feature_summary:
        transforms = []
        for transform in item["transformations"]:
            if transform["type"] in {"impute_value", "outlier_capper", "discretiser"}:
                detail = transform.copy()
                detail_type = detail.pop("type")
                transforms.append(
                    f"**{detail_type}**: {json.dumps(detail, sort_keys=True)}"
                )
            elif transform["type"] == "impute_median":
                transforms.append("**impute_median**")
            elif transform["type"] == "missing_indicator":
                transforms.append("missing_indicator")
            elif transform["type"] == "catboost_categorical":
                transforms.append("catboost_categorical")
            else:
                transforms.append(transform["type"])
        rows.append(
            "| {feature} | {derived} | {categorical} | {transforms} |".format(
                feature=item["feature"],
                derived="✅" if item["derived"] else "",
                categorical="✅" if item["categorical"] else "",
                transforms="<br>".join(transforms),
            )
        )

    return "\n".join(header + rows)


def feature_parameters_for_mlflow(feature_summary: Sequence[Mapping]):
    """Flatten feature summary for MLflow parameter logging."""

    params = {}
    for item in feature_summary:
        feature_name = item["feature"]
        prefix = f"ft_{_sanitize_feature_name(feature_name)}"

        transforms = []
        for transform in item["transformations"]:
            base_descriptor = transform["type"]
            payload = {
                key: value
                for key, value in transform.items()
                if key != "type" and value is not None
            }
            if payload:
                payload_json = json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                )
                transforms.append(f"{base_descriptor}:{payload_json}")
            else:
                transforms.append(base_descriptor)

        if transforms:
            params[f"{prefix}_transformations"] = " | ".join(transforms)

        if item.get("impute_value") is not None:
            params[f"{prefix}_impute_value"] = str(item["impute_value"])

        if item.get("impute_median"):
            params[f"{prefix}_impute_median"] = "true"

        outlier = item.get("outlier")
        if outlier:
            for k, v in outlier.items():
                params[f"{prefix}_outlier_{k}"] = v

        bins = item.get("bins")
        if bins:
            for k, v in bins.items():
                params[f"{prefix}_bins_{k}"] = v

    return params


def feature_importance_metrics(importances: Mapping[str, float], prefix: str = "fi"):
    """Prepare MLflow metric names for feature importances."""

    metrics = {}
    for feature_name, value in importances.items():
        if value is None:
            continue
        metric_name = f"{prefix}_{_sanitize_feature_name(str(feature_name))}"
        metrics[metric_name] = float(value)

    return metrics


def log_run_parameters(
    args_dict: Mapping[str, object],
    features: Iterable[str],
    categorical_features: Iterable[str],
    train_size: int,
    test_size: int,
):
    """Log basic run configuration parameters to MLflow."""

    feature_list = list(features)
    cat_list = list(categorical_features)

    base_params = {
        "n_features": len(feature_list),
        "features": ",".join(feature_list),
        "categorical_features": ",".join(cat_list),
        "train_rows": int(train_size),
        "test_rows": int(test_size),
    }

    mlflow.log_params(base_params)
    if args_dict:
        mlflow.log_params(dict(args_dict))


def log_feature_config_artifacts(feature_config: Mapping[str, Mapping]):
    """Log feature configuration metadata, tags, and summaries to MLflow."""

    feature_summary = summarize_feature_transformations(feature_config)
    fingerprint = feature_config_fingerprint(feature_summary)
    transformation_index = defaultdict(list)
    for item in feature_summary:
        for transform in item["transformations"]:
            transformation_index[transform["type"]].append(item["feature"])

    transformation_counts = {
        name: len(sorted(set(feature_names)))
        for name, feature_names in transformation_index.items()
    }

    mlflow.set_tags(
        {
            "feature_fingerprint": fingerprint,
            "feature_count": len(feature_summary),
            "feature_transformations": ";".join(
                f"{name}:{count}"
                for name, count in sorted(transformation_counts.items())
            ),
        }
    )

    mlflow.log_dict(
        {
            "features": feature_summary,
            "transformation_counts": transformation_counts,
            "transformation_index": {
                name: sorted(set(feature_names))
                for name, feature_names in transformation_index.items()
            },
        },
        "feature_pipeline_summary.json",
    )
    mlflow.log_text(
        feature_summary_markdown(feature_summary),
        "feature_pipeline_summary.md",
    )

    mlflow.log_params(feature_parameters_for_mlflow(feature_summary))
    return feature_summary


def log_classification_metrics(y_true, y_prob):
    """Log basic classification metrics to MLflow."""

    metrics = {}
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        metrics.update(
            {
                f"accuracy_{int(100*t)}": float(accuracy_score(y_true, (y_prob >= t).astype(int))),
                f"precision_{int(100*t)}": float(precision_score(y_true, (y_prob >= t).astype(int), zero_division=0)),
                f"recall_{int(100*t)}": float(recall_score(y_true, (y_prob >= t).astype(int), zero_division=0)),
                f"negative_precision_{int(100*t)}": float(precision_score((y_true==0).astype(int), (y_prob < t).astype(int), zero_division=0)),
                f"negative_recall_{int(100*t)}": float(recall_score((y_true==0).astype(int), (y_prob < t).astype(int), zero_division=0)),
            }
        )
    mlflow.log_metrics(metrics)


def _log_confusion_matrices(y_true, y_pred):
    """Render and log raw and normalized confusion matrices to MLflow."""
    cm = confusion_matrix(y_true, y_pred)
    cmn = confusion_matrix(y_true, y_pred, normalize="true")

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


def _log_curves(y_true, proba):
    """Log ROC and precision-recall diagnostic curves for the run."""
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax_roc, drop_intermediate=True)
    ax_roc.set_title("ROC Curve")
    fig_roc.tight_layout()
    mlflow.log_figure(fig_roc, "roc_curve.png")
    plt.close(fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax_pr)
    ax_pr.set_title("Precision-Recall Curve")
    fig_pr.tight_layout()
    mlflow.log_figure(fig_pr, "precision_recall_curve.png")
    plt.close(fig_pr)


def _log_acceptance_probability_histograms(y_true, proba):
    """Log histograms comparing ticketed versus expired acceptance scores."""
    fig_ap, ax_ap = plt.subplots(1, 2, figsize=(12, 5))
    ax_ap[0].hist(proba[y_true == 1], alpha=0.4, label="Ticketed", bins=100)
    ax_ap[0].hist(proba[y_true == 0], alpha=0.4, label="Expired", bins=100)
    ax_ap[0].grid(zorder=0)
    ax_ap[0].set_axisbelow(True)
    ax_ap[0].legend(loc="best")
    ax_ap[0].set_xlabel("Acceptance Probability")
    ax_ap[0].set_title("Linear Scale")

    ax_ap[1].hist(proba[y_true == 1], alpha=0.4, label="Ticketed", bins=100)
    ax_ap[1].hist(proba[y_true == 0], alpha=0.4, label="Expired", bins=100)
    ax_ap[1].set_yscale("log")
    ax_ap[1].grid(zorder=0)
    ax_ap[1].set_axisbelow(True)
    ax_ap[1].legend(loc="best")
    ax_ap[1].set_xlabel("Acceptance Probability")
    ax_ap[1].set_title("Log Scale")

    fig_ap.tight_layout()
    mlflow.log_figure(fig_ap, "acceptance_probability.png")
    plt.close(fig_ap)


def log_evaluation_figures(y_true, y_pred, proba):
    """Generate and log evaluation figures to MLflow."""

    _log_confusion_matrices(y_true, y_pred)
    _log_curves(y_true, proba)
    _log_acceptance_probability_histograms(y_true, proba)


def log_feature_importances(
    pipeline, X_train, y_train, categorical_features: Iterable[str]
):
    """Log feature importances and associated plots to MLflow."""

    X_train_transformed = pipeline[:-1].transform(X_train)
    active_cat_features = [
        feature
        for feature in categorical_features
        if feature in X_train_transformed.columns
    ]
    train_pool = Pool(X_train_transformed, y_train, cat_features=active_cat_features)
    fi_vals = pipeline[-1].get_feature_importance(train_pool)
    fi = pd.Series(fi_vals, index=X_train_transformed.columns).sort_values()

    mlflow.log_metrics(feature_importance_metrics(fi))

    fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
    fi.plot.barh(ax=ax_fi)
    ax_fi.set_title("CatBoost Feature Importance")
    ax_fi.set_xlabel("Importance")
    ax_fi.grid(zorder=0)
    ax_fi.set_axisbelow(True)
    fig_fi.tight_layout()
    mlflow.log_figure(fig_fi, "feature_importance.png")
    plt.close(fig_fi)


def log_classification_metrics_by_time(
    df,
    window_h=6,
    stride_h=1,
    suffix="departure",
    figsize=(12, 10),
    zero_division=np.nan,
    fontsize=8,
    title_fontsize=10,
    figure_path=None,
):
    
    # sliding window metrics on test set
    # adjustable stride (hours) for sliding windows
    if not isinstance(stride_h, int) or stride_h <= 0:
        raise ValueError("stride_h must be a positive integer")

    df = df.copy()
    df["pred"] = (df["Acceptance Probability"] > 0.5).astype(int)
    df["actual"] = (df["offer_status"] == "Accepted").astype(int)
    df[f"hours_before_{suffix}"] = (
        pd.to_datetime(df[f"{suffix}_timestamp"])
        - pd.to_datetime(df["current_timestamp"])
    ).dt.total_seconds() / 3600.0

    start_h = int(np.floor(df[f"hours_before_{suffix}"].min()))
    end_h = int(np.ceil(df[f"hours_before_{suffix}"].max()))

    # ensure at least one window
    last_start = end_h - window_h
    starts = range(start_h, last_start + 1) if last_start >= start_h else [start_h]

    rows = []
    # rebuild starts using the stride
    starts = (
        range(start_h, last_start + 1, stride_h) if last_start >= start_h else [start_h]
    )
    for s in starts:
        e = s + window_h
        sub = df[
            (df[f"hours_before_{suffix}"] >= s) & (df[f"hours_before_{suffix}"] < e)
        ]
        if sub.empty:
            rows.append(
                {
                    "start_hour": s,
                    "end_hour": e,
                    "n": 0,
                    "n_tp": 0,
                    "n_tn": 0,
                    "n_fp": 0,
                    "n_fn": 0,
                    "accuracy": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "neg_recall": np.nan,
                    "neg_precision": np.nan,
                }
            )
            continue
        y_true = sub["actual"]
        y_pred = sub["pred"]
        rows.append(
            {
                "start_hour": s,
                "end_hour": e,
                "n": len(sub),
                "n_tp": ((y_true == 1) * (y_pred == 1)).sum(),
                "n_tn": ((y_true == 0) * (y_pred == 0)).sum(),
                "n_fp": ((y_true == 0) * (y_pred == 1)).sum(),
                "n_fn": ((y_true == 1) * (y_pred == 0)).sum(),
                "accuracy": accuracy_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred, zero_division=zero_division),
                "precision": precision_score(y_true, y_pred, zero_division=zero_division),
                "neg_recall": recall_score(y_true == 0, y_pred == 0, zero_division=zero_division),
                "neg_precision": precision_score(y_true == 0, y_pred == 0, zero_division=zero_division),
            }
        )

    metrics = pd.DataFrame(rows)

    metrics["n_ap"] = metrics.n_tp + metrics.n_fn
    metrics["n_an"] = metrics.n_tn + metrics.n_fp

    metrics["n_pp"] = metrics.n_tp + metrics.n_fp
    metrics["n_pn"] = metrics.n_tn + metrics.n_fn

    metrics[f"hour_before_{suffix}"] = (metrics.start_hour + metrics.end_hour) / 2
    metrics = metrics.set_index(f"hour_before_{suffix}")

    # Define the formatter function
    def thousands_formatter(value, tick_number):
        if value >= 1000:
            return f'{int(value / 1000)}K'
        return str(int(value))

    fig_m, ax_m = plt.subplots(3, 2, figsize=figsize)

    fig_m.suptitle(f"Classification Metrics vs. Time until {suffix.capitalize()}\n (All quantities are calculated over a {window_h} hour window every {stride_h} hour(s))")

    # Top-left: count
    metrics.n_ap.plot(ax=ax_m[0][0], label="Actual Positives")
    metrics.n_an.plot(ax=ax_m[0][0], label="Actual Negatives")
    ax_m[0][0].yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax_m[0][0].set_xlabel(f"Hours Until {suffix.capitalize()}", fontsize=fontsize)
    ax_m[0][0].set_title("Number of Active Bids", fontsize=title_fontsize)
    ax_m[0][0].grid(zorder=0)
    ax_m[0][0].set_axisbelow(True)
    ax_m[0][0].set_ylim(bottom=0)
    ax_m[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)

    # Top-right: accuracy
    metrics.accuracy.plot(ax=ax_m[0][1])
    ax2 = metrics.n.plot(ax=ax_m[0][1], secondary_y=True, label="Active Bids")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax_m[0][1].set_xlabel(f"Hours Until {suffix.capitalize()}", fontsize=fontsize)
    ax_m[0][1].set_title("Accuracy", fontsize=title_fontsize)
    ax_m[0][1].grid(zorder=0)
    ax_m[0][1].set_axisbelow(True)
    ax_m[0][1].set_ylim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)

    # Middle-left: recall + n_ap on secondary y
    metrics.recall.plot(ax=ax_m[1][0], label="Recall")
    ax2 = metrics.n_ap.plot(ax=ax_m[1][0], secondary_y=True, label="Actual Positives")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax_m[1][0].set_xlabel(f"Hours Until {suffix.capitalize()}", fontsize=fontsize)
    ax_m[1][0].set_title("Recall", fontsize=title_fontsize)
    ax_m[1][0].grid(zorder=0)
    ax_m[1][0].set_axisbelow(True)
    ax_m[1][0].set_ylim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)


    # Middle-right: precision + n_pp on secondary y
    metrics.precision.plot(ax=ax_m[1][1])
    ax2 = metrics.n_pp.plot(ax=ax_m[1][1], secondary_y=True, label="Predicted Positives")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax_m[1][1].set_xlabel(f"Hours Until {suffix.capitalize()}", fontsize=fontsize)
    ax_m[1][1].set_title("Precision", fontsize=title_fontsize)
    ax_m[1][1].grid(zorder=0)
    ax_m[1][1].set_axisbelow(True)
    ax_m[1][1].set_ylim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)

    # Bottom-left: negative recall + n_an on secondary y
    metrics.neg_recall.plot(ax=ax_m[2][0])
    ax2 = metrics.n_an.plot(ax=ax_m[2][0], secondary_y=True, label="Actual Negatives")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax_m[2][0].set_xlabel(f"Hours Until {suffix.capitalize()}", fontsize=fontsize)
    ax_m[2][0].set_title("Negative Recall", fontsize=title_fontsize)
    ax_m[2][0].grid(zorder=0)
    ax_m[2][0].set_axisbelow(True)
    ax_m[2][0].set_ylim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)

    # Bottom-right: negative precision + n_pn on secondary y
    metrics.neg_precision.plot(ax=ax_m[2][1])
    ax2 = metrics.n_pn.plot(ax=ax_m[2][1], secondary_y=True, label="Predicted Negatives")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax_m[2][1].set_xlabel(f"Hours Until {suffix.capitalize()}", fontsize=fontsize)
    ax_m[2][1].set_title("Negative Precision", fontsize=title_fontsize)
    ax_m[2][1].grid(zorder=0)
    ax_m[2][1].set_axisbelow(True)
    ax_m[2][1].set_ylim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)

    # Apply fontsize to tick labels and any secondary y-axes created by pandas plotting
    for ax_row in ax_m:
        for ax in ax_row:
            ax.tick_params(axis="both", labelsize=fontsize)
            # apply to secondary y-axis if pandas created one
            if hasattr(ax, "right_ax"):
                try:
                    ax.right_ax.tick_params(axis="y", labelsize=fontsize)
                except Exception:
                    pass
            # also apply to potential twin axes created differently
            for twin in ax.figure.axes:
                if twin is ax:
                    continue
                # if twin shares x or y with ax, adjust its tick labels size
                if twin.get_shared_x_axes().joined(ax, twin) or twin.get_shared_y_axes().joined(ax, twin):
                    twin.tick_params(axis="both", labelsize=fontsize)

    fig_m.tight_layout()

    if figure_path is None:
        figure_path = f"classification_metrics_vs_time_utill_{suffix}.png"
    mlflow.log_figure(fig_m, figure_path)
    plt.close(fig_m)


def log_prob_examples(df):
    bid_attr_cols = [
        "Bid #",
        "offer_status",
        "usd_base_amount",
        "item_count",
        "fare_class",
        "time_until_departure",
    ]
    df = df.sort_index()

    df["time_until_departure"] = df.departure_timestamp - df.current_timestamp

    cnts_snap = (
        df.reset_index()[_GROUPBY_KEY_FEATURES]
        .drop_duplicates()
        .groupby(_GROUPBY_KEY_FEATURES[:-1], observed=True)
        .size()
    )
    
    cand = cnts_snap.reset_index(name="n")
    cand = cand[(cand["n"] >= 5) & (cand["n"] < 10)]

    sampled = (
        cand.groupby("carrier_code", observed=True)
        .apply(
            lambda g: g.sample(n=min(5, len(g)), random_state=10), include_groups=False
        )
        .reset_index(level="carrier_code")
    )

    # create auctions list of tuples (carrier_code, flight_number, travel_date, upgrade_type)
    auctions = [
        tuple(x)
        for x in sampled[
            ["carrier_code", "flight_number", "travel_date", "upgrade_type"]
        ].values
    ]

    for auc in auctions:
        acc_prob = df.loc[auc, bid_attr_cols + ["Acceptance Probability"]]
        acc_prob = (
            acc_prob.reset_index()
            .set_index(bid_attr_cols)["Acceptance Probability"]
            .unstack()
            .T
        )
        acc_prob.columns = [" - ".join(map(str, c)) for c in acc_prob.columns]
        acc_prob.index = acc_prob.index.round("h")
        acc_prob = acc_prob.sort_index(ascending=False)
        

        fig_p, ax_p = plt.subplots(figsize=(10, 8))
        acc_prob.plot.bar(ax=ax_p, zorder=1, alpha=0.9)

        if "seats_available" in df.columns:
            seats_avail = (
                df.loc[auc, ["seats_available", "time_until_departure"]]
                .reset_index()[["time_until_departure", "seats_available"]]
                .drop_duplicates()
                .set_index("time_until_departure")
            )
            seats_avail.index = seats_avail.index.round("h")
            seats_avail = seats_avail.sort_index(ascending=False)
        
            ax_s = seats_avail.plot(ax=ax_p, secondary_y=True)
            ax_s.set_ylabel("Available Seats")
        
        # ax_p.set_ylim(0,1)
        ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2)
        # draw grid behind the bars
        ax_p.set_axisbelow(True)
        ax_p.grid()
        ax_p.set_title(auc[0] + auc[1] + ", " + str(auc[2].date()) + ", " + auc[3])
        fig_p.tight_layout()

        mlflow.log_figure(
            fig_p,
            f"bid-acceptance-prob-examples/{'-'.join((auc[0]+auc[1], str(auc[2].date()), auc[3]))}.png",
        )
        plt.close(fig_p)


def log_pipeline_model(pipeline, artifact_path: str = "pipeline"):
    """Log the trained pipeline model to MLflow."""

    mlflow.sklearn.log_model(pipeline, artifact_path)


def log_evaluation_results_parquet(
    evaluation_results: pd.DataFrame,
    artifact_name: str = "evaluation_results.parquet",
) -> None:
    """Log evaluation results as a parquet artifact in MLflow."""

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_name
        evaluation_results.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def follow_tsv(
    path: str,
    key: str,
    run_id: str,
    client: mlflow.MlflowClient,
    stop_event: threading.Event,
):
    """
    Tail a CatBoost TSV metrics file and log values to MLflow for a specific run.

    Parameters
    ----------
    path : str
        Path to the TSV file (e.g., learn_error.tsv, test_error.tsv).
    key : str
        Metric name to log (e.g., "train_Logloss", "eval_Logloss").
    run_id : str
        Active MLflow run ID to log into.
    client : mlflow.MlflowClient
        Reusable MLflow client.
    stop_event : threading.Event
        Cooperative stop signal for the tailer thread.
    """
    # Wait (a bit) for the file to appear
    t0 = time.time()
    while not os.path.exists(path) and not stop_event.is_set():
        if time.time() - t0 > 300:  # 5-minute safety
            return
        time.sleep(0.2)
    if not os.path.exists(path):
        return

    with open(path, "r") as f:
        # skip header if present
        header_line = f.readline()
        _ = header_line  # not used; discard

        while not stop_event.is_set():
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.2)
                f.seek(pos)
                continue

            parts = line.strip().split("\t")
            if not parts:
                continue

            # Typical format: iter \t <metric(s)>
            try:
                step = int(parts[0])
            except Exception:
                continue

            # Log the last numeric column
            for c in reversed(parts[1:]):
                try:
                    val = float(c)
                    client.log_metric(run_id, key, val, step=step)
                    break
                except ValueError:
                    continue


def start_catboost_mlflow_stream(train_dir: str, run_id: str, metric_prefix: str = ""):
    """
    Start background threads that tail CatBoost metric TSVs and log to MLflow.

    Returns
    -------
    stop_stream : callable
        Call to stop threads and join them.
    """
    client = mlflow.tracking.MlflowClient()
    stop_event = threading.Event()
    threads = []

    learn_path = os.path.join(train_dir, "learn_error.tsv")
    test_path = os.path.join(train_dir, "test_error.tsv")

    t1 = threading.Thread(
        target=follow_tsv,
        args=(
            learn_path,
            f"{metric_prefix}train_Logloss" if metric_prefix else "train_Logloss",
            run_id,
            client,
            stop_event,
        ),
        daemon=True,
    )
    t2 = threading.Thread(
        target=follow_tsv,
        args=(
            test_path,
            f"{metric_prefix}eval_Logloss" if metric_prefix else "eval_Logloss",
            run_id,
            client,
            stop_event,
        ),
        daemon=True,
    )
    t1.start()
    t2.start()
    threads.extend([t1, t2])

    def stop_stream():
        stop_event.set()
        for t in threads:
            t.join(timeout=2)

    return stop_stream
