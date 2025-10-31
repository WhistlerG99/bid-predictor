import numpy as np
import pandas as pd
import pytest

from bid_predictor import tracking


def make_feature_config():
    return {
        "feature_metadata": {
            "feature_num": {
                "categorical": False,
                "include_in_model": True,
                "derived": False,
                "impute_value": None,
                "impute_median": False,
                "outlier": {"min": 0},
                "bins": None,
            },
            "feature_cat": {
                "categorical": True,
                "include_in_model": True,
                "derived": False,
                "impute_value": 0,
                "impute_median": False,
                "outlier": None,
                "bins": {"interval": 1, "min": 0, "max": 10},
            },
            "skipped": {
                "categorical": False,
                "include_in_model": False,
                "derived": False,
                "impute_value": None,
                "impute_median": False,
                "outlier": None,
                "bins": None,
            },
        }
    }


def test_summarize_feature_transformations():
    summary = tracking.summarize_feature_transformations(make_feature_config())
    assert len(summary) == 2
    assert summary[0]["feature"] in {"feature_num", "feature_cat"}


def test_feature_config_fingerprint_changes_on_update():
    summary = tracking.summarize_feature_transformations(make_feature_config())
    fp1 = tracking.feature_config_fingerprint(summary)
    summary[0]["transformations"].append({"type": "extra"})
    fp2 = tracking.feature_config_fingerprint(summary)
    assert fp1 != fp2


def test_feature_summary_markdown_contains_table():
    summary = tracking.summarize_feature_transformations(make_feature_config())
    markdown = tracking.feature_summary_markdown(summary)
    assert "| Feature |" in markdown


def test_feature_parameters_for_mlflow():
    summary = tracking.summarize_feature_transformations(make_feature_config())
    params = tracking.feature_parameters_for_mlflow(summary)
    assert any(key.startswith("ft_feature_cat") for key in params)


def test_feature_importance_metrics_sanitizes_names():
    metrics = tracking.feature_importance_metrics({"feat name": 1.0})
    assert "feat_name" in list(metrics.keys())[0]


def test_log_classification_metrics_records_metrics(stub_mlflow):
    tracking.log_classification_metrics([0, 1, 1], [0, 1, 0])
    assert stub_mlflow.calls.metrics


def test_log_classification_metrics_by_time_requires_positive_stride(stub_mlflow):
    df = pd.DataFrame(
        {
            "Acceptance Probability": np.linspace(0, 1, 5),
            "offer_status": ["Accepted", "Rejected", "Accepted", "Rejected", "Accepted"],
            "departure_timestamp": pd.date_range("2023-08-01", periods=5, freq="h"),
            "decision_timestamp": pd.date_range("2023-08-01", periods=5, freq="h"),
            "current_timestamp": pd.date_range("2023-08-01", periods=5, freq="h"),
            "carrier_code": ["AC"] * 5,
            "flight_number": ["1"] * 5,
            "travel_date": pd.date_range("2023-08-02", periods=5, freq="D"),
            "upgrade_type": ["BUS"] * 5,
            "snapshot_num": [0] * 5,
            "id": range(5),
        }
    )
    with pytest.raises(ValueError):
        tracking.log_classification_metrics_by_time(df, stride_h=0)

    tracking.log_classification_metrics_by_time(df, stride_h=1)
