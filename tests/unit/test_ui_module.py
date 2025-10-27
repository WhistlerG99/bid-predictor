from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd
import pytest

from bid_predictor.ui import data_loading, filtering, labeling, prediction, records


class DummyModel:
    def __init__(self, predictions: Iterable[Iterable[float]], schema: List[str] | None = None):
        self._predictions = list(predictions)
        self.metadata = None
        if schema is not None:
            self.metadata = DummyMetadata(schema)

    def predict_proba(self, df: pd.DataFrame):
        return pd.DataFrame(self._predictions, index=df.index, columns=["neg", "Acceptance Probability"])


class DummyMetadata:
    def __init__(self, columns: List[str]):
        self._columns = columns

    def get_input_schema(self):
        return DummySchema(self._columns)


class DummySchema:
    def __init__(self, columns: List[str]):
        self._columns = columns

    def input_names(self):
        return list(self._columns)


def test_load_dataset_cached_normalizes_datetime(monkeypatch):
    sample = pd.DataFrame(
        {
            "carrier_code": ["AC"],
            "flight_number": [1],
            "travel_date": ["2024-01-01"],
            "upgrade_type": ["J"],
            "snapshot_num": [1],
            "current_timestamp": ["2024-01-01T12:00:00"],
            "departure_timestamp": ["2024-01-02T12:00:00"],
        }
    )
    call_counter = {"count": 0}

    def fake_loader(path: str) -> pd.DataFrame:
        call_counter["count"] += 1
        return sample.copy()

    monkeypatch.setattr(data_loading, "load_training_data", fake_loader)
    data_loading._load_dataset_cached.cache_clear()

    result = data_loading._load_dataset_cached("dummy")
    assert pd.api.types.is_datetime64_any_dtype(result["travel_date"])
    assert pd.api.types.is_datetime64_any_dtype(result["current_timestamp"])
    assert call_counter["count"] == 1

    # Second call uses cache
    _ = data_loading._load_dataset_cached("dummy")
    assert call_counter["count"] == 1


@pytest.mark.parametrize(
    "value,expected",
    [("5", 5), (5.6, 5), (-3, 0), (None, None), ("bad", None)],
)
def test_normalize_threshold(value: Any, expected: int | None):
    assert filtering._normalize_threshold(value) == expected


def test_filter_dataset_by_combo_counts_filters_rows():
    df = pd.DataFrame(
        {
            "carrier_code": ["AC", "AC", "AC", "WS"],
            "flight_number": [1, 1, 1, 2],
            "travel_date": pd.to_datetime(["2024-01-01"] * 4),
            "upgrade_type": ["J", "J", "J", "J"],
            "snapshot_num": [1, 2, 3, 1],
            "bid_id": ["a", "b", "c", "d"],
        }
    )
    filtered = filtering._filter_dataset_by_combo_counts(df, min_unique_bids=3, min_snapshot_count=3)
    assert len(filtered) == 3
    assert set(filtered["bid_id"]) == {"a", "b", "c"}


def test_safe_float_and_prepare_bid_record():
    record = {
        "offer_time": "1.23456",
        "usd_base_amount": "10.239",
        "Acceptance Probability": 50,
    }
    prepared = records._prepare_bid_record(record)
    assert "Acceptance Probability" not in prepared
    assert prepared["offer_time"] == pytest.approx(1.2346)
    assert prepared["usd_base_amount"] == pytest.approx(10.24)


def test_recompute_usd_metrics_assigns_peer_quantiles():
    records_list: List[Dict[str, Any]] = [
        {"usd_base_amount": "10"},
        {"usd_base_amount": "20"},
        {"usd_base_amount": "30"},
    ]
    records._recompute_usd_metrics(records_list)
    first = records_list[0]
    assert first["usd_base_amount"] == pytest.approx(10.0)
    assert first["usd_base_amount_50%"] == pytest.approx(25.0)
    assert first["usd_base_amount_max"] == pytest.approx(30.0)


def test_labeling_builds_and_applies_labels():
    df = pd.DataFrame({"bid_id": ["b", "a", "c"]})
    label_map, column = labeling._compute_bid_label_map(df)
    assert column == "bid_id"
    applied = labeling._apply_bid_labels(df, label_map, column)
    assert list(applied["Bid #"]) == [2, 1, 3]


def test_sort_records_and_next_label():
    records_list = [
        {"Bid #": "3"},
        {"Bid #": "1"},
        {"Bid #": None},
    ]
    sorted_records = labeling._sort_records_by_bid(records_list)
    assert [item["Bid #"] for item in sorted_records[:2]] == ["1", "3"]
    assert labeling._get_next_bid_label(records_list) == 4


def test_prepare_prediction_dataframe_converts_columns(monkeypatch):
    sample_records = [
        {"carrier_code": "AC", "feature_a": "1", "travel_date": "2024-01-01"},
    ]
    monkeypatch.setattr(prediction, "_get_feature_columns", lambda: (["feature_a"], []))
    df = prediction._prepare_prediction_dataframe(sample_records)
    assert pd.api.types.is_numeric_dtype(df["feature_a"])
    assert "carrier_code" in df.columns


def test_predict_adds_acceptance_probability(monkeypatch):
    data = pd.DataFrame({"feature_a": [1.0], "extra": [5]})

    def fake_get_features():
        return ["feature_a"], []

    monkeypatch.setattr(prediction, "_get_feature_columns", fake_get_features)
    monkeypatch.setattr(prediction, "_load_model_cached", lambda uri: DummyModel([[0.2, 0.8]]))

    result = prediction._predict("uri", data)
    assert "Acceptance Probability" in result.columns
    assert result["Acceptance Probability"].iloc[0] == pytest.approx(80.0)


def test_predict_aligns_with_model_schema(monkeypatch):
    data = pd.DataFrame({"feature_a": [1.0]})

    dummy_model = DummyModel([[0.3, 0.7]], schema=["feature_a", "feature_b"])
    monkeypatch.setattr(prediction, "_load_model_cached", lambda uri: dummy_model)

    result = prediction._predict("uri", data)
    assert "Acceptance Probability" in result.columns
    assert result.attrs["model_warning"].startswith("Added missing model columns")
