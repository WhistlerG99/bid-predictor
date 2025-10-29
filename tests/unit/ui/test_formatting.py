import pandas as pd

from bid_predictor.ui import (
    BID_IDENTIFIER_COLUMNS,
    DEFAULT_UI_FEATURE_CONFIG,
    USD_MAX_COLUMN,
    USD_PERCENT_COLUMNS,
    apply_bid_labels,
    compute_bid_label_map,
    get_next_bid_label,
    prepare_bid_record,
    recompute_usd_metrics,
    safe_float,
    sort_records_by_bid,
)


def test_prepare_bid_record_strips_probability():
    record = {
        "Acceptance Probability": 42.0,
        "offer_time": "1.23456",
        "usd_base_amount": "100.556",
    }
    prepared = prepare_bid_record(record)
    assert "Acceptance Probability" not in prepared
    assert prepared["offer_time"] == 1.2346
    assert prepared["usd_base_amount"] == 100.56


def test_recompute_usd_metrics_updates_percentiles():
    records = [
        {"Bid #": 1, "usd_base_amount": 100},
        {"Bid #": 2, "usd_base_amount": 200},
        {"Bid #": 3, "usd_base_amount": 300},
    ]
    recompute_usd_metrics(records)
    for column in USD_PERCENT_COLUMNS:
        assert all(column in record for record in records)
    assert all(record[USD_MAX_COLUMN] == 300 for record in records)


def test_compute_and_apply_bid_labels():
    df = pd.DataFrame({"bid_id": ["b", "a", "b"]})
    label_map, column = compute_bid_label_map(df)
    assert column in BID_IDENTIFIER_COLUMNS
    assert label_map["a"] == 1
    assert label_map["b"] == 2

    labelled = apply_bid_labels(df, label_map, column)
    assert list(labelled["Bid #"]) == [2, 1, 2]


def test_sort_and_next_bid_label():
    records = [
        {"Bid #": "3"},
        {"Bid #": 2},
        {"Bid #": ""},
    ]
    sorted_records = sort_records_by_bid(records)
    assert [record.get("Bid #") for record in sorted_records] == [2, "3", ""]
    assert get_next_bid_label(sorted_records) == 4


def test_safe_float_handles_invalid():
    assert safe_float("12.5") == 12.5
    assert safe_float("not-a-number") is None
    assert safe_float(None) is None


def test_default_display_features_include_probability():
    display_features = DEFAULT_UI_FEATURE_CONFIG.get("display_features", [])
    assert "Acceptance Probability" in display_features
