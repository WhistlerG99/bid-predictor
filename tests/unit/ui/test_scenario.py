import pandas as pd

from bid_predictor.ui.scenario import extract_baseline_snapshot, records_to_dataframe


def test_records_to_dataframe_converts_timestamp_columns():
    records = [
        {
            "departure_timestamp": "2023-07-01T10:15:00",
            "current_timestamp": "2023-07-01T08:15:00",
            "travel_date": "2023-07-01",
            "item_count": 2,
        },
        {
            "departure_timestamp": "2023-07-02T09:00:00",
            "current_timestamp": "2023-07-02T07:30:00",
            "travel_date": "2023-07-02",
            "item_count": 3,
        },
    ]

    df = records_to_dataframe(records)

    assert pd.api.types.is_datetime64_any_dtype(df["departure_timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(df["current_timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(df["travel_date"])
    assert list(df["item_count"]) == [2, 3]


def test_records_to_dataframe_handles_empty_input():
    df = records_to_dataframe(None)
    assert df.empty


def test_extract_baseline_snapshot_merges_snapshots():
    dataset = pd.DataFrame(
        [
            {
                "carrier_code": "AC",
                "flight_number": "100",
                "travel_date": pd.Timestamp("2023-08-01"),
                "upgrade_type": "Plus",
                "snapshot_num": 1,
                "bid_number": "A",
                "usd_base_amount": 100.0,
            },
            {
                "carrier_code": "AC",
                "flight_number": "100",
                "travel_date": pd.Timestamp("2023-08-01"),
                "upgrade_type": "Plus",
                "snapshot_num": 2,
                "bid_number": "A",
                "usd_base_amount": 120.0,
            },
            {
                "carrier_code": "AC",
                "flight_number": "100",
                "travel_date": pd.Timestamp("2023-08-01"),
                "upgrade_type": "Plus",
                "snapshot_num": 1,
                "bid_number": "B",
                "usd_base_amount": 80.0,
            },
        ]
    )

    snapshot_df, label = extract_baseline_snapshot(
        dataset, "AC", "100", "2023-08-01", "Plus"
    )

    assert len(snapshot_df) == 2
    assert list(snapshot_df["Bid #"]) == [1, 2]
    # Latest snapshot for bid A should be retained (value 120)
    assert snapshot_df.loc[snapshot_df["Bid #"] == 1, "usd_base_amount"].iloc[0] == 120.0
    assert label == "across 2 snapshots"
