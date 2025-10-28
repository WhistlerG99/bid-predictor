import pandas as pd

from bid_predictor.ui.scenario import records_to_dataframe


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
