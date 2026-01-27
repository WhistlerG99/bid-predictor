import pandas as pd

from bid_predictor import preprocessor


def test_get_seats_available_selects_correct_window():
    row = {
        "decision_timestamp": pd.Timestamp("2023-01-02 12:00"),
        "event_local_date_01h": pd.Timestamp("2023-01-02 13:00"),
        "available_count_01h": 1,
        "event_local_date_12h": pd.Timestamp("2023-01-02 12:00"),
        "available_count_12h": 12,
        "event_local_date_24h": pd.Timestamp("2023-01-01 12:00"),
        "available_count_24h": 24,
        "event_local_date_48h": pd.Timestamp("2022-12-31 12:00"),
        "available_count_48h": 48,
        "event_local_date_72h": pd.Timestamp("2022-12-30 12:00"),
        "available_count_72h": 72,
    }
    assert preprocessor.get_seats_available(row) == 24


def test_load_offer_data_filters_status(tmp_path):
    csv = tmp_path / "offers.csv"
    csv.write_text(
        """operating_carrier,operating_flight_num,offer_status,travel_dt,travel_date,travel_year,travel_month,travel_dow
AC,1,TICKETED,2023-01-01,2023-01-01,2023,1,0
AC,1,CANCELLED,2023-01-02,2023-01-02,2023,1,1
"""
    )
    df = preprocessor.load_offer_data(csv)
    assert set(df["offer_status"].unique()) == {"TICKETED"}
