import pandas as pd
import pytest

from bid_predictor_ui.scenario import (
    TIME_TO_DEPARTURE_SCENARIO_KEY,
    ScenarioFeature,
    build_adjustment_grid,
    extract_baseline_snapshot,
    extract_global_baseline_values,
    select_baseline_snapshot,
    resolve_locked_cells,
    records_to_dataframe,
)


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


def test_extract_global_baseline_values_returns_shared_features():
    df = pd.DataFrame(
        [
            {
                "Bid #": 1,
                "seats_available": 5,
                "departure_timestamp": pd.Timestamp("2023-09-01 12:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 08:00:00"),
            },
            {
                "Bid #": 2,
                "seats_available": 5,
                "departure_timestamp": pd.Timestamp("2023-09-01 15:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 11:00:00"),
            },
        ]
    )

    baselines = extract_global_baseline_values(df)

    assert baselines["seats_available"] == 5.0
    assert baselines[TIME_TO_DEPARTURE_SCENARIO_KEY] == pytest.approx(4.0)


def test_select_baseline_snapshot_matches_baseline_time():
    df = pd.DataFrame(
        [
            {
                "Bid #": 1,
                "snapshot_num": 1,
                "departure_timestamp": pd.Timestamp("2023-09-01 12:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 07:00:00"),
            },
            {
                "Bid #": 2,
                "snapshot_num": 7,
                "departure_timestamp": pd.Timestamp("2023-09-01 12:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 08:00:00"),
            },
        ]
    )

    selected = select_baseline_snapshot(df, baseline_time_hours=4.0)

    assert selected == 7


def test_select_baseline_snapshot_falls_back_to_first_snapshot():
    df = pd.DataFrame(
        [
            {
                "Bid #": 1,
                "snapshot_num": None,
                "departure_timestamp": pd.Timestamp("2023-09-01 12:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 08:00:00"),
            },
            {
                "Bid #": 2,
                "snapshot_num": 5,
                "departure_timestamp": pd.Timestamp("2023-09-01 13:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 10:00:00"),
            },
        ]
    )

    selected = select_baseline_snapshot(df, baseline_time_hours=None)

    assert selected == 5


def test_resolve_locked_cells_identifies_matching_bid():
    feature = ScenarioFeature(
        key="item_count",
        scope="bid",
        label="Item count",
        bid_label=2,
    )
    records = [
        {"Bid #": 1, "item_count": 2},
        {"Bid #": 2, "item_count": 3},
    ]

    locked = resolve_locked_cells(records, feature)

    assert locked == {"bid_1": ["item_count"]}


def test_resolve_locked_cells_returns_empty_for_non_bid_feature():
    feature = ScenarioFeature(key="seats_available", scope="global", label="Seats")
    records = [{"Bid #": 1, "item_count": 2}]

    assert resolve_locked_cells(records, feature) == {}


def test_build_adjustment_grid_applies_global_overrides():
    df = pd.DataFrame(
        [
            {
                "Bid #": 1,
                "seats_available": 9,
                "departure_timestamp": pd.Timestamp("2023-09-01 12:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 08:00:00"),
            },
            {
                "Bid #": 2,
                "seats_available": 9,
                "departure_timestamp": pd.Timestamp("2023-09-01 15:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 11:00:00"),
            },
        ]
    )

    feature = ScenarioFeature(
        key="seats_available",
        scope="global",
        label="Seats available",
        is_integer=True,
    )

    overrides = {TIME_TO_DEPARTURE_SCENARIO_KEY: 6.0}
    grid = build_adjustment_grid(df, feature, start=8, stop=10, count=3, global_overrides=overrides)

    assert sorted(grid["scenario_feature_value"].unique()) == [8, 9, 10]
    for _, row in grid.iterrows():
        expected_current = row["departure_timestamp"] - pd.Timedelta(hours=6)
        assert row["current_timestamp"] == expected_current


def test_build_adjustment_grid_applies_seat_overrides_for_time_feature():
    df = pd.DataFrame(
        [
            {
                "Bid #": 1,
                "seats_available": 9,
                "departure_timestamp": pd.Timestamp("2023-09-01 12:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 08:00:00"),
            },
            {
                "Bid #": 2,
                "seats_available": 7,
                "departure_timestamp": pd.Timestamp("2023-09-01 14:00:00"),
                "current_timestamp": pd.Timestamp("2023-09-01 10:00:00"),
            },
        ]
    )

    feature = ScenarioFeature(
        key=TIME_TO_DEPARTURE_SCENARIO_KEY,
        scope="global",
        label="Time to departure (hours)",
        is_integer=False,
        kind="time_to_departure",
    )

    grid = build_adjustment_grid(
        df,
        feature,
        start=4.0,
        stop=6.0,
        count=3,
        global_overrides={"seats_available": 12},
    )

    assert grid["seats_available"].unique().tolist() == [12.0]
    unique_steps = sorted(grid["scenario_feature_value"].unique())
    assert unique_steps == [4.0, 5.0, 6.0]
    for _, row in grid.iterrows():
        hours = row["scenario_feature_value"]
        expected_current = row["departure_timestamp"] - pd.Timedelta(hours=hours)
        assert row["current_timestamp"] == expected_current
