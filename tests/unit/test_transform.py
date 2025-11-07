import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bid_predictor import transform


def make_sample_df():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "carrier_code": ["AC", "AC", "UA", "UA"],
            "flight_number": ["1", "1", "2", "2"],
            "departure_timestamp": pd.to_datetime(
                ["2023-07-01 00:00", "2023-07-02 00:00", "2023-07-03 00:00", "2023-07-04 00:00"]
            ),
            "current_timestamp": pd.to_datetime(
                ["2023-06-30 00:00", "2023-07-01 00:00", "2023-07-02 00:00", "2023-07-03 00:00"]
            ),
            "upgrade_type": ["BUS", "BUS", "BUS", "BUS"],
            "snapshot_num": [0, 0, 0, 0],
            "travel_date": pd.to_datetime(
                ["2023-07-02", "2023-07-03", "2023-07-04", "2023-07-05"]
            ),
            "usd_base_amount": [200.0, 300.0, 400.0, 500.0],
        }
    )
    df["carrier_code"] = df["carrier_code"].astype("category")
    df["flight_number"] = df["flight_number"].astype("category")
    return df


def test_add_missing_indicator_custom_handles_missing_column():
    df = make_sample_df()
    indicator = transform.AddMissingIndicatorCustom(variables=["a", "missing"])
    indicator.fit(df)
    transformed = indicator.transform(df)
    assert "a_na" in transformed.columns
    assert "missing_na" not in transformed.columns


def test_arbitrary_number_imputer_filters_unknown_columns():
    df = make_sample_df()
    imputer = transform.ArbitraryNumberImputerCustom(imputer_dict={"a": -1, "x": 0})
    imputer.fit(df)
    transformed = imputer.transform(df)
    assert transformed["a"].isna().sum() == 0
    assert "x" not in transformed.columns


def test_mean_median_imputer_skips_unknown_columns():
    df = make_sample_df()
    imputer = transform.MeanMedianImputerCustom(variables=["a", "missing"], imputation_method="median")
    imputer.fit(df)
    transformed = imputer.transform(df)
    assert transformed["a"].isna().sum() == 0


def test_outlier_capper_noop_without_matching_columns():
    df = make_sample_df()
    capper = transform.ArbitraryOutlierCapperCustom(min_capping_dict={"missing": 0})
    capper.fit(df)
    transformed = capper.transform(df)
    assert_frame_equal(transformed, df)


def test_arbitrary_discretiser_custom_applies_bins():
    df = make_sample_df()
    discretiser = transform.ArbitraryDiscretiserCustom(binning_dict={"b": [-np.inf, 15, np.inf]})
    discretiser.fit(df)
    transformed = discretiser.transform(df)
    assert transformed["b"].dtype.name == "category"
    assert len(transformed["b"].cat.categories) == 2


def test_add_flight_code_creates_category():
    df = make_sample_df()
    result = transform.add_flight_code(df)
    assert "flight_code" in result.columns
    assert result["flight_code"].dtype.name == "category"


def test_add_days_b4_depart():
    df = make_sample_df()
    result = transform.add_days_b4_depart(df)
    assert "days_before_departure" in result.columns
    assert np.isclose(result.loc[0, "days_before_departure"], 1.0)


def test_group_features_adds_counts():
    df = make_sample_df()
    result = transform.add_group_features(df)
    assert "num_offers" in result.columns
    assert "usd_base_amount_max" in result.columns


def test_quantiles_vectorized_matches_numpy():
    vals = np.array([1, 2, 3, 4])
    q25, q50, q75 = transform.quantiles_vectorized(vals)
    assert q25.shape == (4,)
    assert np.all(q25 <= q50)
    assert np.all(q50 <= q75)


def test_column_reducer_selects_existing_columns():
    df = make_sample_df()
    reducer = transform.ColumnReducer(["a", "b", "c"])
    reducer.fit(df)
    reduced = reducer.transform(df)
    assert list(reduced.columns) == ["a", "b"]


def _make_eval_transform_df(active_snapshot, inactive_snapshot):
    base = {
        "id": [1, 1],
        "active": [True, False],
        "carrier_code": ["AC", "AC"],
        "flight_number": ["1", "1"],
        "travel_date": pd.to_datetime(["2023-07-02", "2023-07-02"]),
        "upgrade_type": ["BUS", "BUS"],
        "snapshot_num": [active_snapshot, inactive_snapshot],
        "last_snapshot": [active_snapshot, inactive_snapshot],
    }
    return pd.DataFrame(base)


def test_eval_transforms_merges_when_last_snapshot_is_numeric_like_strings():
    df = _make_eval_transform_df("001", "001")

    def annotate(frame):
        frame = frame.copy()
        frame["filled"] = 7
        return frame

    result = transform.eval_transforms(df, annotate, ["filled"])
    inactive_row = result.loc[result.active == False].iloc[0]
    assert inactive_row["filled"] == 7


def test_eval_transforms_merges_when_last_snapshot_is_non_numeric():
    df = _make_eval_transform_df("snap_1", "snap_1")

    def annotate(frame):
        frame = frame.copy()
        frame["bonus"] = 3
        return frame

    result = transform.eval_transforms(df, annotate, ["bonus"])
    inactive_row = result.loc[result.active == False].iloc[0]
    assert inactive_row["bonus"] == 3
