import pandas as pd


def datetime_join_with_unique_date_fallback(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on_dt_1: str,
    on_dt_2: str,
    on_other: list[str] | None = None,
    suffixes=("_x", "_y"),
):
    """
    Join df1 and df2 in two passes:
      1) Exact match on datetime columns + on_other
      2) For remaining rows, match on DATE(on_dt) + on_other, but only if
         that (date + on_other) key is unique in each dataframe.

    Returns a single DataFrame with a 'match_type' column in {'exact','date_only'}.
    """
    on_other = on_other or []

    # Ensure datetimes
    df1 = df1.copy()
    df2 = df2.copy()
    df1[on_dt_1] = pd.to_datetime(df1[on_dt_1])
    df2[on_dt_2] = pd.to_datetime(df2[on_dt_2])

    # Give each row a stable id so we can avoid double-using rows
    df1["_rowid1"] = range(len(df1))
    df2["_rowid2"] = range(len(df2))

    # ---------- Pass 1: exact datetime match ----------
    exact_keys_left  = [on_dt_1] + on_other
    exact_keys_right = [on_dt_2] + on_other

    exact = df1.merge(
        df2,
        left_on=exact_keys_left,
        right_on=exact_keys_right,
        how="inner",
        suffixes=suffixes,
        copy=False,
    )
    exact["match_type"] = "exact"

    used_left_ids  = set(exact["_rowid1"])
    used_right_ids = set(exact["_rowid2"])

    # Filter remaining (unmatched) rows on each side
    rem1 = df1[~df1["_rowid1"].isin(used_left_ids)].copy()
    rem2 = df2[~df2["_rowid2"].isin(used_right_ids)].copy()

    # Short-circuit if nothing left to match
    if rem1.empty or rem2.empty:
        out = exact.drop(columns=["_rowid1", "_rowid2"])
        return out

    # ---------- Pass 2: date-only unique match ----------
    rem1["_date_key"] = rem1[on_dt_1].dt.normalize()
    rem2["_date_key"] = rem2[on_dt_2].dt.normalize()

    date_keys = ["_date_key"] + on_other

    # Keep only keys that are unique in each dataframe
    rem1_key_counts = rem1.groupby(date_keys, dropna=False, observed=False).size().rename("n1")
    rem2_key_counts = rem2.groupby(date_keys, dropna=False, observed=False).size().rename("n2")

    rem1 = rem1.merge(rem1_key_counts.reset_index(), on=date_keys, how="left")
    rem2 = rem2.merge(rem2_key_counts.reset_index(), on=date_keys, how="left")

    rem1_unique = rem1[rem1["n1"] == 1].drop(columns=["n1"])
    rem2_unique = rem2[rem2["n2"] == 1].drop(columns=["n2"])

    date_only = rem1_unique.merge(
        rem2_unique,
        on=date_keys + on_other,  # date + other join columns
        how="inner",
        suffixes=suffixes,
        copy=False,
    )
    date_only["match_type"] = "date_only"

    # Combine results
    out = pd.concat([exact, date_only], ignore_index=True)

    # Cleanup temp columns
    drop_cols = ["_rowid1", "_rowid2", "_date_key"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out


def load_flight_data(path):
    """Load raw flight metadata CSVs and derive calendar helper columns."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"cabin_type": "upgrade_type"})

    df["departure_date_utc"] = pd.to_datetime(df.departure_date_utc)
    df["travel_year_month"] = pd.to_datetime(
        df.apply(
            lambda x: f"{x['departure_date_utc'].year}-{x['departure_date_utc'].month}",
            axis=1,
        )
    )
    df["travel_date_utc"] = pd.to_datetime(
        df["departure_date_utc"].apply(lambda x: x.date())
    )
    df["departure_local_date_time"] = pd.to_datetime(df["departure_local_date_time"])
    df["travel_date_local"] = pd.to_datetime(
        df["departure_local_date_time"].apply(lambda x: x.date())
    )
    df["flight_number"] = pd.Categorical(df.flight_number)
    return df


def load_offer_data(path):
    """Load bid offer CSVs and normalize column names and categorical fields."""
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(
        columns={
            "operating_carrier": "carrier_code",
            "operating_flight_num": "flight_number",
        }
    )
    df = df[df.offer_status.isin(["TICKETED", "EXPIRED"])]
    df["travel_date"] = pd.to_datetime(df.travel_dt)

    df["departure_timestamp"] = pd.to_datetime(df[["travel_dt", "dep_tm"]].agg(" ".join, axis=1))

    df[["travel_year", "travel_month", "travel_dow"]] = df.apply(
        lambda x: (
            x["travel_date"].year,
            x["travel_date"].month,
            x["travel_date"].day_of_week,
        ),
        axis=1,
        result_type="expand",
    )
    for col in ["flight_number", "travel_year", "travel_month", "travel_dow"]:
        df[col] = pd.Categorical(df[col])
    return df


def preprocess_data(df_flights, df_offers):
    df_flights_ = df_flights.drop(columns=["equip"]).drop_duplicates(
        subset=[
            "carrier_code",
            "flight_number",
            "departure_local_date_time",
            "upgrade_type",
            "origination",
            "destination",
            "booking_fare_class",
        ]
    )
    df_offers_ = df_offers.replace(
        {
            "from_cabin": {
                "REGIONAL_PREMIUM_ECONOMY": "PREMIUM_ECONOMY",
                "REGIONAL_BUSINESS": "BUSINESS",
            },
            "upgrade_type": {
                "REGIONAL_PREMIUM_ECONOMY": "PREMIUM_ECONOMY",
                "REGIONAL_BUSINESS": "BUSINESS",
            },
        }
    )

    data = datetime_join_with_unique_date_fallback(
        df_offers_,
        df_flights_,
        "departure_timestamp",
        "departure_local_date_time",
        [
            "carrier_code",
            "flight_number",
            "partner_id",
            "upgrade_type",
        ],
    )

    data = data.replace(
        {
            k: {"PREMIUM_ECONOMY": "P-ECON", "ECONOMY": "ECON", "BUSINESS": "BUS"}
            for k in ["upgrade_type"]
        }
    )
    data["decision_timestamp"] = pd.to_datetime(
        data["upgrade_timestamp"].combine_first(data["expiration_timestamp"])
    )
    for i in [1, 12, 24, 48, 72]:
        data[f"event_date_{i:02d}h"] = pd.to_datetime(data[f"event_date_{i:02d}h"])
        data[f"update_time_{i:02d}h"] = pd.to_datetime(data[f"update_time_{i:02d}h"])
    data["created"] = pd.to_datetime(data["created"])

    data["departure_timestamp_utc"] = data["departure_local_date_time"] - pd.to_timedelta(
        data["utc_diff"], "m"
    )
    for i in [1, 12, 24, 48, 72]:
        data[f"event_local_date_{i:02d}h"] = data[f"event_date_{i:02d}h"] + (
            data["departure_local_date_time"] - data["departure_date_utc"]
        )
        data[f"update_local_time_{i:02d}h"] = data[f"update_time_{i:02d}h"] + (
            data["departure_local_date_time"] - data["departure_date_utc"]
        )
    data["offer_time"] = data.apply(
        lambda x: (x["departure_local_date_time"] - x["created"]).total_seconds()
        / (60 * 60 * 24),
        axis=1,
    )
    data = data[data.instant_upgrade == 0].reset_index(drop=True)
    data["usd_base_amount"] = (data["base_amount"] * data["inverse_rate"]).round(2)
    return data    


def get_seats_available(row):
    """Estimate seats available at decision time using staggered inventory snapshots."""
    time_columns = [
        ("event_local_date_01h", "available_count_01h"),
        ("event_local_date_12h", "available_count_12h"),
        ("event_local_date_24h", "available_count_24h"),
        ("event_local_date_48h", "available_count_48h"),
        ("event_local_date_72h", "available_count_72h"),
    ]
    for i in range(len(time_columns) - 1):
        if (
            row["decision_timestamp"] <= row[time_columns[i][0]]
            and row["decision_timestamp"] > row[time_columns[i + 1][0]]
        ):
            return row[time_columns[i + 1][1]]
    if row["decision_timestamp"] <= row[time_columns[-1][0]]:
        return row[time_columns[-1][1]]
    return None
