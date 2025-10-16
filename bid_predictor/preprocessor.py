import pandas as pd


def load_flight_data(path):
    df = pd.read_csv(path, low_memory=False)
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
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(
        columns={
            "operating_carrier": "carrier_code",
            "operating_flight_num": "flight_number",
        }
    )
    df = df[df.offer_status.isin(["TICKETED", "EXPIRED"])]
    df["travel_date"] = pd.to_datetime(df.travel_dt)
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
            "travel_date_local",
            "cabin_type",
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
    data = df_offers_.merge(
        df_flights_,
        left_on=[
            "carrier_code",
            "flight_number",
            "partner_id",
            "travel_date",
            "upgrade_type",
        ],
        right_on=[
            "carrier_code",
            "flight_number",
            "partner_id",
            "travel_date_local",
            "cabin_type",
        ],
        how="inner",
    )
    data = data.replace(
        {
            k: {"PREMIUM_ECONOMY": "P-ECON", "ECONOMY": "ECON", "BUSINESS": "BUS"}
            for k in ["from_cabin", "upgrade_type"]
        }
    )
    data["decision_timestamp"] = pd.to_datetime(
        data["upgrade_timestamp"].combine_first(data["expiration_timestamp"])
    )
    for i in [1, 12, 24, 48, 72]:
        data[f"event_date_{i:02d}h"] = pd.to_datetime(data[f"event_date_{i:02d}h"])
        data[f"update_time_{i:02d}h"] = pd.to_datetime(data[f"update_time_{i:02d}h"])
    data["created"] = pd.to_datetime(data["created"])
    data["departure_local_date_time"] = pd.to_datetime(
        data["departure_local_date_time"]
    )
    data["departure_timestamp"] = pd.to_datetime(
        data[["travel_dt", "dep_tm"]].apply(
            lambda x: x["travel_dt"] + " " + x["dep_tm"], axis=1
        )
    )
    data["departure_timestamp_utc"] = data["departure_timestamp"] - pd.to_timedelta(
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
        lambda x: (x["departure_timestamp"] - x["created"]).total_seconds()
        / (60 * 60 * 24),
        axis=1,
    )
    data = data[data.instant_upgrade == 0].reset_index(drop=True)
    data["usd_base_amount"] = (data["base_amount"] * data["inverse_rate"]).round(2)
    data["flight_number"] = pd.Categorical(data["flight_number"])
    return data


def get_seats_available(row):
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
