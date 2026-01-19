import re
from typing import List
from bid_predictor.preprocessor import (
    load_flight_data,
    load_offer_data,
    preprocess_data,
    get_seats_available,
    AUCTION_DATE_COLS,
    CABIN_DATE_COLS,
)


BUCKET = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
PREFIX = "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output/bsp-historical"
OUTPUT_PREFIX = "s3://amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8/dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/data"


avail_cols: List[str] = [
    "available_count_01h",
    "available_count_12h",
    "available_count_24h",
    "available_count_48h",
    "available_count_72h",
]

event_date_cols: List[str] = [
    "event_local_date_01h",
    "event_local_date_12h",
    "event_local_date_24h",
    "event_local_date_48h",
    "event_local_date_72h",
]

event_utc_date_cols: List[str] = [
    "event_date_01h",
    "event_date_12h",
    "event_date_24h",
    "event_date_48h",
    "event_date_72h",
]

update_time_cols: List[str] = [
    "update_time_01h",
    "update_time_12h",
    "update_time_24h",
    "update_time_48h",
    "update_time_72h",
]

features: List[str] = (
    [
        "id",
        "partner_id",
        "conf_num",
        "offer_status",
        "carrier_code",
        "flight_number",
        "origination",
        "destination",
        "travel_date",
        "item_count",
        "usd_base_amount",
        "fare_class",
        "offer_time",
        "multiplier_fare_class",
        "multiplier_loyalty",
        "multiplier_success_history",
        "multiplier_payment_type",
        "from_cabin",
        "upgrade_type",
    ]
    + [
        "decision_timestamp",
        "departure_timestamp",
        "departure_local_date_time",
        "created",
    ]
    + event_date_cols
    + avail_cols
)


if __name__ == "__main__":
    carrier_code = "EY"
    
    data_dir = f"{carrier_code}/historical-offers-availability-2026-01-14T05-05-05"

    if carrier_code=="SV":
        output_path = "saudia/bid_and_flight_data_20260107_v2.parquet"
    elif carrier_code=="EY":
        output_path = "etihad/bid_and_flight_data_20260107_v2.parquet"

    flights_file = f"s3://{BUCKET}/{PREFIX}/{data_dir}/flight-data.parquet"
    offers_file = (
        f"s3://{BUCKET}/{PREFIX}/{data_dir}/offer-data.parquet"
    )

    df_flights = load_flight_data(flights_file)
    if carrier_code=="EY":
        df_flights = df_flights[
            (df_flights.booking_fare_class.isin(["F","J","Y"]))
            &(df_flights.cabin_type.isin(["BUSINESS","FIRST"]))
        ]
    elif carrier_code=="SV":
        df_flights = df_flights[
            (df_flights.cabin_type.isin(["BUSINESS","FIRST"]))
        ]

    
    df_flights = df_flights[
        df_flights.columns.intersection(features).tolist()
        + event_utc_date_cols
        + update_time_cols
        + [
            "travel_date_local",
            "cabin_type",
            "departure_date_utc",
        ]
    ]

    df_offers = load_offer_data(offers_file)

    print()
    print(df_offers.groupby("offer_status").size().sort_values()[::-1])
    print()

    df_offers = df_offers[df_offers.offer_status.isin(["TICKETED", "EXPIRED", "CC_AUTH_DECLINED", "CC_AUTH_RETRY"])]
    num_bids1 = len(df_offers[["id"] + AUCTION_DATE_COLS].drop_duplicates())
    print(f"Number of unique bids (after status filter): {num_bids1}")
    
    df_offers = df_offers[df_offers.instant_upgrade == 0] # Dropping instant upgrades

    num_bids2 = len(df_offers[["id"] + AUCTION_DATE_COLS].drop_duplicates())
    print(f"Number of unique bids (after instant_upgrade filter): {num_bids2} (change: {num_bids1 - num_bids2})")

    # extract "FIRST" or "BUSINESS" when they appear as suffixes like "XXX_FIRST" or "XXX_BUSINESS"
    df_offers["upgrade_type"] = (
        df_offers["upgrade_type"]
        .str.extract(r"(?:_|^)(FIRST|BUSINESS|EXTRA_SEAT)(?:$|\b)", flags=re.IGNORECASE)[0]
        .str.upper()
    )
    # use None for no-match (consistent with earlier cells)
    df_offers["upgrade_type"] = df_offers["upgrade_type"].where(
        df_offers["upgrade_type"].notna(), None
    )

    print()
    print(df_offers.groupby("upgrade_type").size().sort_values()[::-1])
    print()

    df_offers = df_offers[df_offers["upgrade_type"].isin(["BUSINESS","FIRST"])]

    num_bids3 = len(df_offers[["id"] + AUCTION_DATE_COLS].drop_duplicates())
    print(f"Number of unique bids (after upgrade_type filter): {num_bids3} (change: {num_bids2 - num_bids3})\n")

    df_offers = df_offers[
        df_offers.columns.intersection(features).tolist()
    ].reset_index(drop=True)

    data = preprocess_data(df_flights, df_offers)

    data["seats_available"] = data.apply(get_seats_available, axis=1)

    data = data[features + ["seats_available"]]

    data.to_parquet(
        f"{OUTPUT_PREFIX}/{output_path}",
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )