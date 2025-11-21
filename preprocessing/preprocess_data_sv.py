import re
from typing import List
from bid_predictor.preprocessor import (
    load_flight_data,
    load_offer_data,
    preprocess_data,
    get_seats_available,
)


BUCKET = "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
PREFIX = "dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/output"
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


features: List[str] = (
    [
        "id",
        "partner_id",
        "conf_num",
        "offer_status",
        "carrier_code",
        "flight_number",
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
    data_dir = "bid-predictor-historical-by-partners-EY-SV/SV"
    
    flights_file = f"s3://{BUCKET}/{PREFIX}/{data_dir}/joined-cols-20251117T202158/"
    offers_file = f"s3://{BUCKET}/{PREFIX}/{data_dir}/joined-offers-ord-20251117T200423/"

    grp_cols = ["flight_number", "carrier_code", "departure_local_date_time", "cabin_type"]

    df_flights = load_flight_data(flights_file)

    grps = df_flights.groupby(grp_cols, observed =True).size().sort_values()
    df_flights = df_flights.merge(grps[grps==1].reset_index().drop(columns=0),on=grp_cols)

    df_offers = load_offer_data(offers_file)

    # extract "FIRST" or "BUSINESS" when they appear as suffixes like "XXX_FIRST" or "XXX_BUSINESS"
    df_offers["upgrade_type"] = (
        df_offers["upgrade_type"]
        .str.extract(r'(?:_|^)(FIRST|BUSINESS)(?:$|\b)', flags=re.IGNORECASE)[0]
        .str.upper()
    )
    # use None for no-match (consistent with earlier cells)
    df_offers["upgrade_type"] = df_offers["upgrade_type"].where(df_offers["upgrade_type"].notna(), None)

    df_offers = df_offers.rename(columns={
            "ord_multiplier_fare_class": "multiplier_fare_class",
            "ord_multiplier_loyalty": "multiplier_loyalty",
            "ord_multiplier_success_history": "multiplier_success_history",
            "ord_multiplier_payment_type": "multiplier_payment_type",
        }
    )

    data = preprocess_data(df_flights, df_offers)
    data["seats_available"] = data.apply(get_seats_available, axis=1)

    data = data[features + ["seats_available"]]

    data.to_parquet(f"{OUTPUT_PREFIX}/saudia/bid_and_flight_data.parquet", coerce_timestamps="us", allow_truncated_timestamps=True)