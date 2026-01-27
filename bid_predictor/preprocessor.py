import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

AUCTION_DATE_COLS = [
    "carrier_code",
    "partner_id",
    "flight_number",
    "origination",
    "destination",
    "travel_date",
    "upgrade_type",
]

CABIN_DATE_COLS = [
    "carrier_code",
    "partner_id",
    "flight_number",
    "origination",
    "destination",
    "travel_date_local",
    "cabin_type",
]


AUCTION_DATETIME_COLS = [
    "carrier_code",
    "partner_id",
    "flight_number",
    "origination",
    "destination",
    "departure_timestamp",
    "upgrade_type",
]

CABIN_DATETIME_COLS = [
    "carrier_code",
    "partner_id",
    "flight_number",
    "origination",
    "destination",
    "departure_local_date_time",
    "cabin_type",
]

def load_flight_data(path):
    """Load raw flight metadata CSVs or Parquets and derive calendar helper columns."""
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")


    df["departure_local_date_time"] = pd.to_datetime(df["departure_local_date_time"], errors="coerce")
    df["travel_date_local"] = pd.to_datetime(
        df["departure_local_date_time"].apply(lambda x: x.date()),
        errors="coerce"
    )


    num_flight_rows = len(df)
    num_flight_cabins = len(df[CABIN_DATETIME_COLS].drop_duplicates())
    logger.info("Loaded Flight Table")
    logger.info(f"Number of rows: {num_flight_rows:,}")
    logger.info(f"Number of unique auctions/cabins: {num_flight_cabins:,}\n")

    return df


def load_offer_data(path):
    """Load bid offer CSVs or Parquets and normalize column names and categorical fields."""
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")
    df = df.rename(
        columns={
            "operating_carrier": "carrier_code",
            "operating_flight_num": "flight_number",
            "ord_multiplier_fare_class": "multiplier_fare_class", 
            "ord_multiplier_loyalty": "multiplier_loyalty", 
            "ord_multiplier_success_history": "multiplier_success_history", 
            "ord_multiplier_payment_type": "multiplier_payment_type",
        },
        errors="ignore",
    )

    df["created"] = pd.to_datetime(df["created"], errors="coerce")

    df["travel_date"] = pd.to_datetime(df["travel_dt"], errors="coerce")

    df["decision_timestamp"] = pd.to_datetime(
        df["upgrade_timestamp"].combine_first(df["expiration_timestamp"])
    )

    dates = pd.to_datetime(df["travel_date"], errors="coerce")
    times = pd.to_timedelta(df["dep_tm"], errors="coerce")
    df["departure_timestamp"] = dates + times

    df["departure_timestamp_utc"] = df["departure_timestamp"] - pd.to_timedelta(
        df["utc_diff"], "m"
    )

    df["usd_base_amount"] = (df["base_amount"].astype(float) * df["inverse_rate"].astype(float)).round(2)

    num_offer_rows = len(df)
    num_offer_bids = len(df[["id"] + AUCTION_DATE_COLS].drop_duplicates())
    num_offer_cabins = len(df[AUCTION_DATE_COLS].drop_duplicates())
    logger.info("Finished Loading Offers Table")
    logger.info(f"Number of rows: {num_offer_rows:,}")
    logger.info(f"Number of unique bids: {num_offer_bids:,}")
    logger.info(f"Number of unique auctions/cabins: {num_offer_cabins:,}\n")

    return df


def preprocess_data(df_flights, df_offers):
    """Join flight and offer datasets and engineer shared temporal features."""

    start = str(max(df_offers["travel_date"].min(), df_flights["travel_date_local"].min()).date())
    end = str(min(df_offers["travel_date"].max(), df_flights["travel_date_local"].max()).date())

    df_flights = df_flights[(df_flights["travel_date_local"]>=start)&(df_flights["travel_date_local"]<=end)]
    df_offers = df_offers[(df_offers["travel_date"]>=start)&(df_offers["travel_date"]<=end)]

    num_flight_rows = len(df_flights)
    num_flight_cabins = len(df_flights[CABIN_DATETIME_COLS].drop_duplicates())

    num_offer_rows = len(df_offers)
    num_offer_bids = len(df_offers[["id"] + AUCTION_DATE_COLS].drop_duplicates())
    num_offer_cabins = len(df_offers[AUCTION_DATE_COLS].drop_duplicates())

    logger.info(f"Filtered Flights and Offers bewteen {start} and {end}")
    logger.info("Flight Table")
    logger.info(f"Number of rows: {num_flight_rows:,}")
    logger.info(f"Number of unique auctions/cabins: {num_flight_cabins:,}\n")
    logger.info("Offers Table")
    logger.info(f"Number of rows: {num_offer_rows:,}")
    logger.info(f"Number of unique bids: {num_offer_bids:,}")
    logger.info(f"Number of unique auctions/cabins: {num_offer_cabins:,}\n")


    flight_dups = df_flights.groupby(CABIN_DATE_COLS, observed=True).size().sort_values()

    if (flight_dups>2).any():
        logger.warn(
            "There are records in the flight table that have more\n"
            "than 2 duplicates with the same flight identifiers.\n"
            "This might mean that you have not filtered out the\n"
            "right booking_fare_class types."
        )


    df_flights_dedup = df_flights.merge(
        flight_dups[flight_dups == 1].reset_index().drop(columns=0), on=CABIN_DATE_COLS
    )

    data_dedup = df_offers.merge(
        df_flights_dedup,
        left_on=AUCTION_DATE_COLS,
        right_on=CABIN_DATE_COLS,
        how="inner"
    )

    df_flights_dup = df_flights.merge(
        flight_dups[flight_dups > 1].reset_index().drop(columns=0), on=CABIN_DATE_COLS
    )

    data_dup = df_offers.merge(
        df_flights_dup,
        left_on=AUCTION_DATETIME_COLS,
        right_on=CABIN_DATETIME_COLS,
        how="inner"
    )

    data = pd.concat((data_dedup, data_dup)).reset_index(drop=True)
    data["departure_timestamp"] = data["departure_local_date_time"] # <- trust departure datetime from flight table over offers table  

    num_rows = len(data)
    num_bids = len(data[["id"] + AUCTION_DATE_COLS].drop_duplicates())
    num_cabins = len(data[AUCTION_DATE_COLS].drop_duplicates())

    logger.info("Offers+Flight Table")
    logger.info(f"Number of rows: {num_rows:,}")
    logger.info(f"Number of unique bids: {num_bids} (dropped bids: {num_offer_bids-num_bids:,})")
    logger.info(f"Number of unique auctions/cabins: {num_cabins:,}\n")

    if num_rows!=num_bids:
        logger.warn(
            "There are duplicate offer id's in the output table.\n"
            "You should double check that nothing is wrong!"
        )

    data["departure_date_utc"] = pd.to_datetime(data["departure_date_utc"], errors="coerce")

    for i in [1, 12, 24, 48, 72]:
        data[f"event_date_{i:02d}h"] = pd.to_datetime(data[f"event_date_{i:02d}h"], errors="coerce")
        data[f"update_time_{i:02d}h"] = pd.to_datetime(data[f"update_time_{i:02d}h"], errors="coerce")

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

    data["flight_number"] = pd.Categorical(data["flight_number"])
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
