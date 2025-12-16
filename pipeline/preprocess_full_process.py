# preprocess.py

import os
import glob
import logging
import json
import boto3
import mlflow
import pandas as pd

import argparse
import requests
from datetime import datetime
import uuid
import psycopg2
from sqlalchemy import create_engine
import sqlalchemy


INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"
MODEL_CONFIG_DIR = (
    "/opt/ml/processing/model_config"  # CHANGED: new directory for model_config.json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX = os.getenv(
    "MODEL_NAME_PREFIX", "bid-predictor"
)  # e.g. "bid-predictor"
MODEL_REGISTRY_STAGE = os.getenv("MODEL_REGISTRY_STAGE", "Production")

_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not _tracking_uri:
    _default_region = os.getenv("AWS_REGION") or boto3.session.Session().region_name
    if _default_region:
        _tracking_uri = f"https://{_default_region}.api.mlflow.sagemaker.aws"


def _get_mlflow_client() -> mlflow.tracking.MlflowClient:
    if not _tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set and AWS region could not be inferred")

    mlflow.set_tracking_uri(_tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=_tracking_uri)


rate_cache = {}


def convert_to_usd(amount, currency):
    if amount is None or currency is None:
        return None
    currency = currency.upper()
    if currency == "USD":
        return float(amount)
    if currency not in rate_cache:
        url = f"https://currency.stg.internal.plusgrade.com/currency-app/service/currency/json/{currency}/USD"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                rate_cache[currency] = float(resp.json().get("rate", 1))
            else:
                rate_cache[currency] = 1.0
        except Exception:
            rate_cache[currency] = 1.0
    return float(amount) * rate_cache[currency]

def save_single_parquet_file(df, output_s3_path):
    if not output_s3_path.endswith("/"):
        output_s3_path += "/"

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    final_filename = f"availability-offers-{timestamp}.parquet"
    final_s3_key = output_s3_path + final_filename

    tmp_suffix = str(uuid.uuid4())
    tmp_output_path = f"{output_s3_path}tmp_output_{tmp_suffix}/"
    print(f"Writing temporary data → {tmp_output_path}")

    # Save locally first
    local_tmp_file = f"/tmp/{tmp_suffix}.parquet"
    df.to_parquet(local_tmp_file, index=False)

    # Upload to S3
    s3 = boto3.client("s3")
    bucket = output_s3_path.replace("s3://", "").split("/")[0]
    prefix = "/".join(output_s3_path.replace("s3://", "").split("/")[1:])

    # Copy to final location
    s3.upload_file(local_tmp_file, bucket, prefix + final_filename)
    print(f"Final single file saved → {final_s3_key}")


def select_model_for_carrier(carrier_code: str) -> str:
    """
    Resolve the latest MLflow Model Registry version for the carrier and
    return its artifact URI.
    """
    if not MODEL_NAME_PREFIX:
        raise RuntimeError("MODEL_NAME_PREFIX not set")

    model_name = f"{MODEL_NAME_PREFIX}-{carrier_code}"
    client = _get_mlflow_client()

    logger.info(
        "Selecting MLflow model", extra={"model": model_name, "stage": MODEL_REGISTRY_STAGE}
    )

    filter_str = f"name='{model_name}'"
    if MODEL_REGISTRY_STAGE:
        filter_str += f" and current_stage='{MODEL_REGISTRY_STAGE}'"
    versions = list(client.search_model_versions(filter_str))

    if not versions:
        raise RuntimeError(f"No MLflow model versions found for carrier {carrier_code}")

    best_version = max(versions, key=lambda mv: int(mv.version))
    model_data = best_version.source

    logger.info(
        "Selected MLflow model",
        extra={
            "model": model_name,
            "stage": MODEL_REGISTRY_STAGE,
            "version": best_version.version,
            "source": model_data,
        },
    )
    return model_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--port", default="5439")
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--partner_csv_s3_path", required=True)
    parser.add_argument("--temp_s3_path", required=True)
    args = parser.parse_args()
    DB_HOST = args.host
    DB_PORT = int(args.port)
    DB_NAME = args.database
    DB_USER = args.user
    DB_PASSWORD = args.password
    TEMP_S3_PATH = args.temp_s3_path.rstrip("/")
    PARTNER_CSV_S3_PATH = args.partner_csv_s3_path


    # Read partners CSV from S3
    s3 = boto3.client("s3")
    bucket = PARTNER_CSV_S3_PATH.replace("s3://", "").split("/")[0]
    key = "/".join(PARTNER_CSV_S3_PATH.replace("s3://", "").split("/")[1:])
    obj = s3.get_object(Bucket=bucket, Key=key)
    partners_df = pd.read_csv(obj['Body'])

    # Create Redshift connection
    # engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    row = partners_df[partners_df["operating_carrier"] == args.carrier]
    if row.shape[0] == 1:
        row = row.iloc[0]
        carrier = row["operating_carrier"]
        OUTPUT_S3_PATH = row["live_data_s3_path"].rstrip("/")
        print(f"\nProcessing carrier: {carrier} → {OUTPUT_S3_PATH}")

        if carrier == "SV":
            upgrade_type_filter = "'FIRST','BUSINESS'"
        elif carrier == "EY":
            upgrade_type_filter = (
                "'A380_FIRST','FIRST','REGIONAL_FIRST',"
                "'A380_BUSINESS','BUSINESS','REGIONAL_BUSINESS'"
            )
        else:
            upgrade_type_filter = (
                "'A380_FIRST','FIRST','REGIONAL_FIRST',"
                "'A380_BUSINESS','BUSINESS','REGIONAL_BUSINESS'"
            )

        query = f"""
        SELECT
            o.id AS offer_id,
            o.partner_id,
            o.product_id,
            o.operating_carrier AS carrier_code,
            o.operating_flight_num AS flight_number,
            a.origination AS origination_code,
            a.destination AS destination_code,
            CAST(o.travel_dt || ' ' || o.dep_tm AS TIMESTAMP) AS departure_timestamp,
            o.travel_dt,
            a.available_count AS seats_available,
            o.upgrade_type AS upgrade_type,
            a.cabin_type AS cabin_type,
            o.item_count,
            o.fare_class,
            o.from_cabin,
            o.base_amount,
            o.cur_code,
            o.created AS created_timestamp,
            m.multiplier_fare_class,
            m.multiplier_loyalty,
            m.multiplier_success_history,
            m.multiplier_payment_type,
            o.conf_num,
            o.utc_diff
        FROM prd_offers_rds.offers o
        LEFT JOIN continuous_pricing_rl.availability a
            ON o.operating_carrier = a.airline_code
            AND o.operating_flight_num = a.flight_number
            AND o.travel_dt = a.travel_date
            AND o.upgrade_type = a.upgrade_type
        LEFT JOIN prd_offers_rds.offerranklistdetail m
            ON o.id = m.offer_id
        WHERE 
            o.travel_dt >= CURRENT_DATE
            AND o.travel_dt < DATEADD(day, 5, CURRENT_DATE)
            AND o.operating_carrier = '{carrier}'
            AND o.instant_upgrade = 0
            AND o.offer_status IN ('SUBMITTED', 'UNABLE_TO_TICKET')
            AND o.upgrade_type IN ({upgrade_type_filter})
        """

        # df = pd.read_sql(query, engine)
        # df = pd.read_sql_query(sql=query, con=engine)
        # with engine.connect() as conn:
        #     df = pd.read_sql_query(sql=query, con=conn)
        df = pd.read_sql_query(query, conn)

        row_count = len(df)
        print(f"Rows fetched: {row_count}")
        # if row_count == 0:
        #     print(f"No data for carrier {carrier}. Skipping.")
        #     continue

        # Deduplicate
        df = df.sort_values("created_timestamp", ascending=False).drop_duplicates("offer_id")
        print(f"Rows after deduplication: {len(df)}")

        # Convert to USD
        df["usd_base_amount"] = df.apply(lambda x: convert_to_usd(x["base_amount"], x["cur_code"]), axis=1)
        df.drop(columns=["base_amount", "cur_code"], inplace=True)

        # Reorder columns
        cols = [c for c in df.columns if c not in ("conf_num", "utc_diff")] + ["conf_num", "utc_diff"]
        df = df[cols]
    conn.close()

    timestamp_str = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")

    # parquet_files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
    logger.info(f"Retrieved offer and availability data for carrier {carrier}")

    # if not parquet_files:
    #     raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    # parquet_file = max(parquet_files)
    # logger.info(f"Reading parquet file: {parquet_file}")

    # basename = os.path.basename(parquet_file)
    # prefix = "availability-offers-"
    # suffix = ".parquet"
    # if not (basename.startswith(prefix) and basename.endswith(suffix)):
    #     raise ValueError(f"Unexpected parquet filename format: {basename}")
    # timestamp_str = basename[len(prefix) : -len(suffix)]
    # logger.info(f"Parsed file timestamp: {timestamp_str}")

    # df = pd.read_parquet(parquet_file)

    # ---- NEW: get carrier_code and select model ----
    if "carrier_code" not in df.columns:
        raise KeyError("carrier_code column not found in input file")

    carrier_code = df["carrier_code"].iloc[0].lower()
    logger.info(f"Detected carrier_code: {carrier_code}")

    model_data = select_model_for_carrier(carrier_code)
    # Write JSON config so pipeline can read as a PropertyFile
    os.makedirs(MODEL_CONFIG_DIR, exist_ok=True)
    config_path = os.path.join(MODEL_CONFIG_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({"model_data": model_data}, f)
    logger.info(f"Wrote model_config.json to {config_path}")
    # ---- END NEW STUFF ----

    # Existing feature engineering below
    df = df.rename(columns={"travel_dt": "travel_date"})

    df["offer_time"] = df.apply(
        lambda x: (x["departure_timestamp"] - x["created_timestamp"]).total_seconds()
        / (60 * 60 * 24),
        axis=1,
    )
    df["snapshot_num"] = 1
    df["current_timestamp"] = (
        pd.Timestamp.now("utc").round(freq="s") + pd.to_timedelta(df["utc_diff"], "m")
    ).dt.tz_localize(None)
    df["file_timestamp"] = timestamp_str

    for c in ["travel_date", "departure_timestamp"]:
        df[c] = pd.to_datetime(df[c])

    logger.info(f"Loaded DataFrame with shape: {df.shape}")

    processed_path = os.path.join(OUTPUT_DIR, "processed.parquet")
    df.to_parquet(processed_path)
    logger.info(f"Wrote processed Parquet to: {processed_path}")


if __name__ == "__main__":
    main()
