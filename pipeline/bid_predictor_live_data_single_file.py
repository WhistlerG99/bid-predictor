import subprocess
import sys

# Install dependencies at runtime
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "pyarrow", "sqlalchemy", "psycopg2-binary", "requests", "boto3"])

# Now import them
import argparse
import boto3
import pandas as pd
import requests
from datetime import datetime
import uuid
import psycopg2
from sqlalchemy import create_engine
import sqlalchemy


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--port", default="5439")
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

    for _, row in partners_df.iterrows():
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
        if row_count == 0:
            print(f"No data for carrier {carrier}. Skipping.")
            continue

        # Deduplicate
        df = df.sort_values("created_timestamp", ascending=False).drop_duplicates("offer_id")
        print(f"Rows after deduplication: {len(df)}")

        # Convert to USD
        df["usd_base_amount"] = df.apply(lambda x: convert_to_usd(x["base_amount"], x["cur_code"]), axis=1)
        df.drop(columns=["base_amount", "cur_code"], inplace=True)

        # Reorder columns
        cols = [c for c in df.columns if c not in ("conf_num", "utc_diff")] + ["conf_num", "utc_diff"]
        df = df[cols]

        # Save single Parquet file to S3
        save_single_parquet_file(df, OUTPUT_S3_PATH)
    conn.close()

    print("Job completed successfully.")

if __name__ == "__main__":
    main()
