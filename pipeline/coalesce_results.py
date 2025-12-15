import argparse
import boto3
import pandas as pd
import re
from urllib.parse import urlparse
import io

s3 = boto3.client("s3")

FINAL_COLUMNS = [
    "offer_id", "partner_id", "product_id", "carrier_code", "flight_number",
    "departure_timestamp", "origination_code", "destination_code",
    "upgrade_type", "accept_prob"
]

COLUMN_MAPPING = {
    "offer_id": "offer_id",
    "partner_id": "partner_id",
    "product_id": "product_id",
    "carrier_code": "carrier_code",
    "flight_number": "flight_number",
    "departure_timestamp": "departure_timestamp",
    "origination_code": "origination_code",
    "destination_code": "destination_code",
    "upgrade_type": "upgrade_type",
    "acceptance_prob": "accept_prob",
    "accept_prob": "accept_prob"
}


def read_hwm(path):
    parsed = urlparse(path)
    try:
        resp = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        hwm = resp["Body"].read().decode("utf-8").strip()
        print(f"Read HWM from {path}: {hwm}")
        return hwm
    except s3.exceptions.NoSuchKey:
        print(f"No HWM found at {path}. Starting fresh.")
        return None


def write_hwm(path, ts):
    parsed = urlparse(path)
    s3.put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=ts)
    print(f"Wrote HWM to {path}: {ts}")


def list_csv_files(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                files.append(obj["Key"])
    print(f"Found {len(files)} CSV files under s3://{bucket}/{prefix}")
    return files


def extract_ts(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", filename)
    if not match:
        print(f"Warning: Could not extract timestamp from filename {filename}")
        return None
    raw = match.group(1)
    ts = raw.replace("-", ":", 2).replace("-", ":", 1)
    return ts


def read_csv(bucket, key):
    print(f"Reading CSV from s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str)
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Read {len(df)} rows from {key}")
    return df


def append_csv(df, output_path):
    parsed = urlparse(output_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    try:
        existing_obj = s3.get_object(Bucket=bucket, Key=key)
        existing_df = pd.read_csv(io.BytesIO(existing_obj["Body"].read()), dtype=str)
        existing_df = existing_df.rename(columns={k: v for k, v in COLUMN_MAPPING.items()})
        existing_df = existing_df[[c for c in FINAL_COLUMNS if c in existing_df.columns]]

        combined = pd.concat([existing_df, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["offer_id"], keep="last")
        print(f"Merged {len(df)} new rows with existing {len(existing_df)} rows → {len(combined)} total rows")
    except s3.exceptions.NoSuchKey:
        combined = df.drop_duplicates(subset=["offer_id"], keep="last")
        print(f"No existing CSV found. Writing {len(combined)} rows.")

    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"CSV updated at {output_path}")


def load_lookup(path):
    print(f"Loading lookup table from {path}")
    parsed = urlparse(path)
    obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str)
    carriers = df["operating_carrier"].dropna().unique().tolist()
    print(f"Found {len(carriers)} carriers: {carriers}")
    return carriers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--hwm_path", required=True)
    parser.add_argument("--lookup_path", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    s3_bucket = args.s3_bucket
    hwm_path = args.hwm_path
    lookup_path = args.lookup_path
    output_csv = args.output_csv

    carriers = load_lookup(lookup_path)

    for carrier in carriers:
        print(f"\nProcessing carrier: {carrier}")
        # prefix = f"dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/offer_probability_csv/{carrier}/"
        prefix = f"bid_success_predictor/results/offer_probability_csv/{carrier}/"

        last_hwm = read_hwm(hwm_path)
        all_files = list_csv_files(s3_bucket, prefix)

        new_files = []
        timestamps = []

        for key in all_files:
            ts = extract_ts(key)
            if ts and (last_hwm is None or ts > last_hwm):
                new_files.append(key)
                timestamps.append(ts)

        print(f"Found {len(new_files)} new files to process for carrier {carrier}")
        if not new_files:
            print(f"No new files to process for carrier {carrier}.")
            continue

        df_all = pd.concat(
            [read_csv(s3_bucket, key) for key in new_files],
            ignore_index=True
        )
        print(f"Total rows to append for carrier {carrier}: {len(df_all)}")

        append_csv(df_all, output_csv)

        max_ts = max(timestamps)
        write_hwm(hwm_path, max_ts)

    print("\nProcessing finished for all carriers.")


if __name__ == "__main__":
    main()
