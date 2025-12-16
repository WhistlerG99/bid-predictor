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


def read_high_water_mark(path):
    parsed = urlparse(path)
    try:
        resp = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        hwm = resp["Body"].read().decode("utf-8").strip()
        print(f"Read HWM: {hwm}")
        return hwm
    except s3.exceptions.NoSuchKey:
        print("No HWM found. First run.")
        return None


def write_high_water_mark(path, ts):
    parsed = urlparse(path)
    s3.put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=ts)
    print(f"Updated HWM to: {ts}")


def list_csv_files(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    results = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".csv"):
                    results.append(obj["Key"])
    print(f"Listed {len(results)} CSV files under s3://{bucket}/{prefix}")
    return results


def extract_ts_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", filename)
    if not match:
        print(f"Warning: Could not extract timestamp from filename {filename}")
        return None
    raw_ts = match.group(1)
    ts = raw_ts.replace("-", ":", 2).replace("-", ":", 1)
    return ts


def read_csv_from_s3(bucket, key):
    s3_object = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(s3_object["Body"].read()), dtype=str, low_memory=False)
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})
    df = df[[col for col in FINAL_COLUMNS if col in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Read {len(df)} rows from s3://{bucket}/{key}")
    return df


def overwrite_the_csv(df, output_csv_path):
    df = df.loc[:, ~df.columns.duplicated()]

    if "offer_id" in df.columns:
        df = df.drop_duplicates(subset=["offer_id"], keep="last")

    print(f"Writing {len(df)} deduplicated rows (overwrite mode).")

    parsed = urlparse(output_csv_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue()
    )

    print(f"CSV overwritten at s3://{bucket}/{key}")


def read_lookup_carriers(path):
    parsed = urlparse(path)
    s3_object = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    df_lookup = pd.read_csv(io.BytesIO(s3_object['Body'].read()), dtype=str)
    if "operating_carrier" not in df_lookup.columns:
        raise ValueError("Lookup CSV does not contain 'operating_carrier' column")
    carriers = df_lookup["operating_carrier"].dropna().unique().tolist()
    print(f"Found {len(carriers)} operating carriers in lookup: {carriers}")
    return carriers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--hwm_path", required=True)
    parser.add_argument("--lookup_path", required=True)
    parser.add_argument("--input_prefix", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    s3_bucket = args.s3_bucket
    hwm_path = args.hwm_path
    lookup_path = args.lookup_path
    input_prefix = args.input_prefix
    output_csv = args.output_csv

    operating_carriers = read_lookup_carriers(lookup_path)

    # Read HWM ONCE
    last_hwm = read_high_water_mark(hwm_path)

    all_carriers_df = pd.DataFrame()
    all_new_timestamps = []

    for carrier in operating_carriers:
        print(f"\nProcessing carrier: {carrier}")

        # s3_root_prefix = f"dzd_4dt0rvdnr1hoiv/dfbsxtgjets9wn/offer_probability_csv/{carrier}/"
        s3_root_prefix = f"{input_prefix}/{carrier}/"

        all_files = list_csv_files(s3_bucket, s3_root_prefix)

        new_files = []
        new_timestamps = []

        for key in all_files:
            ts = extract_ts_from_filename(key)
            if ts and (last_hwm is None or ts > last_hwm):
                new_files.append(key)
                new_timestamps.append(ts)

        print(f"Found {len(new_files)} new CSV files to process for carrier {carrier}.")

        if not new_files:
            print(f"No new files to process for carrier {carrier}.")
            continue

        df_carrier = pd.DataFrame()
        for key in new_files:
            df_part = read_csv_from_s3(s3_bucket, key)
            df_carrier = pd.concat([df_carrier, df_part], ignore_index=True)

        all_carriers_df = pd.concat([all_carriers_df, df_carrier], ignore_index=True)
        all_new_timestamps.extend(new_timestamps)

    if not all_carriers_df.empty:
        overwrite_the_csv(all_carriers_df, output_csv)

    if all_new_timestamps:
        final_hwm = max(all_new_timestamps)
        write_high_water_mark(hwm_path, final_hwm)

    print("\nProcessing finished for all carriers.")


if __name__ == "__main__":
    main()
