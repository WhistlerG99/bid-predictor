# preprocess.py

import os
import glob
import logging
import json
import boto3
import pandas as pd

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"
MODEL_CONFIG_DIR = (
    "/opt/ml/processing/model_config"  # CHANGED: new directory for model_config.json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These will be passed via env from pipeline_setup.py
MODEL_BUCKET = os.getenv(
    "MODEL_BUCKET", "amazon-sagemaker-622055002283-us-east-1-b37b41a56cd8"
)
MODEL_BASE_PREFIX = os.getenv(
    "MODEL_BASE_PREFIX", "dzd_4dt0rvdnr1hoiv/5vt5uv9jpcqmxz/dev"
)  # e.g. "dzd_.../dev"

MODEL_NAME_PREFIX = os.getenv(
    "MODEL_NAME_PREFIX", "bid-predictor"
)  # e.g. "bid-predictor"

def select_model_for_carrier(carrier_code: str) -> str:
    """
    List models in S3 under:
      <MODEL_BASE_PREFIX>/bid-predictor-<carrier>-YYYY-mm-dd-HH-MM-SS/model.tar.gz
    and return the S3 URI of the most recent one.
    """
    if not MODEL_BUCKET or not MODEL_BASE_PREFIX or not MODEL_NAME_PREFIX:
        raise RuntimeError("MODEL_BUCKET, MODEL_BASE_PREFIX or MODEL_NAME_PREFIX not set")

    s3 = boto3.client("s3")

    # keys look like:
    #   <MODEL_BASE_PREFIX>/bid-predictor-<carrier>-YYYY-mm-dd-HH-MM-SS/model.tar.gz
    prefix = f"{MODEL_BASE_PREFIX}/{MODEL_NAME_PREFIX}-{carrier_code}-"
    logger.info(f"Listing models with prefix: s3://{MODEL_BUCKET}/{prefix}")

    resp = s3.list_objects_v2(Bucket=MODEL_BUCKET, Prefix=prefix)
    contents = resp.get("Contents", [])

    candidates = []
    for obj in contents:
        key = obj["Key"]
        if not key.endswith("/model.tar.gz"):
            continue

        # Extract timestamp from the directory name
        # key: .../<prefix>-<carrier>-YYYY-mm-dd-HH-MM-SS/model.tar.gz
        model_dir = key.rsplit("/", 1)[0].split("/")[-2]
        # model_dir = "<prefix>-<carrier>-YYYY-mm-dd-HH-MM-SS"
        prefix_str = f"{MODEL_NAME_PREFIX}-{carrier_code}-"
        if not model_dir.startswith(prefix_str):
            continue
        ts_str = model_dir[len(prefix_str) :]  # "YYYY-mm-dd-HH-MM-SS"

        candidates.append((ts_str, key))

    if not candidates:
        raise RuntimeError(f"No model candidates found for carrier {carrier_code}")

    # Your timestamp format is lexicographically sortable
    best_ts, best_key = max(candidates, key=lambda x: x[0])
    model_data = f"s3://{MODEL_BUCKET}/{best_key}"

    logger.info(f"Selected model for {carrier_code}: {model_data}")
    return model_data


def main():
    parquet_files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))
    logger.info(f"Reading Parquet file from: {parquet_files}")

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    parquet_file = max(parquet_files)
    logger.info(f"Reading parquet file: {parquet_file}")

    basename = os.path.basename(parquet_file)
    prefix = "availability-offers-"
    suffix = ".parquet"
    if not (basename.startswith(prefix) and basename.endswith(suffix)):
        raise ValueError(f"Unexpected parquet filename format: {basename}")
    timestamp_str = basename[len(prefix) : -len(suffix)]
    logger.info(f"Parsed file timestamp: {timestamp_str}")

    df = pd.read_parquet(parquet_file)

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
