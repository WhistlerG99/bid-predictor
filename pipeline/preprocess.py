# preprocess.py

import os
import glob
import logging
import json
import boto3
import mlflow
import pandas as pd

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

    if MODEL_REGISTRY_STAGE:
        versions = client.get_latest_versions(model_name, stages=[MODEL_REGISTRY_STAGE])
    else:
        versions = client.search_model_versions(f"name='{model_name}'")

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
    df["current_timestamp"] = pd.Timestamp.now("utc").round(freq="s").tz_localize(None) # TODO: change to local time
    # df["current_timestamp"] = (
    #     pd.Timestamp.now("utc").round(freq="s") + pd.to_timedelta(df["utc_diff"], "m")
    # ).dt.tz_localize(None)
    df["file_timestamp"] = timestamp_str

    for c in ["travel_date", "departure_timestamp"]:
        df[c] = pd.to_datetime(df[c])

    logger.info(f"Loaded DataFrame with shape: {df.shape}")

    processed_path = os.path.join(OUTPUT_DIR, "processed.parquet")
    df.to_parquet(processed_path)
    logger.info(f"Wrote processed Parquet to: {processed_path}")


if __name__ == "__main__":
    main()
