# preprocess.py

import os
import glob
import logging
import pandas as pd

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_parquet_folder(base_dir: str):
    """
    Example directory structure delivered by ProcessingInput:
        /opt/ml/processing/input/history/
            availability-offers-data-20251113T183618/
            availability-offers-data-20251117T142354/
            availability-offers-data-20251118T142322/
    We pick the latest folder lexicographically (timestamp included).
    """
    folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ]
    if not folders:
        raise FileNotFoundError("No subfolders found under /input/history")
    # folders are timestamped, so max() gives the newest
    latest = max(folders)
    full_path = os.path.join(base_dir, latest)
    print(f"Using newest historical folder: {latest}")
    return full_path


def main():
    # Find parquet files in the input directory
    parquet_files = glob.glob(os.path.join(INPUT_DIR, "*.parquet"))

    logger.info(f"Reading Parquet file from: {parquet_files}")

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {INPUT_DIR}")

    # pick the latest (lexicographically) parquet file
    parquet_file = max(parquet_files)

    logger.info(f"Reading parquet file: {parquet_file}")

    # derive timestamp from input parquet name
    basename = os.path.basename(parquet_file)
    # Expected: availability-offers-YYYY-mm-ddTHH-MM-SS.parquet
    prefix = "availability-offers-"
    suffix = ".parquet"
    if not (basename.startswith(prefix) and basename.endswith(suffix)):
        raise ValueError(
            f"Unexpected parquet filename format: {basename}"
        )
    timestamp_str = basename[len(prefix):-len(suffix)]
    logger.info(f"Parsed file timestamp: {timestamp_str}")

    # Read parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    df = df.rename(columns={"travel_dt": "travel_date"})

    df["offer_time"] = df.apply(
        lambda x: (x["departure_timestamp"] - x["created_timestamp"]).total_seconds()
        / (60 * 60 * 24),
        axis=1,
    )
    df["snapshot_num"] = 1
    df["current_timestamp"] = pd.Timestamp.now().round(freq="s")

    # store original file timestamp so final step can name the CSV
    df["file_timestamp"] = timestamp_str

    for c in ["travel_date", "departure_timestamp"]:
        df[c] = pd.to_datetime(df[c])

    logger.info(f"Loaded DataFrame with shape: {df.shape}")

    # Make sure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # write out as Parquet: processed.parquet
    output_path = os.path.join(OUTPUT_DIR, "processed.parquet")
    df.to_parquet(output_path)
    logger.info(f"Wrote processed Parquet to: {output_path}")  # CHANGED (message now accurate)


if __name__ == "__main__":
    main()