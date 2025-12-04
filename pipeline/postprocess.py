# postprocess.py
import os
import re
# from datetime import datetime
import glob
import logging
import pandas as pd

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # SageMaker Batch Transform writes *.parquet.out (and sometimes just *.out),
    # so we look for all of those.
    parquet_files = (
        glob.glob(os.path.join(INPUT_DIR, "*.parquet")) +
        glob.glob(os.path.join(INPUT_DIR, "*.parquet.out"))
    )
    parquet_files = list(set(parquet_files))

    logger.info(f"Found possible batch output files: {parquet_files}")

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet-like files found in {INPUT_DIR}. "
            f"Looked for *.parquet, *.parquet.out, *.out"
        )

    dfs = []
    for path in parquet_files:
        logger.info(f"Reading batch output file as Parquet: {path}")
        try:
            dfs.append(pd.read_parquet(path))
        except Exception as e:
            logger.error(f"Failed to read {path} as parquet: {e}")
            raise

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined batch output shape: {df.shape}")

    if "file_timestamp" not in df.columns:
        raise KeyError(
            "Expected 'file_timestamp' column not found. "
            "Check preprocess.py output."
        )

    for c in ["departure_timestamp", "created_timestamp", "accept_prob_timestamp"]:
        df[c] = pd.to_datetime(df[c]).dt.round("s").astype("datetime64[ms]")

    timestamp = df["file_timestamp"].iloc[0]
    # carrier_code = df["carrier_code"].iloc[0]

    logger.info(f"Using file_timestamp={timestamp} for final parquet name")

    year, month, day = re.search(r"(\d{4})-(\d{2})-(\d{2})T", timestamp).groups()
    # now = datetime.now()
    # year, month, day = f"{now.year:04d}", f"{now.month:02d}", f"{now.day:02d}"

    # output_dir = OUTPUT_DIR + f"/year={year}/month={month}/day={day}/" + carrier_code
    output_dir = OUTPUT_DIR + f"/year={year}/month={month}/day={day}"

    # Optional: drop file_timestamp from final parquet if you don't want it in output
    df = df.drop(columns=["file_timestamp"])

    output_filename = f"{timestamp}-audit_bid_predictor.parquet"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote final parquet to: {output_path}")


if __name__ == "__main__":
    main()