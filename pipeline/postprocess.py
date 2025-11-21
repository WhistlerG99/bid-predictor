# postprocess.py
import os
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
        glob.glob(os.path.join(INPUT_DIR, "*.parquet.out")) +
        glob.glob(os.path.join(INPUT_DIR, "*.out"))
    )

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

    timestamp = df["file_timestamp"].iloc[0]
    logger.info(f"Using file_timestamp={timestamp} for final CSV name")

    # Optional: drop file_timestamp from final CSV if you don't want it in output
    df = df.drop(columns=["file_timestamp"])

    output_filename = f"availability-offers-probability-{timestamp}.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    df.to_csv(output_path, index=False)
    logger.info(f"Wrote final CSV to: {output_path}")


if __name__ == "__main__":
    main()