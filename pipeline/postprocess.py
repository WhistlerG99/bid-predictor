# postprocess.py
import glob
import logging
import os
import re
import pandas as pd

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OFFER_PROB_COLS = [
    "offer_id",
    "partner_id",
    "product_id",
    "carrier_code",
    "flight_number",
    "departure_timestamp",
    "origination_code",
    "destination_code",
    "upgrade_type",
    "accept_prob",
    "accept_prob_timestamp",
]


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
    logger.info(f"Using file_timestamp={timestamp} in final parquet name")

    partner_id = df["partner_id"].iloc[0]
    logger.info(f"Using partner_id={partner_id}")

    # Drop file_timestamp from final parquet
    df = df.drop(columns=["file_timestamp"])

    year, month, day = re.search(r"(\d{4})-(\d{2})-(\d{2})T", timestamp).groups()
    # now = datetime.now()
    # year, month, day = f"{now.year:04d}", f"{now.month:02d}", f"{now.day:02d}"
    
    # Write output to file_name=audit_bid_predictor/
    audit_dir = OUTPUT_DIR + f"/file_name=audit_bid_predictor/partner_id={partner_id}/year={year}/month={month}/day={day}"

    audit_filename = (
        f"{timestamp}-audit_bid_predictor.parquet"
    )
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = os.path.join(audit_dir, audit_filename)

    df.to_parquet(audit_path, index=False)
    logger.info(f"Wrote audit_bid_predictor parquet file to: {audit_path}")

    # Write output to file_name=offer_probabilities/
    offer_prob_dir = OUTPUT_DIR + f"/file_name=offer_probabilities/partner_id={partner_id}/year={year}/month={month}/day={day}"

    offer_prob_filename = (
        f"{timestamp}-offer_probabilities.parquet"
    )
    os.makedirs(offer_prob_dir, exist_ok=True)
    offer_prob_path = os.path.join(offer_prob_dir, offer_prob_filename)

    df[OFFER_PROB_COLS].to_parquet(offer_prob_path, index=False)
    logger.info(f"Wrote offer_probabilities parquet file to: {offer_prob_path}")


if __name__ == "__main__":
    main()
