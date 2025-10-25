"""Data loading helpers for the bid predictor tuning utilities."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import pyarrow.dataset as ds

from bid_predictor.utils import detect_execution_environment


def resolve_train_file(arg_value: Optional[str]) -> str:
    """Resolve the training dataset path based on the execution environment."""

    if arg_value:
        return arg_value

    env, _ = detect_execution_environment()
    if env == "sagemaker_job":
        return os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    if env in {"sagemaker_notebook", "sagemaker_terminal"}:
        bucket = os.environ.get("S3_BUCKET_DATA")
        if not bucket:
            raise RuntimeError(
                "S3_BUCKET_DATA environment variable must be set for SageMaker environments"
            )
        return bucket + "/data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
    return "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"


def load_training_data(train_path: str) -> pd.DataFrame:
    """Load the parquet training dataset into a pandas DataFrame."""

    dataset = ds.dataset(train_path, format="parquet")
    table = dataset.to_table()
    data = table.to_pandas()

    for col in ["carrier_code", "flight_number", "fare_class"]:
        if col in data.columns:
            data[col] = data[col].astype("category")

    data = data.rename(columns={"current_available_seats": "seats_available"}, errors="ignore")
    return data
