"""Centralized data loading, caching, and preparation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import mlflow
import pandas as pd
import pyarrow.dataset as ds
from pyarrow import fs as pyfs

from .feature_config import _GROUPBY_KEY_FEATURES, load_feature_config
from .utils import detect_execution_environment


DEFAULT_DATASET_PATH = "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
DEFAULT_CATEGORY_COLUMNS: Tuple[str, ...] = (
    "carrier_code",
    "flight_number",
    "fare_class",
)
DEFAULT_RENAME_COLUMNS: Mapping[str, str] = {
    "current_available_seats": "seats_available",
}


@dataclass(frozen=True)
class DatasetLoadOptions:
    """Options that control how the training dataset is loaded."""

    dataset_path: Optional[str] = None
    storage_options: Optional[Mapping[str, object]] = None
    columns_as_category: Sequence[str] = DEFAULT_CATEGORY_COLUMNS
    rename_columns: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_RENAME_COLUMNS)
    )
    max_departure_offset_days: Optional[int] = None


def resolve_dataset_path(explicit_path: Optional[str] = None) -> str:
    """Resolve the dataset path, taking the execution environment into account."""

    if explicit_path:
        return explicit_path

    env, _ = detect_execution_environment()
    if env == "sagemaker_job":
        return os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    if env in {"sagemaker_notebook", "sagemaker_terminal"}:
        bucket = os.environ.get("S3_BUCKET_DATA")
        if not bucket:
            raise RuntimeError(
                "S3_BUCKET_DATA environment variable must be set for SageMaker environments"
            )
        return os.path.join(
            bucket.rstrip("/"),
            "data/air_canada_and_lot/bid_data_snapshots_v2.parquet",
        )

    return DEFAULT_DATASET_PATH


def _resolve_filesystem(
    dataset_path: str, storage_options: Optional[Mapping[str, object]]
):
    """Infer the filesystem implementation that should back the dataset path."""

    if dataset_path.startswith("s3://"):
        options = storage_options or {}
        return pyfs.S3FileSystem(**options)
    return None


def load_training_data(
    dataset_path: str,
    *,
    storage_options: Optional[Mapping[str, object]] = None,
    columns_as_category: Sequence[str] = DEFAULT_CATEGORY_COLUMNS,
    rename_columns: Mapping[str, str] = DEFAULT_RENAME_COLUMNS,
    max_departure_offset_days: Optional[int] = None,
) -> pd.DataFrame:
    """Load the parquet dataset into a pandas DataFrame.

    Args:
        dataset_path: Location of the parquet dataset (local path or S3 URI).
        storage_options: Optional filesystem options (for example S3 credentials).
        columns_as_category: Columns to convert to categorical dtype.
        rename_columns: Column rename mapping applied after loading.
        max_departure_offset_days: If provided, filter out rows where the
            difference between ``departure_timestamp`` and ``current_timestamp``
            is greater than the specified number of days.
    """

    filesystem = _resolve_filesystem(dataset_path, storage_options)
    dataset = ds.dataset(dataset_path, format="parquet", filesystem=filesystem)
    table = dataset.to_table()
    data = table.to_pandas()

    data = data.rename(columns=rename_columns, errors="ignore")

    for column in columns_as_category:
        if column in data.columns:
            data[column] = data[column].astype("category")

    _coerce_datetime_columns(data, [
        "current_timestamp",
        "departure_timestamp",
        "travel_date",
    ])

    if max_departure_offset_days is not None:
        cutoff = pd.to_timedelta(max_departure_offset_days, unit="D")
        if {"departure_timestamp", "current_timestamp"}.issubset(data.columns):
            delta = data["departure_timestamp"] - data["current_timestamp"]
            data = data.loc[delta < cutoff].copy()

    return data


def load_dataset(options: DatasetLoadOptions | None = None) -> pd.DataFrame:
    """Load the dataset based on ``DatasetLoadOptions`` settings."""

    opts = options or DatasetLoadOptions()
    dataset_path = resolve_dataset_path(opts.dataset_path)
    return load_training_data(
        dataset_path,
        storage_options=opts.storage_options,
        columns_as_category=opts.columns_as_category,
        rename_columns=opts.rename_columns,
        max_departure_offset_days=opts.max_departure_offset_days,
    )


def resolve_train_file(arg_value: Optional[str]) -> str:
    """Compatibility wrapper used by CLI entry points.

    Historically tuning utilities imported :func:`resolve_train_file` from
    ``bid_predictor.tuning.data_access``. The helper now simply proxies to the
    centralized :func:`resolve_dataset_path` implementation.
    """

    return resolve_dataset_path(arg_value)


@lru_cache(maxsize=4)
def load_dataset_cached(dataset_path: str) -> pd.DataFrame:
    """Load and cache the training dataset for interactive use.

    The cached dataset always applies the five day departure window filter to
    match the Dash UI requirements.
    """

    data = load_training_data(
        dataset_path,
        max_departure_offset_days=5,
    )

    missing = [col for col in _GROUPBY_KEY_FEATURES if col not in data.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: {}".format(
                ", ".join(sorted(missing))
            )
        )

    _coerce_datetime_columns(
        data,
        ["current_timestamp", "departure_timestamp", "travel_date"],
    )

    return data


@lru_cache(maxsize=4)
def load_model_cached(model_uri: str):
    """Load and cache an MLflow model by URI."""

    return mlflow.sklearn.load_model(model_uri)


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """Return the feature and categorical column lists from the feature config."""

    feature_config = load_feature_config()
    features = list(feature_config["pre_features"])
    categorical = list(feature_config["cat_features"])
    return features, categorical


def prepare_prediction_dataframe(
    table_records: Iterable[Dict[str, object]]
) -> pd.DataFrame:
    """Convert edited table records back into a feature dataframe."""

    df = pd.DataFrame(list(table_records))
    if df.empty:
        return df

    for column in df.columns:
        if column in {
            "carrier_code",
            "flight_number",
            "fare_class",
            "offer_status",
            "upgrade_type",
            "Bid #",
        }:
            continue
        if df[column].isnull().all():
            continue

        try:
            df[column] = pd.to_numeric(df[column])
            continue
        except (TypeError, ValueError):
            pass

        try:
            df[column] = pd.to_datetime(df[column])
        except (TypeError, ValueError):
            continue

    _coerce_datetime_columns(
        df,
        ["travel_date", "current_timestamp", "departure_timestamp"],
    )

    features, _ = get_feature_columns()
    feature_df = df.reindex(columns=features)
    extra_columns = [col for col in df.columns if col not in feature_df.columns]
    for column in extra_columns:
        feature_df[column] = df[column]

    return feature_df


def prepare_features(
    data: pd.DataFrame,
    pre_features: Sequence[str],
    *,
    testing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepare features for training or evaluation splits."""

    data = data.copy()
    _coerce_datetime_columns(
        data,
        ["current_timestamp", "departure_timestamp"],
    )
    if {"departure_timestamp", "current_timestamp"}.issubset(data.columns):
        delta = data["departure_timestamp"] - data["current_timestamp"]
        data = data.loc[delta < pd.to_timedelta("5d")].copy()

    data = data.sort_values(["travel_date", "carrier_code", "flight_number"]).reset_index(
        drop=True
    )

    available_pre_features = [feature for feature in pre_features if feature in data.columns]
    selection_columns: List[str] = list(
        dict.fromkeys(available_pre_features + ["offer_status", "id", "decision_timestamp"])
    )

    if testing:
        cutoff = "2023-08-01"
        yX_test = data[
            (data.travel_date >= cutoff) & (data.travel_date <= "2023-08-15")
        ][selection_columns].copy()
    else:
        cutoff = "2025-05-01"
        yX_test = data.loc[data.travel_date >= cutoff, selection_columns].copy()

    yX_train = data.loc[data.travel_date < cutoff, selection_columns].copy()

    X_train = yX_train.loc[:, available_pre_features]
    X_test = yX_test.loc[:, available_pre_features]
    y_train = (yX_train["offer_status"] == "TICKETED").astype(int)
    y_test = (yX_test["offer_status"] == "TICKETED").astype(int)

    yX_test.loc[:, "offer_status"] = "Rejected"
    yX_test.loc[y_test == 1, "offer_status"] = "Accepted"

    yX_test = yX_test.set_index(_GROUPBY_KEY_FEATURES)
    yX_test["Bid #"] = (
        yX_test.groupby(level=_GROUPBY_KEY_FEATURES[:-1], observed=True)["id"]
        .transform(lambda s: pd.factorize(s)[0] + 1)
        .astype(int)
        .apply(lambda n: f"Bid {n}")
    )

    return X_train, X_test, y_train, y_test, yX_test


def _coerce_datetime_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column])


__all__ = [
    "DatasetLoadOptions",
    "DEFAULT_DATASET_PATH",
    "load_training_data",
    "load_dataset",
    "load_dataset_cached",
    "load_model_cached",
    "get_feature_columns",
    "prepare_prediction_dataframe",
    "prepare_features",
    "resolve_dataset_path",
    "resolve_train_file",
]

