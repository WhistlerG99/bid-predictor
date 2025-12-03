"""Standalone data helpers for the UI package."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import mlflow
import pandas as pd
import pyarrow.dataset as ds
from pyarrow import fs as pyfs

_DEFAULT_GROUPBY_KEYS: Tuple[str, ...] = (
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
    "snapshot_num",
)

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


def _resolve_filesystem(dataset_path: str, storage_options: Optional[Mapping[str, object]]):
    """Infer the filesystem implementation that should back the dataset path."""

    if dataset_path.startswith("s3://"):
        options = storage_options or {}
        return pyfs.S3FileSystem(**options)
    return None


def _coerce_datetime_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    """Convert the specified columns to pandas datetime dtype when present."""

    for column in columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column])


def load_training_data(
    dataset_path: str,
    *,
    storage_options: Optional[Mapping[str, object]] = None,
    columns_as_category: Sequence[str] = DEFAULT_CATEGORY_COLUMNS,
    rename_columns: Mapping[str, str] = DEFAULT_RENAME_COLUMNS,
    max_departure_offset_days: Optional[int] = None,
) -> pd.DataFrame:
    """Load the parquet dataset into a pandas DataFrame."""

    filesystem = _resolve_filesystem(dataset_path, storage_options)
    dataset = ds.dataset(dataset_path.replace("s3://", ""), format="parquet", filesystem=filesystem)
    table = dataset.to_table()
    data = table.to_pandas()

    data = data.rename(columns=rename_columns, errors="ignore")

    for column in columns_as_category:
        if column in data.columns:
            data[column] = data[column].astype("category")

    _coerce_datetime_columns(
        data,
        ["current_timestamp", "departure_timestamp", "travel_date"],
    )

    if max_departure_offset_days is not None:
        cutoff = pd.to_timedelta(max_departure_offset_days, unit="D")
        if {"departure_timestamp", "current_timestamp"}.issubset(data.columns):
            delta = data["departure_timestamp"] - data["current_timestamp"]
            data = data.loc[delta < cutoff].copy()

    return data


def resolve_dataset_path(explicit_path: Optional[str] = None) -> str:
    """Return ``explicit_path`` or the packaged default."""

    if explicit_path:
        return explicit_path
    return "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"


@lru_cache(maxsize=4)
def load_dataset_cached(dataset_path: str) -> pd.DataFrame:
    """Load and cache the training dataset for interactive use."""

    resolved_path = resolve_dataset_path(dataset_path)
    data = load_training_data(
        resolved_path,
        max_departure_offset_days=5,
    )

    missing = [col for col in _DEFAULT_GROUPBY_KEYS if col not in data.columns]
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


def get_model_feature_config(model_uri: str) -> Optional[Mapping[str, object]]:
    """Return the raw feature configuration stored on the cached model."""

    model = load_model_cached(model_uri)
    for attr in ("feature_config_", "feature_config"):
        config = getattr(model, attr, None)
        if config:
            return copy.deepcopy(config)
    return None


def prepare_prediction_dataframe(
    table_records: Iterable[Dict[str, object]],
    feature_config: Optional[Mapping[str, Sequence[str]]] = None,
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

    features: Sequence[str] = ()
    if feature_config is not None:
        features = feature_config.get("pre_features", []) or ()

    if features:
        feature_df = df.reindex(columns=list(features))
    else:
        feature_df = df.copy()
    extra_columns = [col for col in df.columns if col not in feature_df.columns]
    for column in extra_columns:
        feature_df[column] = df[column]

    return feature_df


__all__ = [
    "DatasetLoadOptions",
    "DEFAULT_CATEGORY_COLUMNS",
    "DEFAULT_RENAME_COLUMNS",
    "load_training_data",
    "resolve_dataset_path",
    "load_dataset_cached",
    "load_model_cached",
    "get_model_feature_config",
    "prepare_prediction_dataframe",
]
