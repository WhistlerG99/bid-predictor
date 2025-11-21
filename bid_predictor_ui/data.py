"""Data loading and preparation utilities for the UI package."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import mlflow
import pandas as pd
import pyarrow.dataset as ds
from pyarrow import fs as pyfs

DEFAULT_DATASET_PATH = "./data/air_canada_and_lot/bid_data_snapshots_v2.parquet"
DEFAULT_CATEGORY_COLUMNS: Tuple[str, ...] = (
    "carrier_code",
    "flight_number",
    "fare_class",
)
DEFAULT_RENAME_COLUMNS: Mapping[str, str] = {
    "current_available_seats": "seats_available",
}

_GROUPBY_KEY_FEATURES = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
    "snapshot_num",
]


def _in_jupyter() -> bool:
    """Return ``True`` if the current interpreter is backed by IPython."""

    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _has_any_env(keys) -> bool:
    """Check whether any environment variable in ``keys`` is defined."""

    env = os.environ
    return any(k in env and str(env[k]).strip() != "" for k in keys)


def _any_file_exists(paths) -> bool:
    """Return ``True`` if any of the provided filesystem paths exist."""

    return any(os.path.exists(p) for p in paths)


def detect_execution_environment() -> Tuple[str, str]:
    """Identify whether the code runs locally or inside SageMaker contexts."""

    job_env_keys = [
        "SM_TRAINING_ENV",
        "SM_JOB_NAME",
        "SM_CURRENT_HOST",
        "SM_RESOURCE_CONFIG",
        "SM_INPUT_DATA_CONFIG",
    ]
    job_config_files = [
        "/opt/ml/input/config/hyperparameters.json",
        "/opt/ml/input/config/resourceconfig.json",
        "/opt/ml/input/config/inputdataconfig.json",
        "/opt/ml/config/processingjobconfig.json",
        "/opt/ml/model/.sagemaker-inference",
    ]
    if _has_any_env(job_env_keys) or _any_file_exists(job_config_files):
        return (
            "sagemaker_job",
            "Found SageMaker job config (SM_* env or /opt/ml/*config*.json).",
        )

    notebook_env_keys = [
        "SAGEMAKER_DOMAIN_ID",
        "SAGEMAKER_STUDIO_USER_PROFILE_NAME",
        "SAGEMAKER_INTERNAL_IMAGE_URI",
        "SAGEMAKER_JUPYTERSERVER_IMAGE_URI",
        "JUPYTER_SERVER_NAME",
        "SAGEMAKER_PROJECT_NAME",
    ]
    aws_exec_env = os.environ.get("AWS_EXECUTION_ENV", "")

    if _in_jupyter() and (
        _has_any_env(notebook_env_keys) or "AmazonSageMakerNotebook" in aws_exec_env
    ):
        return (
            "sagemaker_notebook",
            "Running in a Jupyter kernel with SageMaker Studio/Notebook indicators.",
        )

    home = os.path.expanduser("~")
    if (
        _in_jupyter()
        and os.name == "posix"
        and os.path.exists(os.path.join(home, "SageMaker"))
    ):
        return (
            "sagemaker_notebook",
            "Jupyter on Linux with ~/SageMaker suggests a SageMaker Notebook Instance.",
        )

    if (not _in_jupyter()) and (
        _has_any_env(notebook_env_keys)
        or "AmazonSageMakerNotebook" in aws_exec_env
        or (
            os.name == "posix"
            and os.path.exists(os.path.join(home, "SageMaker"))
        )
    ):
        return (
            "sagemaker_terminal",
            "No Jupyter kernel, but SageMaker Studio/Notebook environment markers present (terminal session).",
        )

    return ("local", "No SageMaker job/notebook/terminal markers detected.")


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


def _resolve_filesystem(dataset_path: str, storage_options: Optional[Mapping[str, object]]):
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
        [
            "current_timestamp",
            "departure_timestamp",
            "travel_date",
        ],
    )

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
    """Compatibility wrapper used by CLI entry points."""

    return resolve_dataset_path(arg_value)


@lru_cache(maxsize=4)
def load_dataset_cached(dataset_path: str) -> pd.DataFrame:
    """Load and cache the training dataset for interactive use."""

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


def _coerce_datetime_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    """Convert the specified columns to pandas datetime dtype when present."""

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
    "get_model_feature_config",
    "prepare_prediction_dataframe",
    "resolve_dataset_path",
    "resolve_train_file",
]
