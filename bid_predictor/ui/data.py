"""Data loading helpers for the Dash UI."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import mlflow
import pandas as pd

from ..feature_config import _GROUPBY_KEY_FEATURES, load_feature_config
from ..tuning.data_access import load_training_data


@lru_cache(maxsize=4)
def load_dataset_cached(path: str) -> pd.DataFrame:
    """Load the training dataset and cache it for repeated access."""

    df = load_training_data(path)
    missing = [col for col in _GROUPBY_KEY_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: {}".format(
                ", ".join(sorted(missing))
            )
        )

    for col in ("current_timestamp", "departure_timestamp", "travel_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


@lru_cache(maxsize=4)
def load_model_cached(model_uri: str):
    """Load and cache the MLflow model given a model URI."""

    return mlflow.sklearn.load_model(model_uri)


def get_feature_columns() -> Tuple[List[str], List[str]]:
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

    for col in df.columns:
        if col in {
            "carrier_code",
            "flight_number",
            "fare_class",
            "offer_status",
            "upgrade_type",
            "Bid #",
        }:
            continue
        if df[col].isnull().all():
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            try:
                df[col] = pd.to_datetime(df[col])
            except (TypeError, ValueError):
                pass

    for column in (
        "travel_date",
        "current_timestamp",
        "departure_timestamp",
    ):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])

    features, _ = get_feature_columns()
    feature_df = df.reindex(columns=features)
    extra_columns = [col for col in df.columns if col not in feature_df.columns]
    for column in extra_columns:
        feature_df[column] = df[column]
    return feature_df


__all__ = [
    "get_feature_columns",
    "load_dataset_cached",
    "load_model_cached",
    "prepare_prediction_dataframe",
]
