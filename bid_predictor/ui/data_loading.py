"""Data loading helpers for the Dash UI."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import mlflow
import pandas as pd

from ..feature_config import _GROUPBY_KEY_FEATURES, load_feature_config
from ..tuning.data_access import load_training_data


@lru_cache(maxsize=4)
def _load_dataset_cached(path: str) -> pd.DataFrame:
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
def _load_model_cached(model_uri: str):
    """Load and cache the MLflow model given a model URI."""

    return mlflow.sklearn.load_model(model_uri)


def _get_feature_columns() -> Tuple[List[str], List[str]]:
    feature_config = load_feature_config()
    features = list(feature_config["pre_features"])
    categorical = list(feature_config["cat_features"])
    return features, categorical
