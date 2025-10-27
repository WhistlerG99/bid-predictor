"""Prediction utilities for the Dash UI."""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .data_loading import _get_feature_columns, _load_model_cached


def _prepare_prediction_dataframe(table_records: Iterable[dict]) -> pd.DataFrame:
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
    if "travel_date" in df.columns:
        df["travel_date"] = pd.to_datetime(df["travel_date"])
    if "current_timestamp" in df.columns:
        df["current_timestamp"] = pd.to_datetime(df["current_timestamp"])
    if "departure_timestamp" in df.columns:
        df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
    features, _ = _get_feature_columns()
    feature_df = df.reindex(columns=features)
    extra_columns = [col for col in df.columns if col not in feature_df.columns]
    for column in extra_columns:
        feature_df[column] = df[column]
    return feature_df


def _predict(model_uri: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    model = _load_model_cached(model_uri)
    feature_df = df.copy()
    model_warning: Optional[str] = None

    expected_columns: Optional[List[str]] = None
    try:
        metadata = getattr(model, "metadata", None)
        if metadata is not None:
            input_schema = metadata.get_input_schema()
            if input_schema is not None:
                names = list(input_schema.input_names())
                expected_columns = names or None
    except AttributeError:
        expected_columns = None

    if expected_columns:
        missing = [col for col in expected_columns if col not in feature_df.columns]
        if missing:
            model_warning = (
                "Added missing model columns with empty values: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        feature_df = feature_df.reindex(columns=expected_columns)
    else:
        features, _ = _get_feature_columns()
        missing = [col for col in features if col not in feature_df.columns]
        if missing:
            model_warning = (
                "Added missing feature config columns with empty values: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        feature_df = feature_df.reindex(columns=features)

    predictions = model.predict_proba(feature_df)
    if isinstance(predictions, pd.DataFrame) and "Acceptance Probability" in predictions.columns:
        acceptance = predictions["Acceptance Probability"].astype(float).to_numpy()
    else:
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            acceptance = predictions[:, 1]
        else:
            acceptance = predictions
    acceptance_series = pd.Series(acceptance, index=df.index, dtype="float64") * 100.0
    df["Acceptance Probability"] = acceptance_series.round(4)
    if model_warning:
        df.attrs["model_warning"] = model_warning
    return df
