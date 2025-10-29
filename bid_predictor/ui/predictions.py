"""Prediction helpers for the Dash UI."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd

from ..data import get_model_feature_config, load_model_cached
from .feature_config import build_ui_feature_config


def predict(
    model_uri: str,
    df: pd.DataFrame,
    feature_config: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    model = load_model_cached(model_uri)
    feature_df = df.copy()
    model_warning: Optional[str] = None

    expected_columns: Optional[list[str]] = None
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
        ui_config = feature_config
        if ui_config is None:
            raw_config = get_model_feature_config(model_uri)
            ui_config = build_ui_feature_config(raw_config)
        features = list(ui_config.get("pre_features", []))
        missing: list[str] = []
        if features:
            missing = [col for col in features if col not in feature_df.columns]
            if missing:
                model_warning = (
                    "Added missing feature config columns with empty values: {}".format(
                        ", ".join(sorted(missing))
                    )
                )
            feature_df = feature_df.reindex(columns=features)
        else:
            features = list(feature_df.columns)

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


__all__ = ["predict"]
