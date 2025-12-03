"""Prediction helpers for the Dash UI."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .data import get_model_feature_config, load_model_cached
from .feature_config import build_ui_feature_config


def predict(
    model_uri: str,
    df: pd.DataFrame,
    feature_config: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    return_transformed: bool = False,
) -> pd.DataFrame:
    """Generate acceptance probabilities for ``df`` using the cached model.

    The function mirrors the logic used in the training pipeline to ensure the
    UI feeds the model the correct set of features.  It first attempts to use
    the model's stored input schema, then falls back to the UI feature
    configuration when the schema is missing (which happens for older
    serialisations).  Missing feature columns are added as empty entries so the
    CatBoost wrapper can apply its default handling, and a warning is attached
    to the resulting frame via ``df.attrs`` when this corrective behaviour is
    triggered.

    Parameters
    ----------
    model_uri:
        URI pointing to the persisted model artefact understood by
        :func:`bid_predictor_ui.data.load_model_cached`.
    df:
        Feature rows to score; the original frame is modified in-place so
        callers receive the predictions alongside the inputs they passed in.
    feature_config:
        Optional UI-specific feature configuration.  When omitted the helper
        loads metadata embedded in the model artefact and converts it to the UI
        shape with :func:`build_ui_feature_config`.

    Returns
    -------
    pandas.DataFrame
        The same frame that was provided via ``df`` with an additional
        ``"Acceptance Probability"`` column expressed in percentages.  If the
        model required additional columns to be inserted, a human-readable
        message is stored on ``df.attrs["model_warning"]``.
    """

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

    transformed_features: Optional[pd.DataFrame] = None

    if return_transformed:
        transform_and_predict = getattr(model, "_transform_and_predict_proba", None)
        if callable(transform_and_predict):
            probabilities, transformed_features = transform_and_predict(feature_df)
            predictions = probabilities
        else:
            predictions = model.predict_proba(feature_df)
            transform_only = getattr(model, "_transform", None)
            if callable(transform_only):
                transformed_features = transform_only(feature_df)
    else:
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
    if return_transformed and transformed_features is not None:
        if not isinstance(transformed_features, pd.DataFrame):
            columns = getattr(model, "feature_names_in_", None)
            try:
                transformed_features = pd.DataFrame(transformed_features, columns=columns)
            except Exception:
                transformed_features = pd.DataFrame(transformed_features)
        transformed_features = transformed_features.reset_index(drop=True)
        df.attrs["transformed_features"] = transformed_features
    elif "transformed_features" in df.attrs:
        df.attrs.pop("transformed_features", None)
    return df


def extract_derived_feature_rows(
    transformed: Optional[pd.DataFrame],
    comp_features: Sequence[str] | None,
) -> List[Dict[str, object]]:
    """Return the subset of ``transformed`` that corresponds to derived features."""

    if transformed is None or not isinstance(transformed, pd.DataFrame):
        return []
    features = [feature for feature in (comp_features or []) if feature in transformed.columns]
    if not features:
        return []
    subset = transformed.loc[:, features].reset_index(drop=True)
    return subset.to_dict("records")


__all__ = ["extract_derived_feature_rows", "predict"]
