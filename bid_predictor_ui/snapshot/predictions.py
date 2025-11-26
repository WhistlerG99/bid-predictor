"""Prediction graph callbacks for the snapshot explorer."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State

from ..data import load_dataset_cached, prepare_prediction_dataframe
from ..feature_config import DEFAULT_UI_FEATURE_CONFIG
from ..formatting import apply_bid_labels, compute_bid_label_map
from ..plotting import build_prediction_plot
from ..predictions import extract_derived_feature_rows, predict


def register_prediction_callbacks(app: Dash) -> None:
    """Register callbacks that trigger snapshot predictions."""

    @app.callback(
        Output("prediction-graph", "figure"),
        Output("prediction-store", "data"),
        Output("prediction-warning", "children"),
        Input("bid-records-store", "data"),
        Input("model-uri-store", "data"),
        State("dataset-path-store", "data"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("upgrade-dropdown", "value"),
        State("snapshot-meta-store", "data"),
        State("feature-config-store", "data"),
    )
    def run_predictions(
        records: Optional[List[Dict[str, object]]],
        model_uri: Optional[str],
        dataset_path: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        snapshot_meta: Optional[Dict[str, object]],
        feature_config: Optional[Dict[str, object]],
    ):
        """Run model predictions for the selected snapshot and format outputs.

        The callback prepares model-ready features, merges historical context
        for plotting, performs two prediction passes (table + chart), and
        packages both the figure and per-bid probabilities for downstream
        components.
        """
        if not records:
            return build_prediction_plot(pd.DataFrame()), {}, ""

        selected_df = prepare_prediction_dataframe(
            records, feature_config=feature_config
        )

        if not model_uri:
            empty_fig = build_prediction_plot(pd.DataFrame())
            return (
                empty_fig,
                {"probabilities": {}, "derived_features": []},
                "Load a model to generate acceptance probabilities.",
            )

        plot_source = pd.DataFrame()
        label_map: Dict[object, int] = {}
        label_column: Optional[str] = None
        if dataset_path and carrier and flight_number and travel_date and upgrade_type:
            try:
                dataset = load_dataset_cached(dataset_path)
                required = {"carrier_code", "flight_number", "travel_date", "upgrade_type"}
                if required.issubset(dataset.columns):
                    travel_date_dt = pd.to_datetime(travel_date).date()
                    mask = (
                        (dataset["carrier_code"] == carrier)
                        & (dataset["flight_number"].astype(str) == str(flight_number))
                        & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
                        & (dataset["upgrade_type"] == upgrade_type)
                    )
                    plot_source = dataset.loc[mask].copy()
                    label_map, label_column = compute_bid_label_map(plot_source)
                    plot_source = apply_bid_labels(plot_source, label_map, label_column)
                    if "Bid #" in plot_source.columns:
                        plot_source = plot_source.sort_values("Bid #")
            except Exception:
                plot_source = pd.DataFrame()

        selected_snapshot = snapshot_meta.get("snapshot") if snapshot_meta else None
        selected_snapshot_value = (
            str(selected_snapshot) if selected_snapshot is not None else None
        )

        if label_map and label_column:
            selected_df = apply_bid_labels(selected_df, label_map, label_column)

        if "snapshot_num" not in selected_df.columns and selected_snapshot_value is not None:
            selected_df["snapshot_num"] = selected_snapshot_value
        elif "snapshot_num" in selected_df.columns:
            selected_df["snapshot_num"] = selected_df["snapshot_num"].astype(str)

        if plot_source.empty:
            combined_df = selected_df.copy()
        else:
            if selected_snapshot_value is not None and "snapshot_num" in plot_source.columns:
                mask = plot_source["snapshot_num"].astype(str) == selected_snapshot_value
                plot_source = plot_source.loc[~mask]
            if "snapshot_num" in plot_source.columns:
                plot_source["snapshot_num"] = plot_source["snapshot_num"].astype(str)
            combined_df = pd.concat([plot_source, selected_df], ignore_index=True, sort=False)

        try:
            plot_pred_df = predict(
                model_uri,
                combined_df.copy(),
                feature_config=feature_config,
                return_transformed=False,
            )
            table_pred_df = predict(
                model_uri,
                selected_df.copy(),
                feature_config=feature_config,
                return_transformed=True,
            )
        except Exception as exc:  # pragma: no cover - user feedback
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Prediction failed: {exc}")
            return empty_fig, {}, str(exc)

        figure = build_prediction_plot(plot_pred_df)
        warning = (
            table_pred_df.attrs.get("model_warning", "")
            or plot_pred_df.attrs.get("model_warning", "")
            or ""
        )

        predictions: Dict[str, Optional[float]] = {}
        for idx, _ in enumerate(records):
            column_id = f"bid_{idx}"
            if idx < len(table_pred_df):
                value = table_pred_df.iloc[idx].get("Acceptance Probability")
                if value is None or pd.isna(value):
                    predictions[column_id] = None
                else:
                    try:
                        predictions[column_id] = round(float(value), 4)
                    except (TypeError, ValueError):
                        predictions[column_id] = value
            else:
                predictions[column_id] = None

        config_for_comp = feature_config or DEFAULT_UI_FEATURE_CONFIG
        comp_features = config_for_comp.get("comp_features", [])
        transformed = table_pred_df.attrs.get("transformed_features")
        derived_rows = extract_derived_feature_rows(transformed, comp_features)
        payload = {
            "probabilities": predictions,
            "derived_features": derived_rows,
        }

        return figure, payload, warning


__all__ = ["register_prediction_callbacks"]
