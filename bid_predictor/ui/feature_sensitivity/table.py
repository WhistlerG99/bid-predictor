"""Scenario table rendering and editing callbacks."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from dash import Dash, Input, Output, State, no_update

from ...data import prepare_prediction_dataframe
from ..predictions import predict
from ..scenario import ScenarioFeature, resolve_locked_cells
from ..tables import apply_table_edits, build_bid_table


def register_table_callbacks(app: Dash) -> None:
    """Register callbacks that manage the scenario bid table."""

    @app.callback(
        Output("scenario-bid-table", "columns"),
        Output("scenario-bid-table", "data"),
        Output("scenario-bid-table", "style_data_conditional"),
        Input("scenario-records-store", "data"),
        Input("model-uri-store", "data"),
        Input("scenario-feature-dropdown", "value"),
        State("feature-config-store", "data"),
    )
    def render_scenario_table(
        records: Optional[List[Dict[str, object]]],
        model_uri: Optional[str],
        feature_value: Optional[str],
        feature_config: Optional[Dict[str, object]],
    ):
        predictions: Dict[str, object] = {}
        if records and model_uri:
            feature_df = prepare_prediction_dataframe(
                records, feature_config=feature_config
            )
            if not feature_df.empty:
                try:
                    pred_df = predict(
                        model_uri,
                        feature_df.copy(),
                        feature_config=feature_config,
                    )
                except Exception:
                    pred_df = pd.DataFrame()
                if not pred_df.empty:
                    for idx, _ in enumerate(records):
                        column_id = f"bid_{idx}"
                        if idx < len(pred_df):
                            value = pred_df.iloc[idx].get("Acceptance Probability")
                            if value is None or pd.isna(value):
                                predictions[column_id] = None
                            else:
                                try:
                                    predictions[column_id] = float(value)
                                except (TypeError, ValueError):
                                    predictions[column_id] = value
        decoded_feature = ScenarioFeature.decode(feature_value)
        locked_cells = resolve_locked_cells(records, decoded_feature)

        if locked_cells:
            return build_bid_table(
                records,
                predictions,
                feature_config=feature_config,
                locked_cells=locked_cells,
            )
        return build_bid_table(records, predictions, feature_config=feature_config)

    @app.callback(
        Output("scenario-bid-delete-selector", "options"),
        Output("scenario-bid-delete-selector", "value"),
        Input("scenario-records-store", "data"),
    )
    def sync_scenario_delete_selector(records: Optional[List[Dict[str, object]]]):
        if not records:
            return [], []
        options = [
            {
                "label": f"Bid {record.get('Bid #') or record.get('bid_number') or idx + 1}",
                "value": idx,
            }
            for idx, record in enumerate(records)
        ]
        return options, []

    @app.callback(
        Output("scenario-bid-restore-selector", "options"),
        Output("scenario-bid-restore-selector", "value"),
        Input("scenario-removed-bids-store", "data"),
    )
    def sync_scenario_restore_selector(
        removed: Optional[List[Dict[str, object]]]
    ):
        if not removed:
            return [], []
        options = [
            {"label": item.get("label") or f"Removed bid {idx + 1}", "value": item.get("id")}
            for idx, item in enumerate(removed)
            if item.get("id") is not None
        ]
        return options, []

    @app.callback(
        Output("scenario-records-store", "data", allow_duplicate=True),
        Input("scenario-bid-table", "data_timestamp"),
        State("scenario-bid-table", "data"),
        State("scenario-bid-table", "columns"),
        State("scenario-records-store", "data"),
        State("scenario-feature-dropdown", "value"),
        State("feature-config-store", "data"),
        prevent_initial_call=True,
    )
    def persist_scenario_table_edits(
        data_timestamp: Optional[int],
        table_data: Optional[List[Dict[str, object]]],
        columns: Optional[List[Dict[str, object]]],
        records: Optional[List[Dict[str, object]]],
        feature_value: Optional[str],
        feature_config: Optional[Dict[str, object]],
    ):
        if not data_timestamp or not table_data or not columns or not records:
            return no_update

        decoded_feature = ScenarioFeature.decode(feature_value)
        locked_cells = resolve_locked_cells(records, decoded_feature)

        updated_records = apply_table_edits(
            records,
            table_data,
            columns,
            locked_cells=locked_cells if locked_cells else None,
            feature_config=feature_config,
        )
        if updated_records is None:
            return no_update
        return updated_records


__all__ = ["register_table_callbacks"]
