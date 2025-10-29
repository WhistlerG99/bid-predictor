"""Bid table rendering and editing callbacks for the snapshot explorer."""
from __future__ import annotations

from typing import Dict, List, Optional

from dash import Dash, Input, Output, State, no_update

from ..tables import apply_table_edits, build_bid_table


def register_table_callbacks(app: Dash) -> None:
    """Register callbacks that manage the snapshot bid table."""

    @app.callback(
        Output("bid-table", "columns"),
        Output("bid-table", "data"),
        Output("bid-table", "style_data_conditional"),
        Input("bid-records-store", "data"),
        Input("prediction-store", "data"),
        State("feature-config-store", "data"),
    )
    def render_bid_table(
        records: Optional[List[Dict[str, object]]],
        predictions: Optional[Dict[str, float]],
        feature_config: Optional[Dict[str, object]],
    ):
        return build_bid_table(
            records,
            predictions,
            feature_config=feature_config,
        )

    @app.callback(
        Output("bid-delete-selector", "options"),
        Output("bid-delete-selector", "value"),
        Input("bid-records-store", "data"),
    )
    def sync_delete_selector(records: Optional[List[Dict[str, object]]]):
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
        Output("bid-restore-selector", "options"),
        Output("bid-restore-selector", "value"),
        Input("removed-bids-store", "data"),
    )
    def sync_restore_selector(removed: Optional[List[Dict[str, object]]]):
        if not removed:
            return [], []
        options = [
            {"label": item.get("label") or f"Removed bid {idx + 1}", "value": item.get("id")}
            for idx, item in enumerate(removed)
            if item.get("id") is not None
        ]
        return options, []

    @app.callback(
        Output("bid-records-store", "data", allow_duplicate=True),
        Input("bid-table", "data_timestamp"),
        State("bid-table", "data"),
        State("bid-table", "columns"),
        State("bid-records-store", "data"),
        State("feature-config-store", "data"),
        prevent_initial_call=True,
    )
    def persist_table_edits(
        data_timestamp: Optional[int],
        table_data: Optional[List[Dict[str, object]]],
        columns: Optional[List[Dict[str, object]]],
        records: Optional[List[Dict[str, object]]],
        feature_config: Optional[Dict[str, object]],
    ):
        if not data_timestamp or not table_data or not columns or not records:
            return no_update

        updated_records = apply_table_edits(
            records,
            table_data,
            columns,
            feature_config=feature_config,
        )
        if updated_records is None:
            return no_update
        return updated_records


__all__ = ["register_table_callbacks"]
