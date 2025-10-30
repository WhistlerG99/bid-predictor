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
        predictions: Optional[Dict[str, object]],
        feature_config: Optional[Dict[str, object]],
    ):
        """Render the snapshot bid table with model predictions if available.

        The table view is shared with the scenario tab; we simply forward the
        stored records and per-bid probabilities so columns and styling remain
        consistent between contexts.
        """
        probability_map: Dict[str, object] = {}
        derived_values = None
        show_comp_features = False
        if isinstance(predictions, dict):
            if "probabilities" in predictions or "derived_features" in predictions:
                probability_map = dict(predictions.get("probabilities", {}))
                derived_values = predictions.get("derived_features")
                show_comp_features = bool(derived_values)
            else:
                probability_map = dict(predictions)

        return build_bid_table(
            records,
            probability_map,
            feature_config=feature_config,
            derived_feature_values=derived_values,
            show_comp_features=show_comp_features,
        )

    @app.callback(
        Output("bid-delete-selector", "options"),
        Output("bid-delete-selector", "value"),
        Input("bid-records-store", "data"),
    )
    def sync_delete_selector(records: Optional[List[Dict[str, object]]]):
        """Expose each current bid as a deletable option for the dropdown.

        Listing the indices lets the "Delete selected" action operate even when
        the table scrolls horizontally or vertically.
        """
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
        """Expose each removed bid as a restorable option for the dropdown.

        The metadata preserved in ``removed-bids-store`` is mapped into labels
        that help the user understand which bids they are restoring.
        """
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
        """Persist inline table edits into the stored snapshot records.

        The timestamp guard ensures the callback only fires in response to real
        edits, while ``apply_table_edits`` performs the heavy lifting of
        validating and normalising cell values.
        """
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
