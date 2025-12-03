"""Scenario table rendering and editing callbacks."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from dash import Dash, Input, Output, State, no_update

from ..data import prepare_prediction_dataframe
from ..feature_config import DEFAULT_UI_FEATURE_CONFIG
from ..predictions import extract_derived_feature_rows, predict
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
        """Render the scenario bid table with optional predictions.

        When a model URI is available the callback scores each bid to populate
        the "Acceptance Probability" columns, respecting any cells that should
        remain locked due to the selected feature.
        """
        predictions: Dict[str, object] = {}
        derived_values = None
        show_comp_features = False
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
                        return_transformed=True,
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
                        else:
                            predictions[column_id] = None
                    config_for_comp = feature_config or DEFAULT_UI_FEATURE_CONFIG
                    comp_features = config_for_comp.get("comp_features", [])
                    transformed = pred_df.attrs.get("transformed_features")
                    derived_values = extract_derived_feature_rows(
                        transformed, comp_features
                    )
                    show_comp_features = bool(derived_values)
        decoded_feature = ScenarioFeature.decode(feature_value)
        locked_cells = resolve_locked_cells(records, decoded_feature)

        if locked_cells:
            return build_bid_table(
                records,
                predictions,
                feature_config=feature_config,
                locked_cells=locked_cells,
                derived_feature_values=derived_values,
                show_comp_features=show_comp_features,
            )
        return build_bid_table(
            records,
            predictions,
            feature_config=feature_config,
            derived_feature_values=derived_values,
            show_comp_features=show_comp_features,
        )

    @app.callback(
        Output("scenario-bid-delete-selector", "options"),
        Output("scenario-bid-delete-selector", "value"),
        Input("scenario-records-store", "data"),
    )
    def sync_scenario_delete_selector(records: Optional[List[Dict[str, object]]]):
        """Mirror the currently displayed bids in the delete dropdown options.

        Each record becomes a selectable entry so users can bulk-remove offers
        even when the table has scrolled out of view.  Clearing the records
        resets the dropdown to an empty multi-select.
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
        Output("scenario-bid-restore-selector", "options"),
        Output("scenario-bid-restore-selector", "value"),
        Input("scenario-removed-bids-store", "data"),
    )
    def sync_scenario_restore_selector(
        removed: Optional[List[Dict[str, object]]]
    ):
        """List the previously removed bids that can be restored.

        The removed-bids store holds metadata about deleted records; this
        callback formats that metadata into dropdown options that drive the
        "Restore bids" control.
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
        """Write inline table edits back to the scenario records store.

        Dash hands us the modified table rows and a timestamp marker each time a
        cell edit occurs.  We use ``apply_table_edits`` to reconcile only the
        unlocked cells and return the updated records so downstream callbacks
        stay consistent.
        """
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
