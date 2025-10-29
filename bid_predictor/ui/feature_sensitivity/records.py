"""Scenario record management callbacks."""
from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

from dash import Dash, Input, Output, State, callback_context, no_update

from ..constants import BID_IDENTIFIER_COLUMNS
from ..feature_config import DEFAULT_UI_FEATURE_CONFIG
from ..formatting import (
    get_next_bid_label,
    prepare_bid_record,
    recompute_usd_metrics,
    sort_records_by_bid,
)


def register_record_callbacks(app: Dash) -> None:
    """Register callbacks that add, delete, and restore scenario bids."""

    @app.callback(
        Output("scenario-records-store", "data", allow_duplicate=True),
        Output("scenario-removed-bids-store", "data", allow_duplicate=True),
        Output("scenario-table-feedback", "children"),
        Input("scenario-add-bid", "n_clicks"),
        Input("scenario-delete-bid", "n_clicks"),
        Input("scenario-restore-bid", "n_clicks"),
        Input("scenario-restore-baseline", "n_clicks"),
        State("scenario-records-store", "data"),
        State("scenario-removed-bids-store", "data"),
        State("scenario-bid-delete-selector", "value"),
        State("scenario-bid-restore-selector", "value"),
        State("scenario-original-records-store", "data"),
        State("scenario-bid-table", "selected_columns"),
        State("feature-config-store", "data"),
        prevent_initial_call=True,
    )
    def update_scenario_records(
        add_clicks: int,
        delete_clicks: int,
        restore_clicks: int,
        restore_baseline_clicks: int,
        records: Optional[List[Dict[str, object]]],
        removed_store: Optional[List[Dict[str, object]]],
        delete_selector: Optional[List[int]],
        restore_selector: Optional[List[str]],
        original_records: Optional[List[Dict[str, object]]],
        selected_columns: Optional[List[str]],
        feature_config: Optional[Dict[str, object]],
    ):
        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )

        current_records = [dict(record) for record in records or []]
        existing_removed = list(removed_store or [])

        if triggered == "scenario-restore-baseline":
            if not original_records:
                return no_update, no_update, "No defaults available to restore."
            restored = [dict(record) for record in original_records]
            recompute_usd_metrics(restored)
            return restored, [], "Restored default bids."

        if triggered == "scenario-add-bid":
            if not current_records:
                return no_update, no_update, "Load bids before adding new ones."
            base = current_records[0]
            new_bid = {key: base.get(key) for key in base}
            for identifier in BID_IDENTIFIER_COLUMNS:
                if identifier in new_bid:
                    new_bid[identifier] = None
            config = feature_config or DEFAULT_UI_FEATURE_CONFIG
            display_features = [
                feature
                for feature in config.get("display_features", [])
                if feature != "Acceptance Probability"
            ]
            if not display_features:
                display_features = [
                    feature
                    for feature in DEFAULT_UI_FEATURE_CONFIG.get(
                        "display_features", []
                    )
                    if feature != "Acceptance Probability"
                ]
            for feature in display_features:
                new_bid.setdefault(feature, base.get(feature))
            new_bid["Bid #"] = get_next_bid_label(current_records)
            new_bid.setdefault("offer_status", "pending")
            prepared = prepare_bid_record(new_bid)
            updated = sort_records_by_bid(current_records + [prepared])
            recompute_usd_metrics(updated)
            return updated, existing_removed, ""

        if triggered == "scenario-delete-bid":
            if not current_records:
                return no_update, no_update, "No bids available to delete."
            selections: set[int] = set()
            if selected_columns:
                selections.update(
                    int(col_id.replace("bid_", ""))
                    for col_id in selected_columns
                    if col_id.startswith("bid_") and col_id.replace("bid_", "").isdigit()
                )
            if delete_selector:
                selections.update(int(idx) for idx in delete_selector)
            indices_to_remove = sorted(selections, reverse=True)
            if not indices_to_remove:
                return no_update, existing_removed, "Select bids to delete."
            working = list(current_records)
            removed_entries: List[Dict[str, object]] = []
            for idx in indices_to_remove:
                if 0 <= idx < len(working):
                    removed_record = working.pop(idx)
                    removed_entries.append(
                        {
                            "id": str(uuid4()),
                            "label": f"Bid {removed_record.get('Bid #') or idx + 1}",
                            "record": removed_record,
                        }
                    )
            if not removed_entries:
                return no_update, existing_removed, "No matching bids were removed."
            working = sort_records_by_bid(working)
            recompute_usd_metrics(working)
            updated_removed = existing_removed + removed_entries
            return working, updated_removed, ""

        if triggered == "scenario-restore-bid":
            if not restore_selector:
                return no_update, existing_removed, "Select removed bids to restore."
            restore_ids = set(restore_selector)
            restored_records: List[Dict[str, object]] = []
            remaining_removed: List[Dict[str, object]] = []
            for item in existing_removed:
                if item.get("id") in restore_ids:
                    restored_records.append(prepare_bid_record(item.get("record", {})))
                else:
                    remaining_removed.append(item)
            if not restored_records:
                return no_update, existing_removed, "No matching removed bids found."
            working = sort_records_by_bid(current_records + restored_records)
            recompute_usd_metrics(working)
            return working, remaining_removed, ""

        return no_update, no_update, ""


__all__ = ["register_record_callbacks"]
