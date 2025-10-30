"""Callbacks that synchronize the snapshot summary controls."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, no_update

from ..constants import BID_IDENTIFIER_COLUMNS
from ..formatting import (
    get_next_bid_label,
    normalize_offer_time,
    prepare_bid_record,
    clear_derived_features,
    sort_records_by_bid,
)


def register_summary_callbacks(app: Dash) -> None:
    """Register callbacks that keep the summary inputs in sync with the data."""

    @app.callback(
        Output("seats-available-input", "value"),
        Output("offers-input", "value"),
        Output("time-before-days-input", "value"),
        Output("time-before-hours-input", "value"),
        Input("snapshot-meta-store", "data"),
        Input("bid-records-store", "data"),
    )
    def sync_inputs(meta: Optional[Dict[str, object]], records: Optional[List[Dict[str, object]]]):
        """Synchronise the summary input boxes with the current snapshot data."""
        seats_value = meta.get("seats_available") if meta else None
        offers_value = len(records) if records else 0
        delta_hours = meta.get("time_before_departure_hours") if meta else None
        if delta_hours is not None:
            days = int(delta_hours // 24)
            hours = int(round(delta_hours - days * 24))
        else:
            days = None
            hours = None
        return seats_value, offers_value, days, hours

    @app.callback(
        Output("bid-records-store", "data", allow_duplicate=True),
        Output("snapshot-meta-store", "data", allow_duplicate=True),
        Input("seats-available-input", "value"),
        Input("offers-input", "value"),
        Input("time-before-days-input", "value"),
        Input("time-before-hours-input", "value"),
        State("bid-records-store", "data"),
        State("snapshot-meta-store", "data"),
        State("feature-config-store", "data"),
        prevent_initial_call=True,
    )
    def apply_summary_overrides(
        seats_value: Optional[float],
        offers_value: Optional[int],
        days_value: Optional[int],
        hours_value: Optional[int],
        records: Optional[List[Dict[str, object]]],
        meta: Optional[Dict[str, object]],
        feature_config: Optional[Dict[str, object]],
    ):
        """Apply edits from the summary controls back to the stores.

        Depending on the triggering control, the callback updates seats,
        expands or trims the bid list, or shifts timestamps to reflect the new
        time-before-departure.  All branches recompute helper metrics so the UI
        remains self-consistent.
        """
        if records is None or meta is None:
            return no_update, no_update

        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )

        updated_records = [dict(record) for record in records]
        updated_meta = dict(meta)

        if triggered == "seats-available-input":
            if seats_value is None:
                return no_update, no_update
            for record in updated_records:
                record["seats_available"] = seats_value
            updated_meta["seats_available"] = seats_value
            return updated_records, updated_meta

        if triggered == "offers-input":
            if offers_value is None or offers_value < 0:
                return no_update, no_update
            updated_records = sort_records_by_bid(updated_records)
            current_len = len(updated_records)
            if offers_value == current_len:
                updated_meta["num_offers"] = offers_value
                for record in updated_records:
                    normalize_offer_time(record)
                clear_derived_features(updated_records, feature_config)
                return updated_records, updated_meta
            if offers_value > current_len and current_len > 0:
                template = updated_records[0]
                for _ in range(offers_value - current_len):
                    new_bid = dict(template)
                    for identifier in BID_IDENTIFIER_COLUMNS:
                        if identifier in new_bid:
                            new_bid[identifier] = None
                    new_bid["Bid #"] = get_next_bid_label(updated_records)
                    new_bid.setdefault("offer_status", "pending")
                    updated_records.append(prepare_bid_record(new_bid))
            elif offers_value < current_len:
                updated_records = updated_records[:offers_value]
            updated_records = sort_records_by_bid(updated_records)
            for record in updated_records:
                normalize_offer_time(record)
            clear_derived_features(updated_records, feature_config)
            updated_meta["num_offers"] = len(updated_records)
            return updated_records, updated_meta

        if triggered in {"time-before-days-input", "time-before-hours-input"}:
            if days_value is None and hours_value is None:
                return no_update, no_update
            hours_value = hours_value or 0
            days_value = days_value or 0
            total_hours = max(days_value * 24 + hours_value, 0)
            departure_iso = updated_meta.get("departure_timestamp")
            if departure_iso:
                departure_ts = pd.to_datetime(departure_iso)
                new_current = departure_ts - pd.Timedelta(hours=total_hours)
                for record in updated_records:
                    record["current_timestamp"] = new_current.isoformat()
                updated_meta["current_timestamp"] = new_current.isoformat()
            updated_meta["time_before_departure_hours"] = total_hours
            return updated_records, updated_meta

        return no_update, no_update


__all__ = ["register_summary_callbacks"]
