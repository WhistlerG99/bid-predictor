"""Snapshot loading and bid management callbacks."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import pandas as pd
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback_context,
    html,
    no_update,
)

from ..data import load_dataset_cached
from ..formatting import (
    apply_bid_labels,
    compute_bid_label_map,
    clear_derived_features,
    get_next_bid_label,
    prepare_bid_record,
    sort_records_by_bid,
)
from ..constants import BID_IDENTIFIER_COLUMNS
from ..feature_config import DEFAULT_UI_FEATURE_CONFIG


ReturnType = Tuple[
    html.Div,
    str,
    Optional[Dict[str, object]],
    Optional[List[Dict[str, object]]],
    List[Dict[str, object]],
    object,
    object,
]


def _build_summary_block(
    carrier: str, flight_number: str, travel_date: str, upgrade_type: str
) -> html.Div:
    """Render a simple HTML summary of the selected flight context.

    The block is reused in multiple callback branches to remind users which
    carrier, flight, and upgrade they are exploring.
    """
    return html.Ul(
        [
            html.Li(f"Carrier: {carrier}"),
            html.Li(f"Flight: {flight_number}"),
            html.Li(f"Travel date: {travel_date}"),
            html.Li(f"Upgrade type: {upgrade_type}"),
        ],
        style={"paddingLeft": "1.2rem", "margin": "0"},
    )


def _restore_snapshot(
    summary: html.Div,
    baseline_records: Sequence[Dict[str, object]],
    baseline_meta: Dict[str, object],
    feature_config: Optional[Dict[str, object]],
) -> ReturnType:
    """Return the baseline snapshot state when the user restores defaults.

    The function clones the original snapshot so editing actions do not mutate
    the cached baseline data.
    """
    if not baseline_records:
        return (
            summary,
            "No baseline snapshot available to restore.",
            baseline_meta,
            None,
            [],
            no_update,
            no_update,
        )

    restored_records = [dict(record) for record in baseline_records]
    clear_derived_features(restored_records, feature_config)
    restored_meta = dict(baseline_meta)
    restored_meta["num_offers"] = len(restored_records)
    return (
        summary,
        "Restored snapshot to original values.",
        restored_meta,
        restored_records,
        [],
        no_update,
        no_update,
    )


def _add_bid(
    summary: html.Div,
    existing_records: Sequence[Dict[str, object]],
    snapshot_meta: Optional[Dict[str, object]],
    feature_config: Optional[Dict[str, object]],
) -> ReturnType:
    """Clone the first bid to create a new editable record.

    We seed new bids from the first record so categorical columns retain valid
    values and the table remains sortable.
    """
    if not existing_records:
        return (
            summary,
            "Load bids before adding new ones.",
            snapshot_meta,
            list(existing_records),
            [],
            no_update,
            no_update,
        )

    base = dict(existing_records[0])
    new_bid = {key: base.get(key) for key in base}
    config = feature_config or DEFAULT_UI_FEATURE_CONFIG
    display_features = [
        feature
        for feature in config.get("display_features", [])
        if feature != "Acceptance Probability"
    ]
    if not display_features:
        display_features = [
            feature
            for feature in DEFAULT_UI_FEATURE_CONFIG.get("display_features", [])
            if feature != "Acceptance Probability"
        ]

    for feature in display_features:
        new_bid.setdefault(feature, base.get(feature))
    for identifier in BID_IDENTIFIER_COLUMNS:
        if identifier in new_bid:
            new_bid[identifier] = None
    new_bid["Bid #"] = get_next_bid_label(existing_records)
    new_bid.setdefault("offer_status", "pending")

    prepared = prepare_bid_record(new_bid)
    updated = sort_records_by_bid(list(existing_records) + [prepared])
    clear_derived_features(updated, feature_config)

    new_meta = dict(snapshot_meta or {})
    new_meta["num_offers"] = len(updated)
    return summary, "", new_meta, updated, [], no_update, no_update


def _restore_bids(
    summary: html.Div,
    existing_records: Sequence[Dict[str, object]],
    snapshot_meta: Optional[Dict[str, object]],
    existing_removed: Sequence[Dict[str, object]],
    restore_ids: Sequence[str],
    feature_config: Optional[Dict[str, object]],
) -> ReturnType:
    """Rehydrate previously removed bids and place them back into the list.

    Removed bids are stored with enough metadata to recreate prepared records,
    allowing the UI to support undo operations even across multiple actions.
    """
    if not restore_ids:
        return (
            summary,
            "Select removed bids to restore.",
            snapshot_meta,
            list(existing_records),
            list(existing_removed),
            no_update,
            no_update,
        )

    restore_set = set(restore_ids)
    restored_records: List[Dict[str, object]] = []
    remaining_removed: List[Dict[str, object]] = []
    for item in existing_removed:
        if item.get("id") in restore_set:
            restored_records.append(prepare_bid_record(item.get("record", {})))
        else:
            remaining_removed.append(item)

    if not restored_records:
        return (
            summary,
            "No matching removed bids found.",
            snapshot_meta,
            list(existing_records),
            list(existing_removed),
            no_update,
            no_update,
        )

    working = sort_records_by_bid(list(existing_records) + restored_records)
    clear_derived_features(working, feature_config)
    new_meta = dict(snapshot_meta or {})
    new_meta["num_offers"] = len(working)
    return summary, "", new_meta, working, remaining_removed, no_update, no_update


def _delete_bids(
    summary: html.Div,
    existing_records: Sequence[Dict[str, object]],
    snapshot_meta: Optional[Dict[str, object]],
    existing_removed: Sequence[Dict[str, object]],
    selections: Sequence[int],
    feature_config: Optional[Dict[str, object]],
) -> ReturnType:
    """Remove selected bids and store them for potential restoration.

    The helper normalises selections from both the table and the supporting
    dropdowns before producing a cleaned list and logging removed entries in
    the dedicated store.
    """
    working_records = list(existing_records)
    if not working_records:
        return (
            summary,
            "No bids available to delete.",
            snapshot_meta,
            working_records,
            list(existing_removed),
            no_update,
            no_update,
        )

    indices = sorted({int(idx) for idx in selections}, reverse=True)
    if not indices:
        return (
            summary,
            "Select bids to delete.",
            snapshot_meta,
            working_records,
            list(existing_removed),
            no_update,
            no_update,
        )

    removed_entries: List[Dict[str, object]] = []
    for idx in indices:
        if 0 <= idx < len(working_records):
            removed_record = working_records.pop(idx)
            removed_entries.append(
                {
                    "id": str(uuid4()),
                    "label": f"Bid {removed_record.get('Bid #') or idx + 1}",
                    "record": removed_record,
                }
            )

    working_records = sort_records_by_bid(working_records)
    clear_derived_features(working_records, feature_config)
    new_meta = dict(snapshot_meta or {})
    new_meta["num_offers"] = len(working_records)
    updated_removed = list(existing_removed) + removed_entries
    return summary, "", new_meta, working_records, updated_removed, no_update, no_update


def _load_snapshot(
    dataset: pd.DataFrame,
    summary: html.Div,
    carrier: str,
    flight_number: str,
    travel_date: str,
    upgrade_type: str,
    snapshot_value: str,
    feature_config: Optional[Dict[str, object]],
) -> ReturnType:
    """Load the selected snapshot into Dash stores and compute metadata.

    Besides converting timestamps and preparing records, the helper also
    computes summary metrics (such as time before departure) used throughout
    the snapshot tab.
    """
    travel_date_dt = pd.to_datetime(travel_date).date()
    mask = (
        (dataset["carrier_code"] == carrier)
        & (dataset["flight_number"].astype(str) == str(flight_number))
        & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
        & (dataset["upgrade_type"] == upgrade_type)
    )
    subset = dataset.loc[mask].copy()

    if subset.empty:
        return summary, "No rows found for the selected flight.", None, None, [], no_update, no_update

    if "snapshot_num" not in subset.columns:
        return (
            summary,
            "Snapshot information is unavailable in this dataset.",
            None,
            None,
            [],
            no_update,
            no_update,
        )

    label_map, label_column = compute_bid_label_map(subset)
    snapshot_df = subset.loc[
        subset["snapshot_num"].astype(str) == str(snapshot_value)
    ].copy()

    if snapshot_df.empty:
        return (
            summary,
            "No rows found for the selected snapshot.",
            None,
            None,
            [],
            no_update,
            no_update,
        )

    snapshot_df = apply_bid_labels(snapshot_df, label_map, label_column)
    if "Bid #" in snapshot_df.columns:
        snapshot_df = snapshot_df.sort_values("Bid #")

    for column in ["current_timestamp", "departure_timestamp", "travel_date"]:
        if column in snapshot_df.columns:
            snapshot_df[column] = snapshot_df[column].apply(
                lambda value: value.isoformat() if isinstance(value, pd.Timestamp) else value
            )

    base_data = [prepare_bid_record(record) for record in snapshot_df.to_dict("records")]
    base_data = sort_records_by_bid(base_data)
    clear_derived_features(base_data, feature_config)

    seats_value = None
    seats_available = snapshot_df.get("seats_available")
    if isinstance(seats_available, pd.Series) and not seats_available.empty:
        try:
            seats_value = float(seats_available.iloc[0])
        except (TypeError, ValueError):
            seats_value = None

    departure_ts = pd.to_datetime(snapshot_df.get("departure_timestamp"), errors="coerce")
    departure_ts = departure_ts.iloc[0] if isinstance(departure_ts, pd.Series) else None
    current_ts = pd.to_datetime(snapshot_df.get("current_timestamp"), errors="coerce")
    current_ts = current_ts.iloc[0] if isinstance(current_ts, pd.Series) else None

    delta_hours = None
    if departure_ts is not None and current_ts is not None:
        delta = departure_ts - current_ts
        delta_hours = max(delta.total_seconds() / 3600.0, 0)

    snapshot_meta = {
        "carrier": carrier,
        "flight_number": flight_number,
        "travel_date": travel_date,
        "upgrade_type": upgrade_type,
        "snapshot": snapshot_value,
        "seats_available": seats_value,
        "num_offers": len(base_data),
        "departure_timestamp": departure_ts.isoformat() if isinstance(departure_ts, pd.Timestamp) else None,
        "current_timestamp": current_ts.isoformat() if isinstance(current_ts, pd.Timestamp) else None,
        "time_before_departure_hours": delta_hours,
    }

    baseline_records = [dict(record) for record in base_data]
    baseline_meta = dict(snapshot_meta)
    return summary, "", snapshot_meta, base_data, [], baseline_records, baseline_meta


def register_snapshot_view_callbacks(app: Dash) -> None:
    """Register callbacks that load snapshots and manage bid lists."""

    @app.callback(
        Output("flight-summary", "children"),
        Output("snapshot-feedback", "children"),
        Output("snapshot-meta-store", "data"),
        Output("bid-records-store", "data"),
        Output("removed-bids-store", "data"),
        Output("baseline-bid-records-store", "data"),
        Output("baseline-snapshot-meta-store", "data"),
        Input("snapshot-dropdown", "value"),
        Input("add-bid", "n_clicks"),
        Input("delete-bid", "n_clicks"),
        Input("restore-bid", "n_clicks"),
        Input("restore-snapshot", "n_clicks"),
        State("bid-records-store", "data"),
        State("bid-table", "selected_columns"),
        State("bid-delete-selector", "value"),
        State("bid-restore-selector", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("upgrade-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("snapshot-meta-store", "data"),
        State("removed-bids-store", "data"),
        State("baseline-bid-records-store", "data"),
        State("baseline-snapshot-meta-store", "data"),
        State("feature-config-store", "data"),
    )
    def update_snapshot_view(
        snapshot_value: Optional[str],
        add_clicks: int,
        delete_clicks: int,
        restore_clicks: int,
        restore_snapshot_clicks: int,
        existing_records: Optional[List[Dict[str, object]]],
        selected_columns: Optional[List[str]],
        delete_selector: Optional[List[int]],
        restore_selector: Optional[List[str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
        snapshot_meta: Optional[Dict[str, object]],
        removed_store: Optional[List[Dict[str, object]]],
        baseline_records_store: Optional[List[Dict[str, object]]],
        baseline_meta_store: Optional[Dict[str, object]],
        feature_config: Optional[Dict[str, object]],
    ) -> ReturnType:
        """Handle snapshot loading and bid list mutations for the tab.

        The callback dispatches to the helper routines above based on which
        input fired—supporting load, add, delete, restore, and baseline
        restoration operations.  It also guards against incomplete selections
        and missing datasets so the UI can provide helpful feedback.
        """
        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )
        existing_removed = list(removed_store or [])
        baseline_records = list(baseline_records_store or [])
        baseline_meta = dict(baseline_meta_store or {})

        if not dataset_path:
            return (
                html.Div("Load a dataset to begin."),
                "",
                None,
                None,
                existing_removed,
                no_update,
                no_update,
            )

        dataset = load_dataset_cached(dataset_path)

        if not carrier or not flight_number or not travel_date or not upgrade_type:
            return (
                html.Div("Select a carrier, flight, travel date, and upgrade type."),
                "",
                snapshot_meta,
                existing_records,
                existing_removed,
                no_update,
                no_update,
            )

        summary_block = _build_summary_block(carrier, flight_number, travel_date, upgrade_type)
        records_list = list(existing_records or [])

        if triggered == "restore-snapshot":
            return _restore_snapshot(
                summary_block,
                baseline_records,
                baseline_meta,
                feature_config,
            )

        if triggered == "add-bid":
            return _add_bid(
                summary_block,
                records_list,
                snapshot_meta,
                feature_config,
            )

        if triggered == "restore-bid":
            return _restore_bids(
                summary_block,
                records_list,
                snapshot_meta,
                existing_removed,
                restore_selector or [],
                feature_config,
            )

        if triggered == "delete-bid":
            selections: List[int] = []
            if selected_columns:
                selections.extend(
                    int(col_id.replace("bid_", ""))
                    for col_id in selected_columns
                    if col_id.startswith("bid_") and col_id.replace("bid_", "").isdigit()
                )
            if delete_selector:
                selections.extend(int(value) for value in delete_selector)
            return _delete_bids(
                summary_block,
                records_list,
                snapshot_meta,
                existing_removed,
                selections,
                feature_config,
            )

        if triggered != "snapshot-dropdown":
            return (
                summary_block,
                "",
                snapshot_meta,
                existing_records,
                existing_removed,
                no_update,
                no_update,
            )

        if snapshot_value is None:
            return (
                summary_block,
                "Select a snapshot to view bids.",
                snapshot_meta,
                existing_records,
                existing_removed,
                no_update,
                no_update,
            )

        return _load_snapshot(
            dataset,
            summary_block,
            carrier,
            flight_number,
            travel_date,
            upgrade_type,
            snapshot_value,
            feature_config,
        )


__all__ = ["register_snapshot_view_callbacks"]
