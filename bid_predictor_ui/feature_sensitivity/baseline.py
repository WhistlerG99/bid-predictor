"""Baseline snapshot loading for feature sensitivity."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from dash import Dash, Input, Output, State

from ..data import load_dataset_cached
from ..formatting import clear_derived_features, prepare_bid_record, sort_records_by_bid
from ..scenario import (
    TIME_TO_DEPARTURE_SCENARIO_KEY,
    extract_baseline_snapshot,
    extract_global_baseline_values,
    select_baseline_snapshot,
)

ReturnType = Tuple[
    Optional[List[Dict[str, object]]],
    Optional[List[Dict[str, object]]],
    List[Dict[str, object]],
    str,
    str,
]


def _apply_defaults(
    base_records: List[Dict[str, object]],
    baseline_df: pd.DataFrame,
    feature_config: Optional[Dict[str, object]],
) -> None:
    """Populate each bid record with the global baseline defaults.

    The baseline snapshot may omit values such as seats available or the
    canonical snapshot number.  This helper inspects the extracted
    ``baseline_df`` and writes those defaults onto every record so that the
    UI starts from a consistent state, also aligning timestamps with the
    chosen time-to-departure baseline when possible.
    """
    defaults = extract_global_baseline_values(
        baseline_df, feature_config=feature_config
    )
    seats_default = defaults.get("seats_available")
    if seats_default is not None:
        try:
            seats_default = float(seats_default)
        except (TypeError, ValueError):
            seats_default = None
    time_default = defaults.get(TIME_TO_DEPARTURE_SCENARIO_KEY)
    if time_default is not None:
        try:
            time_default = float(time_default)
        except (TypeError, ValueError):
            time_default = None
    baseline_snapshot_value = select_baseline_snapshot(baseline_df, time_default)

    if not base_records:
        return

    for record in base_records:
        if seats_default is not None:
            record["seats_available"] = seats_default
        if time_default is not None:
            departure_value = record.get("departure_timestamp")
            if departure_value is None:
                continue
            departure_ts = pd.to_datetime(departure_value, errors="coerce")
            if pd.isna(departure_ts):
                continue
            current_ts = departure_ts - pd.to_timedelta(time_default, unit="hour")
            record["current_timestamp"] = current_ts
        if baseline_snapshot_value is not None:
            record["snapshot_num"] = baseline_snapshot_value
        for key, value in defaults.items():
            if key in {"seats_available", TIME_TO_DEPARTURE_SCENARIO_KEY}:
                continue
            if value is not None:
                record[key] = value


def _serialize_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Coerce record values into JSON-serialisable primitives.

    Dash stores callback data as JSON, so timestamps and objects with custom
    ``isoformat`` methods must be converted before returning to the client.
    Any non-serialisable value that cannot be converted is left untouched so
    downstream callbacks can still reason about it.
    """
    serializable: List[Dict[str, object]] = []
    for record in records:
        converted: Dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                converted[key] = value.strftime("%Y-%m-%dT%H:%M:%S")
            elif hasattr(value, "isoformat") and not isinstance(
                value, (str, bytes, int, float, bool)
            ):
                try:
                    converted[key] = value.isoformat()  # type: ignore[attr-defined]
                except Exception:
                    converted[key] = value
            else:
                converted[key] = value
        serializable.append(converted)
    return serializable


def register_baseline_callback(app: Dash) -> None:
    """Register the callback that loads the scenario baseline."""

    @app.callback(
        Output("scenario-records-store", "data"),
        Output("scenario-original-records-store", "data"),
        Output("scenario-removed-bids-store", "data"),
        Output("scenario-snapshot-label", "children"),
        Output("scenario-control-warning", "children"),
        Input("scenario-carrier-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        Input("scenario-upgrade-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("feature-config-store", "data"),
    )
    def update_scenario_baseline(
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
        feature_config: Optional[Dict[str, object]],
    ) -> ReturnType:
        """Load the selected flight snapshot and populate the scenario stores.

        Once the user selects a flight, this callback pulls the corresponding
        bids, applies global defaults, sorts the records, and emits both the
        working and read-only baselines used throughout the scenario tab.  It
        also prepares user-facing summaries and warnings when data cannot be
        retrieved.
        """
        if not dataset_path:
            warning = "Load a dataset to explore scenarios."
            return None, None, [], warning, warning

        if not carrier or not flight_number or not travel_date or not upgrade_type:
            return None, None, [], "", "Select a flight and upgrade type."

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception as exc:
            return None, None, [], "", f"Failed to read dataset: {exc}"

        baseline_df, snapshot_label = extract_baseline_snapshot(
            dataset,
            carrier,
            flight_number,
            travel_date,
            upgrade_type,
        )
        if baseline_df.empty:
            return (
                None,
                None,
                [],
                "No bids found for this selection.",
                "No bids are available for the chosen flight.",
            )

        base_records = [prepare_bid_record(record) for record in baseline_df.to_dict("records")]
        base_records = sort_records_by_bid(base_records)
        _apply_defaults(base_records, baseline_df, feature_config)
        clear_derived_features(base_records, feature_config)

        serializable_records = _serialize_records(base_records)
        summary = f"Using {len(serializable_records)} bids"
        if snapshot_label:
            summary += f" {snapshot_label}"

        original_records = [dict(record) for record in serializable_records]
        return serializable_records, original_records, [], summary, ""


__all__ = ["register_baseline_callback"]
