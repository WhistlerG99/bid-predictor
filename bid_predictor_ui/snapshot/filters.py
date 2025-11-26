"""Dropdown population callbacks for the snapshot explorer."""
from __future__ import annotations

from typing import Dict, List, Optional

from uuid import uuid4

import pandas as pd
from dash import Dash, Input, Output, State, ctx
from dash.exceptions import PreventUpdate

from ..data import load_dataset_cached


def _options_from_series(values: pd.Series) -> List[dict]:
    """Convert a pandas series into a list of Dash dropdown options."""

    return [{"label": str(value), "value": str(value)} for value in values]


def _choose_value(
    options: List[Dict[str, str]],
    requested_value: Optional[str],
    current_value: Optional[str],
) -> Optional[str]:
    """Pick the dropdown value that best matches the current context."""

    def _contains(value: Optional[str]) -> bool:
        return bool(value) and any(option["value"] == value for option in options)

    if _contains(requested_value):
        return requested_value
    if _contains(current_value):
        return current_value
    return options[0]["value"] if options else None


def register_filter_callbacks(app: Dash) -> None:
    """Register callbacks that populate the snapshot filter dropdowns."""

    @app.callback(
        Output("carrier-dropdown", "options"),
        Output("carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
        Input("snapshot-selection-request-store", "data"),
        State("carrier-dropdown", "value"),
    )
    def populate_carriers(
        dataset_path: Optional[str],
        selection_request: Optional[Dict[str, str]],
        current_value: Optional[str],
    ):
        """Populate the carrier dropdown when a dataset is loaded."""

        if not dataset_path:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "carrier_code" not in dataset.columns:
            return [], None

        carriers = dataset["carrier_code"].dropna().drop_duplicates().sort_values()
        options = _options_from_series(carriers)
        requested_carrier = None
        if selection_request:
            requested_carrier = selection_request.get("carrier")

        value = _choose_value(options, requested_carrier, current_value)
        return options, value

    @app.callback(
        Output("flight-number-dropdown", "options"),
        Output("flight-number-dropdown", "value"),
        Input("carrier-dropdown", "value"),
        Input("snapshot-selection-request-store", "data"),
        State("dataset-path-store", "data"),
        State("flight-number-dropdown", "value"),
    )
    def populate_flight_numbers(
        carrier: Optional[str],
        selection_request: Optional[Dict[str, str]],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Filter flight numbers based on the selected carrier."""

        if not dataset_path or not carrier:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if not {"carrier_code", "flight_number"}.issubset(dataset.columns):
            return [], None

        mask = dataset["carrier_code"] == carrier
        flights = (
            dataset.loc[mask, "flight_number"].dropna().drop_duplicates().sort_values()
        )
        options = [{"label": str(number), "value": str(number)} for number in flights]
        requested_flight = None
        if selection_request:
            requested_flight = selection_request.get("flight_number")
            if requested_flight is not None:
                requested_flight = str(requested_flight)

        current_value = str(current_value) if current_value is not None else None
        value = _choose_value(options, requested_flight, current_value)
        return options, value

    @app.callback(
        Output("travel-date-dropdown", "options"),
        Output("travel-date-dropdown", "value"),
        Input("flight-number-dropdown", "value"),
        Input("snapshot-selection-request-store", "data"),
        State("carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("travel-date-dropdown", "value"),
    )
    def populate_travel_dates(
        flight_number: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Filter travel dates once the flight number has been chosen."""

        if not dataset_path or not carrier or not flight_number:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        required = {"carrier_code", "flight_number", "travel_date"}
        if not required.issubset(dataset.columns):
            return [], None

        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
        )
        travel_dates = (
            pd.to_datetime(dataset.loc[mask, "travel_date"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        options = [
            {"label": date.strftime("%Y-%m-%d"), "value": date.strftime("%Y-%m-%d")}
            for date in travel_dates
        ]
        requested_date = None
        if selection_request:
            requested_date = selection_request.get("travel_date")

        value = _choose_value(options, requested_date, current_value)
        return options, value

    @app.callback(
        Output("upgrade-dropdown", "options"),
        Output("upgrade-dropdown", "value"),
        Input("travel-date-dropdown", "value"),
        Input("snapshot-selection-request-store", "data"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("upgrade-dropdown", "value"),
    )
    def populate_upgrade_types(
        travel_date: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """List upgrade types for the fully specified flight selection."""

        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "upgrade_type" not in dataset.columns:
            return [], None

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
        )
        upgrades = (
            dataset.loc[mask, "upgrade_type"].dropna().drop_duplicates().sort_values()
        )
        options = [{"label": str(upg), "value": str(upg)} for upg in upgrades]
        requested_upgrade = None
        if selection_request:
            requested_upgrade = selection_request.get("upgrade_type")

        value = _choose_value(options, requested_upgrade, current_value)
        return options, value

    @app.callback(
        Output("snapshot-dropdown", "options"),
        Output("snapshot-dropdown", "value"),
        Input("upgrade-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_snapshots(
        upgrade_type: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        dataset_path: Optional[str],
    ):
        """List available snapshots after the user selects an upgrade type."""

        if (
            not dataset_path
            or not carrier
            or not flight_number
            or not travel_date
            or not upgrade_type
        ):
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "snapshot_num" not in dataset.columns:
            return [], None

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        snapshots = (
            dataset.loc[mask, "snapshot_num"].dropna().drop_duplicates().sort_values()
        )
        options = [
            {"label": f"Snapshot {snap}", "value": str(snap)} for snap in snapshots
        ]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("snapshot-selection-request-store", "data"),
        Input("random-selection-button", "n_clicks"),
        Input("selection-history-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("selection-history-store", "data"),
        prevent_initial_call=True,
    )
    def handle_selection_requests(
        random_clicks: Optional[int],
        history_selection: Optional[str],
        dataset_path: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ):
        """Dispatch random and history-based selection requests."""

        trigger = ctx.triggered_id
        if trigger == "random-selection-button":
            if not dataset_path:
                raise PreventUpdate

            dataset = load_dataset_cached(dataset_path)
            required = {
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
            }
            if not required.issubset(dataset.columns):
                raise PreventUpdate

            candidates = (
                dataset.dropna(subset=required)[list(required)].copy()
            )
            if candidates.empty:
                raise PreventUpdate

            candidates["flight_number"] = candidates["flight_number"].astype(str)
            candidates["carrier_code"] = candidates["carrier_code"].astype(str)
            travel_dates = pd.to_datetime(
                candidates["travel_date"], errors="coerce"
            )
            candidates = candidates.assign(travel_date=travel_dates)
            candidates = candidates.dropna()
            candidates["travel_date"] = candidates["travel_date"].dt.strftime("%Y-%m-%d")
            candidates = candidates.drop_duplicates()
            if candidates.empty:
                raise PreventUpdate

            selection = candidates.sample(n=1).iloc[0]
            return {
                "carrier": selection["carrier_code"],
                "flight_number": selection["flight_number"],
                "travel_date": selection["travel_date"],
                "upgrade_type": selection["upgrade_type"],
                "trigger": str(uuid4()),
            }

        if trigger == "selection-history-dropdown":
            if not history_selection:
                raise PreventUpdate

            history = history or []
            for entry in history:
                if entry.get("id") == history_selection:
                    return {
                        "carrier": entry["carrier"],
                        "flight_number": entry["flight_number"],
                        "travel_date": entry["travel_date"],
                        "upgrade_type": entry["upgrade_type"],
                        "trigger": str(uuid4()),
                    }
            raise PreventUpdate

        raise PreventUpdate

    @app.callback(
        Output("selection-history-store", "data"),
        Output("selection-history-dropdown", "options"),
        Input("carrier-dropdown", "value"),
        Input("flight-number-dropdown", "value"),
        Input("travel-date-dropdown", "value"),
        Input("upgrade-dropdown", "value"),
        Input("dataset-path-store", "data"),
        State("selection-history-store", "data"),
        prevent_initial_call=True,
    )
    def update_selection_history(
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ):
        """Record the most recent flight selection and update the history dropdown."""

        trigger = ctx.triggered_id
        if trigger == "dataset-path-store":
            return [], []

        if not (carrier and flight_number and travel_date and upgrade_type):
            raise PreventUpdate

        history = history or []
        entry_id = "|".join(
            [
                str(carrier),
                str(flight_number),
                str(travel_date),
                str(upgrade_type),
            ]
        )
        label = f"{carrier} {flight_number} · {travel_date} · {upgrade_type}"
        new_entry = {
            "id": entry_id,
            "carrier": str(carrier),
            "flight_number": str(flight_number),
            "travel_date": str(travel_date),
            "upgrade_type": str(upgrade_type),
            "label": label,
        }

        filtered_history = [entry for entry in history if entry.get("id") != entry_id]
        filtered_history.insert(0, new_entry)
        # Keep the history at a manageable size
        filtered_history = filtered_history[:20]

        options = [
            {"label": entry["label"], "value": entry["id"]}
            for entry in filtered_history
        ]
        return filtered_history, options

    @app.callback(
        Output("selection-history-dropdown", "value"),
        Input("dataset-path-store", "data"),
        prevent_initial_call=True,
    )
    def reset_selection_history_value(_: Optional[str]):
        """Clear the selection history dropdown when the dataset changes."""

        return None


__all__ = ["register_filter_callbacks"]
