"""Dropdown population callbacks for the feature sensitivity tab."""
from __future__ import annotations

from typing import Dict, List, Optional

from uuid import uuid4

import pandas as pd
from dash import Dash, Input, Output, State, ctx
from dash.exceptions import PreventUpdate

from ..data import load_dataset_cached
from ..scenario import (
    build_carrier_options,
    build_flight_number_options,
    build_travel_date_options,
    build_upgrade_options,
)


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
    """Register callbacks that populate scenario selection dropdowns."""

    @app.callback(
        Output("scenario-carrier-dropdown", "options"),
        Output("scenario-carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
        Input("scenario-selection-request-store", "data"),
        State("scenario-carrier-dropdown", "value"),
    )
    def populate_scenario_carriers(
        dataset_path: Optional[str],
        selection_request: Optional[Dict[str, str]],
        current_value: Optional[str],
    ):
        """Extract the carriers present in the currently loaded dataset.

        When the dataset path changes the tab should present a fresh list of
        carriers.  Failures to read the file silently fall back to an empty
        dropdown, allowing the UI to surface a user-friendly warning elsewhere.
        """
        if not dataset_path:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_carrier_options(dataset)
        requested_carrier = None
        if selection_request:
            requested_carrier = selection_request.get("carrier")
        value = _choose_value(options, requested_carrier, current_value)
        return options, value

    @app.callback(
        Output("scenario-flight-number-dropdown", "options"),
        Output("scenario-flight-number-dropdown", "value"),
        Input("scenario-carrier-dropdown", "value"),
        Input("scenario-selection-request-store", "data"),
        State("dataset-path-store", "data"),
        State("scenario-flight-number-dropdown", "value"),
    )
    def populate_scenario_flight_numbers(
        carrier: Optional[str],
        selection_request: Optional[Dict[str, str]],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Populate flight numbers for the chosen carrier.

        Each time the carrier changes the callback reloads the dataset and
        returns all matching flight numbers so the next dropdown stays in sync
        with the selection hierarchy.
        """
        if not dataset_path or not carrier:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_flight_number_options(dataset, carrier)
        requested_flight = None
        if selection_request:
            requested_flight = selection_request.get("flight_number")
            if requested_flight is not None:
                requested_flight = str(requested_flight)
        current_value = str(current_value) if current_value is not None else None
        value = _choose_value(options, requested_flight, current_value)
        return options, value

    @app.callback(
        Output("scenario-travel-date-dropdown", "options"),
        Output("scenario-travel-date-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        Input("scenario-selection-request-store", "data"),
        State("scenario-carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("scenario-travel-date-dropdown", "value"),
    )
    def populate_scenario_travel_dates(
        flight_number: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Populate travel dates for the chosen carrier and flight.

        The available dates come directly from the dataset; invalid or missing
        data produces an empty list so the downstream controls reset
        gracefully.
        """
        if not dataset_path or not carrier or not flight_number:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_travel_date_options(dataset, carrier, flight_number)
        requested_date = None
        if selection_request:
            requested_date = selection_request.get("travel_date")
        value = _choose_value(options, requested_date, current_value)
        return options, value

    @app.callback(
        Output("scenario-upgrade-dropdown", "options"),
        Output("scenario-upgrade-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        Input("scenario-selection-request-store", "data"),
        State("scenario-carrier-dropdown", "value"),
        State("scenario-flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("scenario-upgrade-dropdown", "value"),
    )
    def populate_scenario_upgrades(
        travel_date: Optional[str],
        selection_request: Optional[Dict[str, str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
        current_value: Optional[str],
    ):
        """Populate upgrade types for the selected flight and travel date.

        By cascading all upstream filters into the dataset query the callback
        ensures that only valid upgrade types appear once a flight is fully
        specified.  Any data access errors again fall back to empty options so
        the app can communicate problems through a dedicated feedback area.
        """
        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_upgrade_options(dataset, carrier, flight_number, travel_date)
        requested_upgrade = None
        if selection_request:
            requested_upgrade = selection_request.get("upgrade_type")
        value = _choose_value(options, requested_upgrade, current_value)
        return options, value

    @app.callback(
        Output("scenario-selection-request-store", "data"),
        Input("scenario-random-selection-button", "n_clicks"),
        Input("scenario-selection-history-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("scenario-selection-history-store", "data"),
        prevent_initial_call=True,
    )
    def handle_scenario_selection_requests(
        random_clicks: Optional[int],
        history_selection: Optional[str],
        dataset_path: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ):
        """Dispatch random and history-based selection requests for scenarios."""

        trigger = ctx.triggered_id
        if trigger == "scenario-random-selection-button":
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

            candidates = dataset.dropna(subset=required)[list(required)].copy()
            if candidates.empty:
                raise PreventUpdate

            candidates["flight_number"] = candidates["flight_number"].astype(str)
            candidates["carrier_code"] = candidates["carrier_code"].astype(str)
            travel_dates = pd.to_datetime(candidates["travel_date"], errors="coerce")
            candidates = candidates.assign(travel_date=travel_dates)
            candidates = candidates.dropna()
            candidates["travel_date"] = candidates["travel_date"].dt.strftime(
                "%Y-%m-%d"
            )
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

        if trigger == "scenario-selection-history-dropdown":
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
        Output("scenario-selection-history-store", "data"),
        Output("scenario-selection-history-dropdown", "options"),
        Input("scenario-carrier-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        Input("scenario-upgrade-dropdown", "value"),
        Input("dataset-path-store", "data"),
        State("scenario-selection-history-store", "data"),
        prevent_initial_call=True,
    )
    def update_scenario_selection_history(
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ):
        """Record the most recent scenario selection and update the history list."""

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
        filtered_history = filtered_history[:20]

        options = [
            {"label": entry["label"], "value": entry["id"]}
            for entry in filtered_history
        ]
        return filtered_history, options

    @app.callback(
        Output("scenario-selection-history-dropdown", "value"),
        Input("dataset-path-store", "data"),
        prevent_initial_call=True,
    )
    def reset_scenario_selection_history_value(_: Optional[str]):
        """Clear the scenario selection history dropdown when the dataset changes."""

        return None


__all__ = ["register_filter_callbacks"]
