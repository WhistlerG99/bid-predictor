"""Dropdown population callbacks for the feature sensitivity tab."""
from __future__ import annotations

from typing import Optional

from dash import Dash, Input, Output, State

from ..data import load_dataset_cached
from ..scenario import (
    build_carrier_options,
    build_flight_number_options,
    build_travel_date_options,
    build_upgrade_options,
)


def register_filter_callbacks(app: Dash) -> None:
    """Register callbacks that populate scenario selection dropdowns."""

    @app.callback(
        Output("scenario-carrier-dropdown", "options"),
        Output("scenario-carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
    )
    def populate_scenario_carriers(dataset_path: Optional[str]):
        if not dataset_path:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_carrier_options(dataset)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-flight-number-dropdown", "options"),
        Output("scenario-flight-number-dropdown", "value"),
        Input("scenario-carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_scenario_flight_numbers(
        carrier: Optional[str], dataset_path: Optional[str]
    ):
        if not dataset_path or not carrier:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_flight_number_options(dataset, carrier)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-travel-date-dropdown", "options"),
        Output("scenario-travel-date-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        State("scenario-carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_scenario_travel_dates(
        flight_number: Optional[str],
        carrier: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_travel_date_options(dataset, carrier, flight_number)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-upgrade-dropdown", "options"),
        Output("scenario-upgrade-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        State("scenario-carrier-dropdown", "value"),
        State("scenario-flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_scenario_upgrades(
        travel_date: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_upgrade_options(dataset, carrier, flight_number, travel_date)
        value = options[0]["value"] if options else None
        return options, value


__all__ = ["register_filter_callbacks"]
