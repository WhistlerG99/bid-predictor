"""Dropdown population callbacks for the snapshot explorer."""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from dash import Dash, Input, Output, State

from ...data import load_dataset_cached


def _options_from_series(values: pd.Series) -> Tuple[List[dict], Optional[str]]:
    options = [{"label": str(value), "value": str(value)} for value in values]
    value = options[0]["value"] if options else None
    return options, value


def register_filter_callbacks(app: Dash) -> None:
    """Register callbacks that populate the snapshot filter dropdowns."""

    @app.callback(
        Output("carrier-dropdown", "options"),
        Output("carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
    )
    def populate_carriers(dataset_path: Optional[str]):
        if not dataset_path:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "carrier_code" not in dataset.columns:
            return [], None
        carriers = dataset["carrier_code"].dropna().drop_duplicates().sort_values()
        return _options_from_series(carriers)

    @app.callback(
        Output("flight-number-dropdown", "options"),
        Output("flight-number-dropdown", "value"),
        Input("carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_flight_numbers(carrier: Optional[str], dataset_path: Optional[str]):
        if not dataset_path or not carrier:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if {"carrier_code", "flight_number"}.issubset(dataset.columns):
            mask = dataset["carrier_code"] == carrier
            flights = (
                dataset.loc[mask, "flight_number"].dropna().drop_duplicates().sort_values()
            )
            options = [
                {"label": str(number), "value": str(number)} for number in flights
            ]
            value = options[0]["value"] if options else None
            return options, value
        return [], None

    @app.callback(
        Output("travel-date-dropdown", "options"),
        Output("travel-date-dropdown", "value"),
        Input("flight-number-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_travel_dates(
        flight_number: Optional[str],
        carrier: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if {"carrier_code", "flight_number", "travel_date"}.issubset(dataset.columns):
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
            value = options[0]["value"] if options else None
            return options, value
        return [], None

    @app.callback(
        Output("upgrade-dropdown", "options"),
        Output("upgrade-dropdown", "value"),
        Input("travel-date-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_upgrade_types(
        travel_date: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
    ):
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
        options = [{"label": upg, "value": upg} for upg in upgrades]
        value = options[0]["value"] if options else None
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


__all__ = ["register_filter_callbacks"]
