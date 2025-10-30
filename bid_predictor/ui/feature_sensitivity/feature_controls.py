"""Feature selection and baseline control callbacks."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, no_update

from ..formatting import safe_float
from ..scenario import (
    TIME_TO_DEPARTURE_SCENARIO_KEY,
    ScenarioFeature,
    build_feature_options,
    extract_global_baseline_values,
    records_to_dataframe,
    select_feature,
)


def register_feature_callbacks(app: Dash) -> None:
    """Register callbacks related to feature selection and baseline overrides."""

    @app.callback(
        Output("scenario-feature-dropdown", "options"),
        Output("scenario-feature-dropdown", "value"),
        Input("scenario-records-store", "data"),
        State("scenario-feature-dropdown", "value"),
        State("feature-config-store", "data"),
    )
    def populate_scenario_features(
        baseline_records: Optional[List[Dict[str, object]]],
        current_value: Optional[str],
        feature_config: Optional[Dict[str, object]],
    ):
        """List the features that can be explored for the loaded baseline.

        The available options depend on the columns present in the baseline
        records and any overrides defined in the feature configuration.  The
        callback keeps a previously selected feature active when possible so
        users do not lose their context while adjusting other controls.
        """
        baseline_df = records_to_dataframe(baseline_records)
        features = build_feature_options(
            baseline_df, feature_config=feature_config
        )
        options = [{"label": feature.label, "value": feature.encode()} for feature in features]
        selected_value = None
        if current_value and any(option["value"] == current_value for option in options):
            selected_value = current_value
        elif options:
            selected_value = options[0]["value"]
        return options, selected_value

    @app.callback(
        Output("scenario-baseline-seats", "value"),
        Output("scenario-baseline-time-to-departure", "value"),
        Output("scenario-baseline-seats-container", "style"),
        Output("scenario-baseline-time-container", "style"),
        Input("scenario-records-store", "data"),
        Input("scenario-feature-dropdown", "value"),
        State("scenario-baseline-seats", "value"),
        State("scenario-baseline-time-to-departure", "value"),
        State("feature-config-store", "data"),
    )
    def update_scenario_baseline_controls(
        baseline_records: Optional[List[Dict[str, object]]],
        feature_value: Optional[str],
        seats_state: Optional[float],
        time_state: Optional[float],
        feature_config: Optional[Dict[str, object]],
    ):
        """Adjust baseline override inputs in response to the current context.

        The seats and time-to-departure inputs should mirror the dataset
        defaults when new flights are loaded, but remain untouched when the
        user manually edits them.  This callback orchestrates those rules,
        determines which inputs should be visible for the selected feature, and
        normalises numeric display values to avoid awkward floating point
        representations in the UI.
        """
        baseline_df = records_to_dataframe(baseline_records)
        defaults = extract_global_baseline_values(
            baseline_df, feature_config=feature_config
        )
        seats_default = defaults.get("seats_available")
        time_default = defaults.get(TIME_TO_DEPARTURE_SCENARIO_KEY)

        triggered_prop = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
        dataset_triggered = triggered_prop.startswith("scenario-records-store")

        seats_value = seats_state
        time_value = time_state
        if dataset_triggered:
            seats_value = seats_default
            time_value = time_default
        else:
            if seats_value is None and seats_default is not None:
                seats_value = seats_default
            if time_value is None and time_default is not None:
                time_value = time_default

        decoded_feature = ScenarioFeature.decode(feature_value)
        seats_style: Dict[str, object] = {"display": "none"}
        time_style: Dict[str, object] = {"display": "none"}
        if decoded_feature is not None:
            visible_style = {"display": "block", "marginBottom": "0.75rem"}
            if decoded_feature.key == "seats_available":
                time_style = visible_style
            elif decoded_feature.kind == "time_to_departure":
                seats_style = visible_style
            else:
                seats_style = visible_style
                time_style = visible_style

        seats_numeric = safe_float(seats_value)
        if seats_numeric is not None:
            if float(seats_numeric).is_integer():
                seats_value = int(round(seats_numeric))
            else:
                seats_value = round(float(seats_numeric), 4)

        time_numeric = safe_float(time_value)
        if time_numeric is not None:
            if float(time_numeric).is_integer():
                time_value = int(round(time_numeric))
            else:
                time_value = round(float(time_numeric), 4)

        return seats_value, time_value, seats_style, time_style

    @app.callback(
        Output("scenario-records-store", "data", allow_duplicate=True),
        Input("scenario-baseline-seats", "value"),
        Input("scenario-baseline-time-to-departure", "value"),
        State("scenario-records-store", "data"),
        prevent_initial_call=True,
    )
    def apply_scenario_baseline_overrides(
        seats_value: Optional[float],
        time_to_departure_value: Optional[float],
        records: Optional[List[Dict[str, object]]],
    ):
        """Persist baseline override edits to the stored scenario records.

        When the user modifies either baseline input, the stored records must
        be updated so that downstream components—such as the prediction graph
        or bid table—operate on the adjusted values.  This function clones and
        updates the records only when the change is meaningful, preventing
        redundant callback updates.
        """
        if not records:
            return no_update

        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )
        if triggered not in {
            "scenario-baseline-seats",
            "scenario-baseline-time-to-departure",
        }:
            return no_update

        seats_override = safe_float(seats_value)
        time_override = safe_float(time_to_departure_value)

        updated_records: List[Dict[str, object]] = []
        changed = False

        for record in records:
            updated = dict(record)
            if seats_override is not None:
                existing_seats = safe_float(updated.get("seats_available"))
                if (
                    existing_seats is None
                    or abs(existing_seats - seats_override) > 1e-6
                ):
                    changed = True
                updated["seats_available"] = seats_value

            current_value = updated.get("current_timestamp")
            if isinstance(current_value, pd.Timestamp):
                updated["current_timestamp"] = current_value.isoformat()

            if time_override is not None:
                departure_raw = updated.get("departure_timestamp")
                if departure_raw in (None, ""):
                    updated_records.append(updated)
                    continue
                departure_ts = pd.to_datetime(departure_raw, errors="coerce")
                if pd.isna(departure_ts):
                    updated_records.append(updated)
                    continue
                new_current_ts = departure_ts - pd.to_timedelta(
                    time_override, unit="hour"
                )
                new_iso = new_current_ts.isoformat()
                existing_current = updated.get("current_timestamp")
                if isinstance(existing_current, pd.Timestamp):
                    existing_iso = existing_current.isoformat()
                elif existing_current in (None, ""):
                    existing_iso = existing_current
                else:
                    existing_iso = str(existing_current)
                if existing_iso != new_iso:
                    changed = True
                updated["current_timestamp"] = new_iso

            updated_records.append(updated)

        if not changed:
            for original, updated in zip(records, updated_records):
                if original != updated:
                    changed = True
                    break

        if not changed:
            return no_update

        return updated_records


__all__ = ["register_feature_callbacks"]
