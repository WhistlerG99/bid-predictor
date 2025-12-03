"""Scenario graph rendering callbacks."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State

from ..formatting import safe_float
from ..predictions import predict
from ..scenario import (
    TIME_TO_DEPARTURE_SCENARIO_KEY,
    build_adjustment_grid,
    build_feature_options,
    build_scenario_line_chart,
    compute_default_range,
    records_to_dataframe,
    select_feature,
)


def register_graph_callback(app: Dash) -> None:
    """Register the callback that renders the scenario graph."""

    @app.callback(
        Output("scenario-graph", "figure"),
        Output("scenario-warning", "children"),
        Input("scenario-records-store", "data"),
        Input("scenario-feature-dropdown", "value"),
        Input("scenario-range-min", "value"),
        Input("scenario-range-max", "value"),
        Input("scenario-step-count", "value"),
        Input("model-uri-store", "data"),
        Input("scenario-baseline-seats", "value"),
        Input("scenario-baseline-time-to-departure", "value"),
        State("feature-config-store", "data"),
    )
    def render_scenario_graph(
        baseline_records: Optional[list],
        feature_value: Optional[str],
        range_min: Optional[float],
        range_max: Optional[float],
        step_count: Optional[int],
        model_uri: Optional[str],
        baseline_seats: Optional[float],
        baseline_time_to_departure: Optional[float],
        feature_config: Optional[Dict[str, object]],
    ):
        """Render the line chart showing how acceptance changes across a range.

        The callback derives a grid of candidate adjustments from the baseline
        records, feeds it through the model when one is available, and plots
        the resulting acceptance probabilities.  It also computes helpful
        warnings—such as missing models or prediction failures—so the user has
        immediate feedback when the graph cannot be rendered.
        """
        baseline_df = records_to_dataframe(baseline_records)
        features = build_feature_options(baseline_df, feature_config=feature_config)
        feature = select_feature(features, feature_value)

        overrides: Dict[str, float] = {}
        seats_override = safe_float(baseline_seats)
        if seats_override is not None:
            overrides["seats_available"] = float(seats_override)
        time_override = safe_float(baseline_time_to_departure)
        if time_override is not None:
            overrides[TIME_TO_DEPARTURE_SCENARIO_KEY] = float(time_override)

        if feature is not None:
            if feature.kind == "time_to_departure":
                overrides.pop(TIME_TO_DEPARTURE_SCENARIO_KEY, None)
            overrides.pop(feature.key, None)

        if baseline_df.empty or feature is None:
            placeholder = go.Figure()
            placeholder.update_layout(
                template="plotly_white",
                title="Select a flight and feature to explore",
                xaxis_title="Feature value",
                yaxis_title="Acceptance probability (%)",
            )
            return placeholder, "Select a flight, upgrade, and feature to explore."

        default_range = compute_default_range(baseline_df, feature)
        parsed_min = safe_float(range_min)
        parsed_max = safe_float(range_max)
        if parsed_min is None or parsed_max is None:
            if default_range is not None:
                parsed_min = float(default_range.min_value)
                parsed_max = float(default_range.max_value)
            else:
                parsed_min, parsed_max = 0.0, 1.0

        start = float(parsed_min)
        stop = float(parsed_max)
        if start > stop:
            start, stop = stop, start

        count = int(step_count or 25)
        if count < 2:
            count = 2

        scenario_df = build_adjustment_grid(
            baseline_df,
            feature,
            start,
            stop,
            count,
            global_overrides=overrides if overrides else None,
        )
        if scenario_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template="plotly_white",
                title="Unable to construct scenario adjustments",
                xaxis_title=feature.label,
                yaxis_title="Acceptance probability (%)",
            )
            return empty_fig, "Unable to construct scenario adjustments for this feature."

        if not model_uri:
            empty_fig = build_scenario_line_chart(pd.DataFrame(), feature.label)
            return empty_fig, "Load a model to generate acceptance probabilities."

        try:
            prediction_df = predict(
                model_uri,
                scenario_df.copy(),
                feature_config=feature_config,
            )
        except Exception as exc:  # pragma: no cover - user feedback
            error_fig = go.Figure()
            error_fig.update_layout(title=f"Prediction failed: {exc}")
            return error_fig, str(exc)

        figure = build_scenario_line_chart(prediction_df, feature.label)
        warning = prediction_df.attrs.get("model_warning", "") or ""
        return figure, warning


__all__ = ["register_graph_callback"]
