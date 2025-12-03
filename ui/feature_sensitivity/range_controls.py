"""Range configuration callbacks for the scenario tab."""
from __future__ import annotations

from typing import Dict, Optional

from dash import Dash, Input, Output, State, html

from ..scenario import (
    ScenarioRange,
    build_feature_options,
    compute_default_range,
    records_to_dataframe,
    select_feature,
)


def register_range_callback(app: Dash) -> None:
    """Register the callback that configures the scenario range inputs."""

    @app.callback(
        Output("scenario-range-min", "value"),
        Output("scenario-range-max", "value"),
        Output("scenario-range-min", "step"),
        Output("scenario-range-max", "step"),
        Output("scenario-range-min", "disabled"),
        Output("scenario-range-max", "disabled"),
        Output("scenario-step-count", "value"),
        Output("scenario-base-value", "children"),
        Output("scenario-range-feedback", "children"),
        Input("scenario-records-store", "data"),
        Input("scenario-feature-dropdown", "value"),
        State("feature-config-store", "data"),
    )
    def configure_scenario_range(
        baseline_records: Optional[list],
        feature_value: Optional[str],
        feature_config: Optional[Dict[str, object]],
    ):
        """Set sensible defaults for the feature range inputs.

        The range sliders should reflect the data distribution of the selected
        baseline feature.  This callback calculates the recommended min/max,
        step size, and evaluation count, while also populating helper text that
        reminds the user of the baseline value they are modifying.
        """
        baseline_df = records_to_dataframe(baseline_records)
        features = build_feature_options(
            baseline_df, feature_config=feature_config
        )
        feature = select_feature(features, feature_value)
        if baseline_df.empty or feature is None:
            return None, None, 1.0, 1.0, True, True, 25, "", ""

        scenario_range: Optional[ScenarioRange] = compute_default_range(baseline_df, feature)
        if scenario_range is None:
            return None, None, 1.0, 1.0, True, True, 25, "Feature is not numeric.", ""

        range_min = float(scenario_range.min_value)
        range_max = float(scenario_range.max_value)
        if range_min == range_max:
            range_max = range_min + (1.0 if feature.is_integer else 0.01)
        step = float(scenario_range.step)
        step = max(step, 1.0 if feature.is_integer else 0.01)
        step_count = max(int(scenario_range.count), 2)
        base_value = scenario_range.base_value
        if feature.is_integer:
            range_min_value = int(round(range_min))
            range_max_value = int(round(range_max))
            baseline_text = f"Baseline value: {int(round(base_value))}"
            helper_text = f"Range: {range_min_value} – {range_max_value}"
        else:
            range_min_value = float(range_min)
            range_max_value = float(range_max)
            baseline_text = f"Baseline value: {base_value:.4f}"
            helper_text = f"Range: {range_min_value:.4f} – {range_max_value:.4f}"
        base_text = html.Div(baseline_text)
        return (
            range_min_value,
            range_max_value,
            step,
            step,
            False,
            False,
            step_count,
            base_text,
            helper_text,
        )


__all__ = ["register_range_callback"]
