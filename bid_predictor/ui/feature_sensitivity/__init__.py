"""Feature sensitivity tab helpers."""
from __future__ import annotations

from dash import Dash

from .baseline import register_baseline_callback
from .feature_controls import register_feature_callbacks
from .filters import register_filter_callbacks
from .graph import register_graph_callback
from .layout import build_feature_sensitivity_tab
from .range_controls import register_range_callback
from .records import register_record_callbacks
from .table import register_table_callbacks


def register_feature_sensitivity_callbacks(app: Dash) -> None:
    """Register all callbacks for the feature sensitivity tab."""

    register_filter_callbacks(app)
    register_baseline_callback(app)
    register_feature_callbacks(app)
    register_range_callback(app)
    register_graph_callback(app)
    register_table_callbacks(app)
    register_record_callbacks(app)


__all__ = ["build_feature_sensitivity_tab", "register_feature_sensitivity_callbacks"]
