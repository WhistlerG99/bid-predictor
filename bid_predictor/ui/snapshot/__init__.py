"""Snapshot explorer tab helpers."""
from __future__ import annotations

from dash import Dash

from .layout import build_snapshot_tab
from .filters import register_filter_callbacks
from .predictions import register_prediction_callbacks
from .summary import register_summary_callbacks
from .table import register_table_callbacks
from .view import register_snapshot_view_callbacks


def register_snapshot_callbacks(app: Dash) -> None:
    """Register all snapshot explorer callbacks."""

    register_filter_callbacks(app)
    register_snapshot_view_callbacks(app)
    register_summary_callbacks(app)
    register_table_callbacks(app)
    register_prediction_callbacks(app)


__all__ = ["build_snapshot_tab", "register_snapshot_callbacks"]
