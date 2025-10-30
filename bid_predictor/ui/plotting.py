"""Plotting helpers for the Dash UI."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly import colors as plotly_colors

BAR_COLOR_SEQUENCE = (
    getattr(plotly_colors.qualitative, "G10", None)
    or getattr(plotly_colors.qualitative, "Plotly", None)
    or [
        "#006d77",
        "#ff7f50",
        "#6a4c93",
        "#4361ee",
        "#f4a261",
        "#2a9d8f",
        "#e63946",
        "#8338ec",
        "#ffbe0b",
        "#3a86ff",
    ]
)


def build_prediction_plot(df: pd.DataFrame) -> go.Figure:
    """Create a grouped bar chart showing acceptance probability over time.

    The plot aggregates records by bid, sorts each series by time to departure
    (deriving the column when necessary) and overlays an optional line trace for
    seats available.  Colour palettes are chosen to provide consistent styling
    across the UI, and a descriptive hover tooltip is constructed so analysts
    can see the bid number, snapshot identifier and probability for each bar.
    When no predictions are present an empty figure with an explanatory title is
    returned to keep the layout stable.
    """

    fig = go.Figure()
    if df.empty or "Acceptance Probability" not in df.columns:
        fig.update_layout(title="No predictions available", template="plotly_white")
        return fig

    work = df.copy()
    if "Bid #" not in work.columns and "bid_number" in work.columns:
        work["Bid #"] = work["bid_number"]
    if "departure_timestamp" in work.columns and "current_timestamp" in work.columns:
        work["time_until_departure_hours"] = (
            (pd.to_datetime(work["departure_timestamp"]) - pd.to_datetime(work["current_timestamp"]))
            .dt.total_seconds()
            .div(3600)
            .round()
        )
    elif "snapshot_num" in work.columns:
        work["time_until_departure_hours"] = work["snapshot_num"]
    else:
        work["time_until_departure_hours"] = range(len(work))

    if "Bid #" not in work.columns:
        work["Bid #"] = range(1, len(work) + 1)
    if "offer_status" not in work.columns:
        work["offer_status"] = "unknown"

    status_palette = {
        "accepted": "#2ec4b6",
        "rejected": "#ff6b6b",
        "pending": "#ffd166",
        "unknown": "#5e60ce",
    }

    for color_index, (bid_id, grp) in enumerate(work.groupby("Bid #")):
        grp_sorted = grp.sort_values("time_until_departure_hours")
        status = grp_sorted["offer_status"].iloc[-1]
        label = f"Bid {bid_id} - {status}"
        marker_color = BAR_COLOR_SEQUENCE[color_index % len(BAR_COLOR_SEQUENCE)]
        border_color = status_palette.get(str(status).lower(), "#1b4965")
        snapshot_data = None
        if "snapshot_num" in grp_sorted.columns:
            snapshot_data = grp_sorted["snapshot_num"].astype(str)

        bid_values = (
            pd.Series([bid_id] * len(grp_sorted), index=grp_sorted.index)
            .astype(str)
        )

        custom_columns = []
        hover_lines = []
        if snapshot_data is not None:
            custom_columns.append(snapshot_data)
            hover_lines.append("Snapshot: %{customdata[0]}")

        custom_columns.append(bid_values)
        bid_custom_index = len(custom_columns) - 1
        hover_lines.append(f"Bid #: %{{customdata[{bid_custom_index}]}}")
        hover_lines.extend(["Time: %{x}", "Probability: %{y:.4f}%"])
        hover_template = "<br>".join(hover_lines)

        customdata = pd.concat(custom_columns, axis=1).to_numpy()
        fig.add_trace(
            go.Bar(
                x=grp_sorted["time_until_departure_hours"],
                y=grp_sorted["Acceptance Probability"],
                name=label,
                marker=dict(color=marker_color, line=dict(color=border_color, width=1.5)),
                customdata=customdata,
                hovertemplate=hover_template + "<extra></extra>",
            )
        )

    if "seats_available" in work.columns:
        seats = (
            work[["time_until_departure_hours", "seats_available"]]
            .drop_duplicates()
            .sort_values("time_until_departure_hours")
        )
        fig.add_trace(
            go.Scatter(
                x=seats["time_until_departure_hours"],
                y=seats["seats_available"],
                name="Seats available",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#FF5733"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Acceptance probability by snapshot",
        xaxis_title="Time until departure (hours or snapshot)",
        yaxis=dict(title="Acceptance probability (%)", rangemode="tozero"),
        legend=dict(
            title="Bid and status",
            orientation="v",
            yanchor="top",
            y=1,
            x=1.02,
            xanchor="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
        margin=dict(r=220),
        height=760,
    )
    if "seats_available" in work.columns:
        fig.update_layout(yaxis2=dict(title="Seats available", overlaying="y", side="right"))
    return fig


__all__ = ["BAR_COLOR_SEQUENCE", "build_prediction_plot"]
