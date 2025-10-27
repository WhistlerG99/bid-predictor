"""Plotting helpers for the Dash UI."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .constants import _BAR_COLOR_SEQUENCE


def _build_prediction_plot(df: pd.DataFrame) -> go.Figure:
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
        marker_color = _BAR_COLOR_SEQUENCE[color_index % len(_BAR_COLOR_SEQUENCE)]
        border_color = status_palette.get(str(status).lower(), "#1b4965")
        snapshot_data: Optional[pd.Series] = None
        hover_template_parts = []
        custom_columns = []
        if "snapshot_num" in grp_sorted.columns:
            snapshot_data = grp_sorted["snapshot_num"].astype(str)
            custom_columns.append(snapshot_data)
            hover_template_parts.append("Snapshot: %{customdata[0]}<br>")
        bid_custom_index = len(custom_columns)
        custom_columns.append(grp_sorted["Bid #"].astype(str))
        hover_template_parts.append(
            f"Bid #: %{{customdata[{bid_custom_index}]}}<br>"
        )
        hover_template_parts.append("Time: %{x}<br>Probability: %{y:.4f}%")
        hover_template = "".join(hover_template_parts)
        custom_data_values = None
        if custom_columns:
            combined = pd.concat(custom_columns, axis=1)
            custom_data_values = combined.to_numpy()
        fig.add_trace(
            go.Bar(
                x=grp_sorted["time_until_departure_hours"],
                y=grp_sorted["Acceptance Probability"],
                name=label,
                marker=dict(color=marker_color, line=dict(color=border_color, width=1.5)),
                customdata=custom_data_values,
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
                mode="lines",
                yaxis="y2",
                line=dict(color="#FF5733", dash="dash"),
                connectgaps=True,
                showlegend=False,
                hoverinfo="skip",
                legendgroup="seats-available",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=seats["time_until_departure_hours"],
                y=seats["seats_available"],
                name="Seats available",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color="#FF5733"),
                connectgaps=False,
                legendgroup="seats-available",
                hovertemplate="Time: %{x}<br>Seats available: %{y}<extra></extra>",
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
        uirevision="prediction-graph",
    )
    if "seats_available" in work.columns:
        fig.update_layout(yaxis2=dict(title="Seats available", overlaying="y", side="right"))
    return fig
