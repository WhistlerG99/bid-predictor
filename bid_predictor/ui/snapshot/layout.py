"""Layout components for the snapshot explorer tab."""
from __future__ import annotations

from dash import dash_table, dcc, html


def _build_snapshot_filter_card() -> html.Div:
    """Compose the filter sidebar shown on the snapshot explorer tab.

    The sidebar guides the user through the carrier → flight → travel date
    cascade and surfaces snapshot-specific controls for tweaking seats,
    offers, and time-before-departure inputs.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Button(
                        "Random flight",
                        id="random-selection-button",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "backgroundColor": "#1b4965",
                            "color": "white",
                            "border": "none",
                            "padding": "0.6rem",
                            "borderRadius": "6px",
                        },
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Selection history", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="selection-history-dropdown",
                        placeholder="Previously selected flights",
                        options=[],
                        value=None,
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Carrier", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="carrier-dropdown",
                        placeholder="Select carrier",
                        options=[],
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Flight number", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="flight-number-dropdown",
                        placeholder="Select flight",
                        options=[],
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Travel date", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="travel-date-dropdown",
                        placeholder="Select travel date",
                        options=[],
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Upgrade type", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="upgrade-dropdown",
                        placeholder="Select upgrade type",
                        options=[],
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label("Snapshot", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="snapshot-dropdown",
                        placeholder="Select snapshot",
                        options=[],
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.H3(
                        "Selected flight",
                        style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
                    ),
                    html.Div(id="flight-summary", style={"color": "#16324f"}),
                ],
                style={
                    "backgroundColor": "#f4f1de",
                    "borderRadius": "10px",
                    "padding": "0.75rem",
                    "boxShadow": "inset 0 0 0 1px rgba(27, 73, 101, 0.1)",
                    "marginBottom": "1rem",
                },
            ),
            html.Div(
                [
                    html.H4(
                        "Snapshot controls",
                        style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
                    ),
                    html.Label("Seats available", style={"fontWeight": "600"}),
                    dcc.Input(
                        id="seats-available-input",
                        type="number",
                        min=0,
                        style={
                            "width": "100%",
                            "marginBottom": "0.75rem",
                            "borderRadius": "6px",
                            "border": "1px solid #cbd5e1",
                            "padding": "0.4rem",
                        },
                    ),
                    html.Label("Number of offers", style={"fontWeight": "600"}),
                    dcc.Input(
                        id="offers-input",
                        type="number",
                        min=0,
                        step=1,
                        style={
                            "width": "100%",
                            "marginBottom": "0.75rem",
                            "borderRadius": "6px",
                            "border": "1px solid #cbd5e1",
                            "padding": "0.4rem",
                        },
                    ),
                    html.Label(
                        "Time before departure (days / hours)",
                        style={"fontWeight": "600"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="time-before-days-input",
                                type="number",
                                min=0,
                                step=1,
                                placeholder="Days",
                                style={
                                    "width": "48%",
                                    "borderRadius": "6px",
                                    "border": "1px solid #cbd5e1",
                                    "padding": "0.4rem",
                                },
                            ),
                            dcc.Input(
                                id="time-before-hours-input",
                                type="number",
                                min=0,
                                max=23,
                                step=1,
                                placeholder="Hours",
                                style={
                                    "width": "48%",
                                    "borderRadius": "6px",
                                    "border": "1px solid #cbd5e1",
                                    "padding": "0.4rem",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "gap": "4%",
                            "marginTop": "0.5rem",
                            "marginBottom": "0.5rem",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#edf2fb",
                    "borderRadius": "10px",
                    "padding": "0.75rem",
                    "boxShadow": "inset 0 0 0 1px rgba(22, 50, 79, 0.1)",
                    "marginBottom": "1rem",
                },
            ),
            html.Div(
                id="snapshot-feedback",
                className="status-message",
                style={"color": "#16324f"},
            ),
        ],
        style={
            "flex": "0 0 300px",
            "maxWidth": "320px",
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1rem",
            "alignSelf": "flex-start",
        },
    )


def _build_snapshot_graph_card() -> html.Div:
    """Compose the card that displays the prediction trend graph.

    Besides the graph placeholder, the card reserves space for warning messages
    that surface model failures or missing predictions.
    """
    return html.Div(
        [
            html.H3(
                "Acceptance probability trends",
                style={"color": "#1b4965", "margin": "0"},
            ),
            html.Div(
                id="prediction-warning",
                className="status-message",
                style={"marginTop": "0.5rem"},
            ),
            dcc.Graph(
                id="prediction-graph",
                style={"height": "800px", "marginTop": "1rem"},
            ),
        ],
        style={
            "backgroundColor": "#edf2fb",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1.25rem",
            "marginBottom": "1.5rem",
        },
    )


def _build_snapshot_table_card() -> html.Div:
    """Compose the card that hosts table actions and the editable bid grid.

    It consolidates bulk actions, dropdown-driven selections, and the editable
    table so snapshot exploration feels cohesive.
    """
    button_style = {
        "padding": "0.5rem 1rem",
        "borderRadius": "6px",
        "border": "none",
        "color": "white",
    }
    return html.Div(
        [
            html.Div(
                [
                    html.Button(
                        "Add bid",
                        id="add-bid",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#2ec4b6",
                            "marginRight": "0.5rem",
                            "boxShadow": "0 2px 6px rgba(46, 196, 182, 0.4)",
                        },
                    ),
                    html.Button(
                        "Delete selected",
                        id="delete-bid",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#e71d36",
                            "boxShadow": "0 2px 6px rgba(231, 29, 54, 0.4)",
                        },
                    ),
                    html.Button(
                        "Restore bids",
                        id="restore-bid",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#1b4965",
                            "marginLeft": "0.5rem",
                            "boxShadow": "0 2px 6px rgba(27, 73, 101, 0.35)",
                        },
                    ),
                    html.Button(
                        "Restore snapshot",
                        id="restore-snapshot",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#f4a261",
                            "color": "#16324f",
                            "marginLeft": "0.5rem",
                            "boxShadow": "0 2px 6px rgba(244, 162, 97, 0.45)",
                        },
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            dcc.Dropdown(
                id="bid-delete-selector",
                options=[],
                value=[],
                multi=True,
                placeholder="Select bids to delete",
                style={
                    "marginBottom": "0.75rem",
                    "backgroundColor": "#ffffff",
                },
            ),
            dcc.Dropdown(
                id="bid-restore-selector",
                options=[],
                value=[],
                multi=True,
                placeholder="Select removed bids to restore",
                style={
                    "marginBottom": "0.75rem",
                    "backgroundColor": "#ffffff",
                },
            ),
            dash_table.DataTable(
                id="bid-table",
                columns=[],
                data=[],
                editable=True,
                column_selectable="multi",
                style_table={
                    "overflowX": "auto",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
                },
                style_cell={
                    "textAlign": "center",
                    "padding": "0.6rem",
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #f1f5f9",
                },
                style_header={
                    "backgroundColor": "#1b4965",
                    "color": "white",
                    "fontWeight": "700",
                    "textAlign": "center",
                },
                style_data_conditional=[],
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1.25rem",
        },
    )


def build_snapshot_tab() -> dcc.Tab:
    """Create the snapshot explorer tab."""
    return dcc.Tab(
        label="Snapshot explorer",
        value="snapshot",
        children=[
            html.Div(
                [
                    _build_snapshot_filter_card(),
                    html.Div(
                        [
                            _build_snapshot_graph_card(),
                            _build_snapshot_table_card(),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "1.5rem",
                    "alignItems": "flex-start",
                    "flexWrap": "wrap",
                },
            )
        ],
    )


__all__ = ["build_snapshot_tab"]
