"""Layout components for the acceptance probability explorer tab."""
from __future__ import annotations

from dash import dash_table, dcc, html


def _build_filter_card() -> html.Div:
    """Compose the filter controls for selecting flights and snapshots."""

    return html.Div(
        [
            html.Div(
                [
                    html.Label("Carrier", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="acceptance-carrier-dropdown",
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
                        id="acceptance-flight-number-dropdown",
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
                        id="acceptance-travel-date-dropdown",
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
                        id="acceptance-upgrade-dropdown",
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
                        id="acceptance-snapshot-dropdown",
                        placeholder="Select snapshot",
                        options=[],
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Label(
                        "Departure timestamp", style={"fontWeight": "600"}
                    ),
                    html.Div(
                        id="acceptance-departure-timestamp",
                        style={"padding": "0.4rem 0", "color": "#16324f"},
                        children="–",
                    ),
                    html.Label("Origination", style={"fontWeight": "600"}),
                    html.Div(
                        id="acceptance-origination-code",
                        style={"padding": "0.4rem 0", "color": "#16324f"},
                        children="–",
                    ),
                    html.Label("Destination", style={"fontWeight": "600"}),
                    html.Div(
                        id="acceptance-destination-code",
                        style={"padding": "0.4rem 0", "color": "#16324f"},
                        children="–",
                    ),
                ],
                style={
                    "backgroundColor": "#f8fafc",
                    "borderRadius": "10px",
                    "padding": "0.75rem",
                    "boxShadow": "inset 0 0 0 1px rgba(27, 73, 101, 0.06)",
                    "marginBottom": "1rem",
                },
            ),
            html.Div(
                [
                    html.H3(
                        "Selected flight",
                        style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
                    ),
                    html.Div(
                        id="acceptance-flight-summary", style={"color": "#16324f"}
                    ),
                ],
                style={
                    "backgroundColor": "#f4f1de",
                    "borderRadius": "10px",
                    "padding": "0.75rem",
                    "boxShadow": "inset 0 0 0 1px rgba(27, 73, 101, 0.1)",
                    "marginBottom": "1rem",
                },
            ),
        ],
        style={
            "flex": "0 0 320px",
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1rem",
        },
    )


def _build_graph_card() -> html.Div:
    """Create the card that renders the acceptance probability graph."""

    return html.Div(
        [
            html.H2(
                "Acceptance probability by snapshot",
                style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
            ),
            dcc.Graph(
                id="acceptance-prediction-graph",
                figure={},
                style={"height": "760px"},
            ),
            html.Div(
                id="acceptance-warning",
                className="status-message",
                style={"marginTop": "0.5rem", "color": "#c1121f"},
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1.25rem",
        },
    )


def _build_table_card() -> html.Div:
    """Create the read-only bid table card."""

    return html.Div(
        [
            html.H2(
                "Bids in selected snapshot",
                style={"margin": "0 0 0.5rem 0", "color": "#1b4965"},
            ),
            html.Div(
                id="acceptance-table-feedback",
                className="status-message",
                style={"marginBottom": "0.5rem", "color": "#c1121f"},
            ),
            dash_table.DataTable(
                id="acceptance-bid-table",
                columns=[{"name": "Feature", "id": "Feature", "editable": False}],
                data=[],
                style_data={
                    "whiteSpace": "normal",
                    "height": "auto",
                    "lineHeight": "1.3em",
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
                cell_selectable=False,
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1.25rem",
        },
    )


def build_acceptance_tab() -> dcc.Tab:
    """Create the acceptance probability explorer tab."""

    return dcc.Tab(
        label="Acceptance explorer",
        value="acceptance",
        children=[
            html.Div(
                [
                    _build_filter_card(),
                    html.Div(
                        [
                            _build_graph_card(),
                            _build_table_card(),
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


__all__ = ["build_acceptance_tab"]
