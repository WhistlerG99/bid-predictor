"""Layout for the feature sensitivity (scenario) tab."""
from __future__ import annotations

from dash import dash_table, dcc, html


def _build_control_card() -> html.Div:
    """Create the sidebar containing all scenario controls and filters.

    This card wires up the cascading dropdowns, baseline overrides, and range
    inputs that drive the sensitivity analysis.  The visual styling mirrors the
    snapshot tab so users feel oriented while switching between workflows.
    """
    return html.Div(
        [
            html.H3(
                "Scenario controls",
                style={"margin": "0 0 1rem 0", "color": "#1b4965"},
            ),
            html.Div(
                [
                    html.Button(
                        "Random flight",
                        id="scenario-random-selection-button",
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
                        id="scenario-selection-history-dropdown",
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
                        id="scenario-carrier-dropdown",
                        options=[],
                        placeholder="Select carrier",
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Flight number", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="scenario-flight-number-dropdown",
                        options=[],
                        placeholder="Select flight",
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Travel date", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id="scenario-travel-date-dropdown",
                        options=[],
                        placeholder="Select travel date",
                        style={"width": "100%"},
                    ),
                ],
                style={"marginBottom": "0.75rem"},
            ),
            html.Label("Upgrade type", style={"fontWeight": "600"}),
            dcc.Dropdown(
                id="scenario-upgrade-dropdown",
                options=[],
                placeholder="Select upgrade",
                style={"width": "100%", "marginBottom": "0.75rem"},
            ),
            html.Div(
                id="scenario-snapshot-label",
                style={
                    "marginBottom": "0.75rem",
                    "color": "#16324f",
                    "fontStyle": "italic",
                },
            ),
            html.Label("Feature to adjust", style={"fontWeight": "600"}),
            dcc.Dropdown(
                id="scenario-feature-dropdown",
                options=[],
                placeholder="Select a feature",
                style={"width": "100%", "marginBottom": "0.75rem"},
            ),
            html.Div(
                id="scenario-base-value",
                style={"margin": "0 0 0.75rem 0", "color": "#16324f"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Baseline seats available",
                                style={"fontWeight": "600"},
                            ),
                            dcc.Input(
                                id="scenario-baseline-seats",
                                type="number",
                                min=0,
                                step=1,
                                value=None,
                                style={
                                    "width": "100%",
                                    "marginTop": "0.35rem",
                                    "borderRadius": "6px",
                                    "border": "1px solid #cbd5e1",
                                    "padding": "0.4rem",
                                },
                            ),
                        ],
                        id="scenario-baseline-seats-container",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Baseline time to departure (hours)",
                                style={"fontWeight": "600"},
                            ),
                            dcc.Input(
                                id="scenario-baseline-time-to-departure",
                                type="number",
                                min=0,
                                step=0.5,
                                value=None,
                                style={
                                    "width": "100%",
                                    "marginTop": "0.35rem",
                                    "borderRadius": "6px",
                                    "border": "1px solid #cbd5e1",
                                    "padding": "0.4rem",
                                },
                            ),
                        ],
                        id="scenario-baseline-time-container",
                        style={"display": "none"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "0.75rem",
                    "alignItems": "center",
                },
            ),
            html.Div(
                [
                    html.Label("Range minimum", style={"fontWeight": "600"}),
                    dcc.Input(
                        id="scenario-range-min",
                        type="number",
                        value=None,
                        style={
                            "width": "100%",
                            "marginTop": "0.35rem",
                            "borderRadius": "6px",
                            "border": "1px solid #cbd5e1",
                            "padding": "0.4rem",
                        },
                    ),
                ],
                style={"marginTop": "0.75rem"},
            ),
            html.Div(
                [
                    html.Label("Range maximum", style={"fontWeight": "600"}),
                    dcc.Input(
                        id="scenario-range-max",
                        type="number",
                        value=None,
                        style={
                            "width": "100%",
                            "marginTop": "0.35rem",
                            "borderRadius": "6px",
                            "border": "1px solid #cbd5e1",
                            "padding": "0.4rem",
                        },
                    ),
                ],
                style={"marginTop": "0.75rem"},
            ),
            html.Div(
                id="scenario-range-feedback",
                style={
                    "fontSize": "0.85rem",
                    "color": "#16324f",
                    "marginTop": "0.35rem",
                },
            ),
            html.Label(
                "Number of evaluation points",
                style={"fontWeight": "600", "marginTop": "0.75rem"},
            ),
            dcc.Input(
                id="scenario-step-count",
                type="number",
                min=2,
                max=200,
                step=1,
                value=25,
                style={
                    "width": "100%",
                    "marginTop": "0.5rem",
                    "borderRadius": "6px",
                    "border": "1px solid #cbd5e1",
                    "padding": "0.4rem",
                },
            ),
            html.Div(
                id="scenario-control-warning",
                className="status-message",
                style={"marginTop": "0.75rem"},
            ),
        ],
        style={
            "flex": "0 0 320px",
            "maxWidth": "340px",
            "backgroundColor": "#ffffff",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1.25rem",
            "alignSelf": "flex-start",
        },
    )


def _build_graph_card() -> html.Div:
    """Construct the card that houses the feature sensitivity line chart.

    The container reserves ample vertical space for the plotly figure and adds
    a status message area underneath so prediction warnings surface directly
    beneath the chart.
    """
    return html.Div(
        [
            dcc.Graph(id="scenario-graph", style={"height": "620px"}),
            html.Div(
                id="scenario-warning",
                className="status-message",
                style={"marginTop": "0.75rem"},
            ),
        ],
        style={
            "backgroundColor": "#edf2fb",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
            "padding": "1.25rem",
        },
    )


def _build_table_card() -> html.Div:
    """Construct the interactive bid table card for the scenario tab.

    Besides the data table itself this card includes quick actions for adding,
    deleting, and restoring bids as well as dropdowns that support bulk
    selection when the table grows large.
    """
    button_style = {
        "color": "white",
        "border": "none",
        "padding": "0.5rem 1rem",
        "borderRadius": "6px",
    }
    return html.Div(
        [
            html.Div(
                [
                    html.Button(
                        "Add bid",
                        id="scenario-add-bid",
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
                        id="scenario-delete-bid",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#e71d36",
                            "boxShadow": "0 2px 6px rgba(231, 29, 54, 0.4)",
                        },
                    ),
                    html.Button(
                        "Restore bids",
                        id="scenario-restore-bid",
                        n_clicks=0,
                        style={
                            **button_style,
                            "backgroundColor": "#1b4965",
                            "marginLeft": "0.5rem",
                            "boxShadow": "0 2px 6px rgba(27, 73, 101, 0.35)",
                        },
                    ),
                    html.Button(
                        "Restore defaults",
                        id="scenario-restore-baseline",
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
                style={"marginBottom": "0.75rem", "display": "flex", "flexWrap": "wrap"},
            ),
            dcc.Dropdown(
                id="scenario-bid-delete-selector",
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
                id="scenario-bid-restore-selector",
                options=[],
                value=[],
                multi=True,
                placeholder="Select removed bids to restore",
                style={
                    "marginBottom": "0.75rem",
                    "backgroundColor": "#ffffff",
                },
            ),
            html.Div(
                id="scenario-table-feedback",
                className="status-message",
                style={"marginBottom": "0.75rem"},
            ),
            dash_table.DataTable(
                id="scenario-bid-table",
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


def build_feature_sensitivity_tab() -> dcc.Tab:
    """Construct the feature sensitivity tab."""
    return dcc.Tab(
        label="Feature sensitivity",
        value="sensitivity",
        children=[
            html.Div(
                [
                    _build_control_card(),
                    html.Div(
                        [
                            _build_graph_card(),
                            _build_table_card(),
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "1.5rem",
                        },
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


__all__ = ["build_feature_sensitivity_tab"]
