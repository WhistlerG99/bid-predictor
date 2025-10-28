"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

from typing import Dict, List, Optional
from uuid import uuid4

import mlflow
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from mlflow.exceptions import MlflowException
import plotly.graph_objects as go

from bid_predictor.ui import (
    BAR_COLOR_SEQUENCE,
    BID_IDENTIFIER_COLUMNS,
    DISPLAY_FEATURE_ROWS,
    ScenarioFeature,
    ScenarioRange,
    USD_MAX_COLUMN,
    USD_PERCENT_COLUMNS,
    apply_bid_labels,
    apply_table_edits,
    build_prediction_plot,
    build_bid_table,
    build_adjustment_grid,
    build_carrier_options,
    build_feature_options,
    build_flight_number_options,
    build_scenario_line_chart,
    build_travel_date_options,
    build_upgrade_options,
    compute_default_range,
    compute_bid_label_map,
    extract_baseline_snapshot,
    get_next_bid_label,
    load_dataset_cached,
    load_model_cached,
    normalize_offer_time,
    prepare_bid_record,
    prepare_prediction_dataframe,
    predict,
    recompute_usd_metrics,
    select_feature,
    records_to_dataframe,
    safe_float,
    sort_records_by_bid,
)


# -- Dash application --------------------------------------------------------------------------


def create_app() -> Dash:
    default_dataset_path = "./data/air_canada_and_lot/evaluation_sets/eval_bid_data_snapshots_v2.parquet"

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Bid Predictor Playground", style={"margin": "0", "color": "#1b4965"}),
                    html.P(
                        "Load a dataset snapshot and an MLflow-registered model to explore acceptance probabilities.",
                        style={"margin": "0", "color": "#16324f"},
                    ),
                ],
                style={
                    "background": "linear-gradient(90deg, #e0fbfc 0%, #c2dfe3 100%)",
                    "padding": "1.5rem",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                    "marginBottom": "1.5rem",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Dataset path", style={"fontWeight": "600"}),
                            dcc.Input(
                                id="dataset-path",
                                type="text",
                                value=default_dataset_path,
                                placeholder="Path to bid_data_snapshots_v2.parquet",
                                style={"width": "100%", "marginBottom": "0.5rem"},
                            ),
                            html.Button(
                                "Load dataset",
                                id="load-dataset",
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
                            html.Div(id="dataset-status", className="status-message", style={"marginTop": "0.5rem"}),
                        ],
                        style={
                            "flex": "1",
                            "padding": "1rem",
                            "backgroundColor": "#f7fff7",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("MLflow tracking URI", style={"fontWeight": "600"}),
                            dcc.Input(
                                id="mlflow-tracking-uri",
                                type="text",
                                value=mlflow.get_tracking_uri(),
                                placeholder="http://localhost:5000",
                                style={"width": "100%", "marginBottom": "0.5rem"},
                            ),
                            html.Label("Model name", style={"fontWeight": "600"}),
                            dcc.Input(
                                id="model-name",
                                type="text",
                                placeholder="Registered model name",
                                style={"width": "100%", "marginBottom": "0.5rem"},
                            ),
                            html.Label("Model stage or version", style={"fontWeight": "600"}),
                            dcc.Input(
                                id="model-stage",
                                type="text",
                                placeholder="e.g. Production or 5",
                                style={"width": "100%", "marginBottom": "0.5rem"},
                            ),
                            html.Button(
                                "Load model",
                                id="load-model",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "backgroundColor": "#ff6b6b",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "0.6rem",
                                    "borderRadius": "6px",
                                },
                            ),
                            html.Div(id="model-status", className="status-message", style={"marginTop": "0.5rem"}),
                        ],
                        style={
                            "flex": "1",
                            "padding": "1rem",
                            "backgroundColor": "#f7fff7",
                            "borderRadius": "12px",
                            "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)",
                        },
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "1.5rem", "marginBottom": "1.5rem"},
            ),
            dcc.Store(id="dataset-path-store"),
            dcc.Store(id="model-uri-store"),
            dcc.Store(id="bid-records-store"),
            dcc.Store(id="snapshot-meta-store"),
            dcc.Store(id="prediction-store"),
            dcc.Store(id="removed-bids-store"),
            dcc.Store(id="baseline-bid-records-store"),
            dcc.Store(id="baseline-snapshot-meta-store"),
            dcc.Store(id="scenario-records-store"),
            dcc.Store(id="scenario-original-records-store"),
            dcc.Store(id="scenario-removed-bids-store"),
            dcc.Tabs(
                id="main-tabs",
                value="snapshot",
                children=[
                    dcc.Tab(
                        label="Snapshot explorer",
                        value="snapshot",
                        children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
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
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
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
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Button(
                                                                            "Add bid",
                                                                            id="add-bid",
                                                                            n_clicks=0,
                                                                            style={
                                                                                "backgroundColor": "#2ec4b6",
                                                                                "color": "white",
                                                                                "border": "none",
                                                                                "padding": "0.5rem 1rem",
                                                                                "borderRadius": "6px",
                                                                                "marginRight": "0.5rem",
                                                                                "boxShadow": "0 2px 6px rgba(46, 196, 182, 0.4)",
                                                                            },
                                                                        ),
                                                                        html.Button(
                                                                            "Delete selected",
                                                                            id="delete-bid",
                                                                            n_clicks=0,
                                                                            style={
                                                                                "backgroundColor": "#e71d36",
                                                                                "color": "white",
                                                                                "border": "none",
                                                                                "padding": "0.5rem 1rem",
                                                                                "borderRadius": "6px",
                                                                                "boxShadow": "0 2px 6px rgba(231, 29, 54, 0.4)",
                                                                            },
                                                                        ),
                                                                        html.Button(
                                                                            "Restore bids",
                                                                            id="restore-bid",
                                                                            n_clicks=0,
                                                                            style={
                                                                                "backgroundColor": "#1b4965",
                                                                                "color": "white",
                                                                                "border": "none",
                                                                                "padding": "0.5rem 1rem",
                                                                                "borderRadius": "6px",
                                                                                "marginLeft": "0.5rem",
                                                                                "boxShadow": "0 2px 6px rgba(27, 73, 101, 0.35)",
                                                                            },
                                                                        ),
                                                                        html.Button(
                                                                            "Restore snapshot",
                                                                            id="restore-snapshot",
                                                                            n_clicks=0,
                                                                            style={
                                                                                "backgroundColor": "#f4a261",
                                                                                "color": "#16324f",
                                                                                "border": "none",
                                                                                "padding": "0.5rem 1rem",
                                                                                "borderRadius": "6px",
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
                                                        ),
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
                                        ),                        ],
                    ),
                    dcc.Tab(
                        label="Feature sensitivity",
                        value="scenario",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3(
                                                "Scenario controls",
                                                style={"margin": "0 0 1rem 0", "color": "#1b4965"},
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
                                                    html.Label(
                                                        "Flight number",
                                                        style={"fontWeight": "600"},
                                                    ),
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
                                            html.Label("Feature range", style={"fontWeight": "600"}),
                                            html.Div(
                                                [
                                                    dcc.Input(
                                                        id="scenario-range-min",
                                                        type="number",
                                                        value=None,
                                                        debounce=False,
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "0",
                                                            "padding": "0.4rem",
                                                            "borderRadius": "6px",
                                                            "border": "1px solid #cbd5e1",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="scenario-range-max",
                                                        type="number",
                                                        value=None,
                                                        debounce=False,
                                                        style={
                                                            "flex": "1",
                                                            "minWidth": "0",
                                                            "padding": "0.4rem",
                                                            "borderRadius": "6px",
                                                            "border": "1px solid #cbd5e1",
                                                        },
                                                    ),
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "gap": "0.75rem",
                                                    "alignItems": "center",
                                                },
                                            ),
                                            html.Div(
                                                id="scenario-range-feedback",
                                                style={
                                                    "fontSize": "0.85rem",
                                                    "color": "#16324f",
                                                    "marginTop": "0.35rem",
                                                },
                                            ),
                                            html.Div(
                                                [
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
                                                ],
                                            ),
                                            html.Div(
                                                id="scenario-base-value",
                                                style={"marginTop": "0.75rem", "color": "#16324f"},
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
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Graph(
                                                        id="scenario-graph",
                                                        style={"height": "620px"},
                                                    ),
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
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "Add bid",
                                                                id="scenario-add-bid",
                                                                n_clicks=0,
                                                                style={
                                                                    "backgroundColor": "#2ec4b6",
                                                                    "color": "white",
                                                                    "border": "none",
                                                                    "padding": "0.5rem 1rem",
                                                                    "borderRadius": "6px",
                                                                    "marginRight": "0.5rem",
                                                                    "boxShadow": "0 2px 6px rgba(46, 196, 182, 0.4)",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "Delete selected",
                                                                id="scenario-delete-bid",
                                                                n_clicks=0,
                                                                style={
                                                                    "backgroundColor": "#e71d36",
                                                                    "color": "white",
                                                                    "border": "none",
                                                                    "padding": "0.5rem 1rem",
                                                                    "borderRadius": "6px",
                                                                    "boxShadow": "0 2px 6px rgba(231, 29, 54, 0.4)",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "Restore bids",
                                                                id="scenario-restore-bid",
                                                                n_clicks=0,
                                                                style={
                                                                    "backgroundColor": "#1b4965",
                                                                    "color": "white",
                                                                    "border": "none",
                                                                    "padding": "0.5rem 1rem",
                                                                    "borderRadius": "6px",
                                                                    "marginLeft": "0.5rem",
                                                                    "boxShadow": "0 2px 6px rgba(27, 73, 101, 0.35)",
                                                                },
                                                            ),
                                                            html.Button(
                                                                "Restore defaults",
                                                                id="scenario-restore-baseline",
                                                                n_clicks=0,
                                                                style={
                                                                    "backgroundColor": "#f4a261",
                                                                    "color": "#16324f",
                                                                    "border": "none",
                                                                    "padding": "0.5rem 1rem",
                                                                    "borderRadius": "6px",
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
                                            ),
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
                            ),
                        ],
                    ),
                ],
                style={"marginTop": "1rem"},
            ),
        ],
        style={"fontFamily": "'Segoe UI', sans-serif", "backgroundColor": "#fafafa", "padding": "1.5rem"},
    )

    # Callbacks ---------------------------------------------------------------------------------

    @app.callback(
        Output("dataset-status", "children"),
        Output("dataset-path-store", "data"),
        Input("load-dataset", "n_clicks"),
        State("dataset-path", "value"),
        prevent_initial_call=True,
    )
    def load_dataset(n_clicks: int, path: str):
        if not path:
            return "Please provide a dataset path.", None

        try:
            dataset = load_dataset_cached(path)
        except Exception as exc:  # pragma: no cover - user feedback
            return f"Failed to load dataset: {exc}", None

        status = f"Loaded dataset with {len(dataset):,} rows."
        return status, path

    @app.callback(
        Output("model-status", "children"),
        Output("model-uri-store", "data"),
        Input("load-model", "n_clicks"),
        State("mlflow-tracking-uri", "value"),
        State("model-name", "value"),
        State("model-stage", "value"),
        prevent_initial_call=True,
    )
    def load_model(n_clicks: int, tracking_uri: str, model_name: str, stage_or_version: str):
        if not model_name:
            return "Please enter a registered model name.", None

        mlflow.set_tracking_uri(tracking_uri or mlflow.get_tracking_uri())
        model_uri: Optional[str] = None
        try:
            if stage_or_version:
                stage_or_version = stage_or_version.strip()
                if stage_or_version.isdigit():
                    model_uri = f"models:/{model_name}/{stage_or_version}"
                else:
                    model_uri = f"models:/{model_name}/{stage_or_version}"
            else:
                # Default to production stage if nothing provided.
                model_uri = f"models:/{model_name}/Production"
            load_model_cached(model_uri)
        except MlflowException as exc:  # pragma: no cover - user feedback
            return f"Failed to load model: {exc}", None
        except Exception as exc:  # pragma: no cover
            return f"Unexpected error while loading model: {exc}", None

        return f"Loaded model from {model_uri}", model_uri

    @app.callback(
        Output("scenario-carrier-dropdown", "options"),
        Output("scenario-carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
    )
    def populate_scenario_carriers(dataset_path: Optional[str]):
        if not dataset_path:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_carrier_options(dataset)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-flight-number-dropdown", "options"),
        Output("scenario-flight-number-dropdown", "value"),
        Input("scenario-carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_scenario_flight_numbers(
        carrier: Optional[str], dataset_path: Optional[str]
    ):
        if not dataset_path or not carrier:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_flight_number_options(dataset, carrier)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-travel-date-dropdown", "options"),
        Output("scenario-travel-date-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        State("scenario-carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_scenario_travel_dates(
        flight_number: Optional[str],
        carrier: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_travel_date_options(dataset, carrier, flight_number)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-upgrade-dropdown", "options"),
        Output("scenario-upgrade-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        State("scenario-carrier-dropdown", "value"),
        State("scenario-flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_scenario_upgrades(
        travel_date: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception:
            return [], None

        options = build_upgrade_options(dataset, carrier, flight_number, travel_date)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("scenario-records-store", "data"),
        Output("scenario-original-records-store", "data"),
        Output("scenario-removed-bids-store", "data"),
        Output("scenario-snapshot-label", "children"),
        Output("scenario-control-warning", "children"),
        Input("scenario-carrier-dropdown", "value"),
        Input("scenario-flight-number-dropdown", "value"),
        Input("scenario-travel-date-dropdown", "value"),
        Input("scenario-upgrade-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def update_scenario_baseline(
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path:
            warning = "Load a dataset to explore scenarios."
            return None, None, [], warning, warning

        if not carrier or not flight_number or not travel_date or not upgrade_type:
            return None, None, [], "", "Select a flight and upgrade type."

        try:
            dataset = load_dataset_cached(dataset_path)
        except Exception as exc:
            return None, None, [], "", f"Failed to read dataset: {exc}"

        baseline_df, snapshot_label = extract_baseline_snapshot(
            dataset,
            carrier,
            flight_number,
            travel_date,
            upgrade_type,
        )
        if baseline_df.empty:
            return None, None, [], "No bids found for this selection.", "No bids are available for the chosen flight."

        base_records = [prepare_bid_record(record) for record in baseline_df.to_dict("records")]
        base_records = sort_records_by_bid(base_records)
        recompute_usd_metrics(base_records)

        serializable_records: List[Dict[str, object]] = []
        for record in base_records:
            converted: Dict[str, object] = {}
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    converted[key] = value.strftime("%Y-%m-%dT%H:%M:%S")
                elif hasattr(value, "isoformat") and not isinstance(value, (str, bytes, int, float, bool)):
                    try:
                        converted[key] = value.isoformat()  # type: ignore[attr-defined]
                    except Exception:
                        converted[key] = value
                else:
                    converted[key] = value
            serializable_records.append(converted)

        summary = f"Using {len(serializable_records)} bids"
        if snapshot_label:
            summary += f" {snapshot_label}"

        original_records = [dict(record) for record in serializable_records]

        return serializable_records, original_records, [], summary, ""

    @app.callback(
        Output("scenario-feature-dropdown", "options"),
        Output("scenario-feature-dropdown", "value"),
        Input("scenario-records-store", "data"),
    )
    def populate_scenario_features(baseline_records: Optional[List[Dict[str, object]]]):
        baseline_df = records_to_dataframe(baseline_records)
        features = build_feature_options(baseline_df)
        options = [{"label": feature.label, "value": feature.encode()} for feature in features]
        value = options[0]["value"] if options else None
        return options, value

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
    )
    def configure_scenario_range(
        baseline_records: Optional[List[Dict[str, object]]],
        feature_value: Optional[str],
    ):
        baseline_df = records_to_dataframe(baseline_records)
        features = build_feature_options(baseline_df)
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

    @app.callback(
        Output("scenario-graph", "figure"),
        Output("scenario-warning", "children"),
        Input("scenario-records-store", "data"),
        Input("scenario-feature-dropdown", "value"),
        Input("scenario-range-min", "value"),
        Input("scenario-range-max", "value"),
        Input("scenario-step-count", "value"),
        Input("model-uri-store", "data"),
    )
    def render_scenario_graph(
        baseline_records: Optional[List[Dict[str, object]]],
        feature_value: Optional[str],
        range_min: Optional[float],
        range_max: Optional[float],
        step_count: Optional[int],
        model_uri: Optional[str],
    ):
        baseline_df = records_to_dataframe(baseline_records)
        features = build_feature_options(baseline_df)
        feature = select_feature(features, feature_value)

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

        scenario_df = build_adjustment_grid(baseline_df, feature, start, stop, count)
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
            prediction_df = predict(model_uri, scenario_df.copy())
        except Exception as exc:  # pragma: no cover - user feedback
            error_fig = go.Figure()
            error_fig.update_layout(title=f"Prediction failed: {exc}")
            return error_fig, str(exc)

        figure = build_scenario_line_chart(prediction_df, feature.label)
        warning = prediction_df.attrs.get("model_warning", "") or ""
        return figure, warning

    @app.callback(
        Output("scenario-bid-table", "columns"),
        Output("scenario-bid-table", "data"),
        Output("scenario-bid-table", "style_data_conditional"),
        Input("scenario-records-store", "data"),
        Input("model-uri-store", "data"),
    )
    def render_scenario_table(
        records: Optional[List[Dict[str, object]]],
        model_uri: Optional[str],
    ):
        predictions: Dict[str, object] = {}
        if records and model_uri:
            feature_df = prepare_prediction_dataframe(records)
            if not feature_df.empty:
                try:
                    pred_df = predict(model_uri, feature_df.copy())
                except Exception:
                    pred_df = pd.DataFrame()
                if not pred_df.empty:
                    for idx, _ in enumerate(records):
                        column_id = f"bid_{idx}"
                        if idx < len(pred_df):
                            value = pred_df.iloc[idx].get("Acceptance Probability")
                            if value is None or pd.isna(value):
                                predictions[column_id] = None
                            else:
                                try:
                                    predictions[column_id] = float(value)
                                except (TypeError, ValueError):
                                    predictions[column_id] = value
        return build_bid_table(records, predictions)

    @app.callback(
        Output("scenario-records-store", "data", allow_duplicate=True),
        Input("scenario-bid-table", "data_timestamp"),
        State("scenario-bid-table", "data"),
        State("scenario-bid-table", "columns"),
        State("scenario-records-store", "data"),
        prevent_initial_call=True,
    )
    def persist_scenario_table_edits(
        data_timestamp: Optional[int],
        table_data: Optional[List[Dict[str, object]]],
        columns: Optional[List[Dict[str, object]]],
        records: Optional[List[Dict[str, object]]],
    ):
        if not data_timestamp or not table_data or not columns or not records:
            return no_update

        updated_records = apply_table_edits(records, table_data, columns)
        if updated_records is None:
            return no_update
        return updated_records

    @app.callback(
        Output("scenario-bid-delete-selector", "options"),
        Output("scenario-bid-delete-selector", "value"),
        Input("scenario-records-store", "data"),
    )
    def sync_scenario_delete_selector(records: Optional[List[Dict[str, object]]]):
        if not records:
            return [], []
        options = [
            {
                "label": f"Bid {record.get('Bid #') or record.get('bid_number') or idx + 1}",
                "value": idx,
            }
            for idx, record in enumerate(records)
        ]
        return options, []

    @app.callback(
        Output("scenario-bid-restore-selector", "options"),
        Output("scenario-bid-restore-selector", "value"),
        Input("scenario-removed-bids-store", "data"),
    )
    def sync_scenario_restore_selector(
        removed: Optional[List[Dict[str, object]]]
    ):
        if not removed:
            return [], []
        options = [
            {"label": item.get("label") or f"Removed bid {idx + 1}", "value": item.get("id")}
            for idx, item in enumerate(removed)
            if item.get("id") is not None
        ]
        return options, []

    @app.callback(
        Output("scenario-records-store", "data", allow_duplicate=True),
        Output("scenario-removed-bids-store", "data", allow_duplicate=True),
        Output("scenario-table-feedback", "children"),
        Input("scenario-add-bid", "n_clicks"),
        Input("scenario-delete-bid", "n_clicks"),
        Input("scenario-restore-bid", "n_clicks"),
        Input("scenario-restore-baseline", "n_clicks"),
        State("scenario-records-store", "data"),
        State("scenario-removed-bids-store", "data"),
        State("scenario-bid-delete-selector", "value"),
        State("scenario-bid-restore-selector", "value"),
        State("scenario-original-records-store", "data"),
        State("scenario-bid-table", "selected_columns"),
        prevent_initial_call=True,
    )
    def update_scenario_records(
        add_clicks: int,
        delete_clicks: int,
        restore_clicks: int,
        restore_baseline_clicks: int,
        records: Optional[List[Dict[str, object]]],
        removed_store: Optional[List[Dict[str, object]]],
        delete_selector: Optional[List[int]],
        restore_selector: Optional[List[str]],
        original_records: Optional[List[Dict[str, object]]],
        selected_columns: Optional[List[str]],
    ):
        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )

        current_records = [dict(record) for record in records or []]
        existing_removed = list(removed_store or [])

        if triggered == "scenario-restore-baseline":
            if not original_records:
                return no_update, no_update, "No defaults available to restore."
            restored = [dict(record) for record in original_records]
            recompute_usd_metrics(restored)
            return restored, [], "Restored default bids."

        if triggered == "scenario-add-bid":
            if not current_records:
                return no_update, no_update, "Load bids before adding new ones."
            base = current_records[0]
            new_bid = {key: base.get(key) for key in base}
            for identifier in BID_IDENTIFIER_COLUMNS:
                if identifier in new_bid:
                    new_bid[identifier] = None
            new_bid["Bid #"] = get_next_bid_label(current_records)
            new_bid.setdefault("offer_status", "pending")
            prepared = prepare_bid_record(new_bid)
            updated = sort_records_by_bid(current_records + [prepared])
            recompute_usd_metrics(updated)
            return updated, existing_removed, ""

        if triggered == "scenario-delete-bid":
            if not current_records:
                return no_update, no_update, "No bids available to delete."
            selections: set[int] = set()
            if selected_columns:
                selections.update(
                    int(col_id.replace("bid_", ""))
                    for col_id in selected_columns
                    if col_id.startswith("bid_") and col_id.replace("bid_", "").isdigit()
                )
            if delete_selector:
                selections.update(int(idx) for idx in delete_selector)
            indices_to_remove = sorted(selections, reverse=True)
            if not indices_to_remove:
                return no_update, existing_removed, "Select bids to delete."
            working = list(current_records)
            removed_entries: List[Dict[str, object]] = []
            for idx in indices_to_remove:
                if 0 <= idx < len(working):
                    removed_record = working.pop(idx)
                    removed_entries.append(
                        {
                            "id": str(uuid4()),
                            "label": f"Bid {removed_record.get('Bid #') or idx + 1}",
                            "record": removed_record,
                        }
                    )
            if not removed_entries:
                return no_update, existing_removed, "No matching bids were removed."
            working = sort_records_by_bid(working)
            recompute_usd_metrics(working)
            updated_removed = existing_removed + removed_entries
            return working, updated_removed, ""

        if triggered == "scenario-restore-bid":
            if not restore_selector:
                return no_update, existing_removed, "Select removed bids to restore."
            restore_ids = set(restore_selector)
            restored_records: List[Dict[str, object]] = []
            remaining_removed: List[Dict[str, object]] = []
            for item in existing_removed:
                if item.get("id") in restore_ids:
                    restored_records.append(prepare_bid_record(item.get("record", {})))
                else:
                    remaining_removed.append(item)
            if not restored_records:
                return no_update, existing_removed, "No matching removed bids found."
            working = sort_records_by_bid(current_records + restored_records)
            recompute_usd_metrics(working)
            return working, remaining_removed, ""

        return no_update, no_update, ""

    @app.callback(
        Output("carrier-dropdown", "options"),
        Output("carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
    )
    def populate_carriers(dataset_path: Optional[str]):
        if not dataset_path:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        carriers = (
            dataset["carrier_code"].dropna().drop_duplicates().sort_values()
            if "carrier_code" in dataset.columns
            else pd.Series(dtype=str)
        )
        options = [{"label": carrier, "value": carrier} for carrier in carriers]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("flight-number-dropdown", "options"),
        Output("flight-number-dropdown", "value"),
        Input("carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_flight_numbers(carrier: Optional[str], dataset_path: Optional[str]):
        if not dataset_path or not carrier:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "carrier_code" not in dataset.columns or "flight_number" not in dataset.columns:
            return [], None

        flights = (
            dataset.loc[dataset["carrier_code"] == carrier, "flight_number"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        options = [{"label": str(flt), "value": str(flt)} for flt in flights]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("travel-date-dropdown", "options"),
        Output("travel-date-dropdown", "value"),
        Input("flight-number-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_travel_dates(
        flight_number: Optional[str], carrier: Optional[str], dataset_path: Optional[str]
    ):
        if not dataset_path or not carrier or not flight_number:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "travel_date" not in dataset.columns:
            return [], None

        mask = (dataset["carrier_code"] == carrier) & (
            dataset["flight_number"].astype(str) == str(flight_number)
        )
        dates = (
            pd.to_datetime(dataset.loc[mask, "travel_date"])
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        options = [
            {"label": dt.date().isoformat(), "value": dt.date().isoformat()} for dt in dates
        ]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("upgrade-dropdown", "options"),
        Output("upgrade-dropdown", "value"),
        Input("travel-date-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_upgrade_types(
        travel_date: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number or not travel_date:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "upgrade_type" not in dataset.columns:
            return [], None

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
        )
        upgrades = (
            dataset.loc[mask, "upgrade_type"].dropna().drop_duplicates().sort_values()
        )
        options = [{"label": upg, "value": upg} for upg in upgrades]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("snapshot-dropdown", "options"),
        Output("snapshot-dropdown", "value"),
        Input("upgrade-dropdown", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def populate_snapshots(
        upgrade_type: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        dataset_path: Optional[str],
    ):
        if not dataset_path or not carrier or not flight_number or not travel_date or not upgrade_type:
            return [], None

        dataset = load_dataset_cached(dataset_path)
        if "snapshot_num" not in dataset.columns:
            return [], None

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        snapshots = (
            dataset.loc[mask, "snapshot_num"].dropna().drop_duplicates().sort_values()
        )
        options = [
            {"label": f"Snapshot {snap}", "value": str(snap)} for snap in snapshots
        ]
        value = options[0]["value"] if options else None
        return options, value


    @app.callback(
        Output("flight-summary", "children"),
        Output("snapshot-feedback", "children"),
        Output("snapshot-meta-store", "data"),
        Output("bid-records-store", "data"),
        Output("removed-bids-store", "data"),
        Output("baseline-bid-records-store", "data"),
        Output("baseline-snapshot-meta-store", "data"),
        Input("snapshot-dropdown", "value"),
        Input("add-bid", "n_clicks"),
        Input("delete-bid", "n_clicks"),
        Input("restore-bid", "n_clicks"),
        Input("restore-snapshot", "n_clicks"),
        State("bid-records-store", "data"),
        State("bid-table", "selected_columns"),
        State("bid-delete-selector", "value"),
        State("bid-restore-selector", "value"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("upgrade-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("snapshot-meta-store", "data"),
        State("removed-bids-store", "data"),
        State("baseline-bid-records-store", "data"),
        State("baseline-snapshot-meta-store", "data"),
    )
    def update_snapshot_view(
        snapshot_value: Optional[str],
        add_clicks: int,
        delete_clicks: int,
        restore_clicks: int,
        restore_snapshot_clicks: int,
        existing_records: Optional[List[Dict[str, str]]],
        selected_columns: Optional[List[str]],
        delete_selector: Optional[List[int]],
        restore_selector: Optional[List[str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
        snapshot_meta: Optional[Dict[str, str]],
        removed_store: Optional[List[Dict[str, object]]],
        baseline_records_store: Optional[List[Dict[str, object]]],
        baseline_meta_store: Optional[Dict[str, object]],
    ):
        triggered = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )
        existing_removed = list(removed_store or [])
        baseline_records = [dict(record) for record in baseline_records_store or []]
        baseline_meta = dict(baseline_meta_store or {})

        if not dataset_path:
            return (
                html.Div("Load a dataset to begin."),
                "",
                None,
                None,
                existing_removed,
                no_update,
                no_update,
            )

        dataset = load_dataset_cached(dataset_path)

        if not carrier or not flight_number or not travel_date or not upgrade_type:
            return (
                html.Div("Select a carrier, flight, travel date, and upgrade type."),
                "",
                snapshot_meta,
                existing_records,
                existing_removed,
                no_update,
                no_update,
            )

        travel_date_dt = pd.to_datetime(travel_date).date()

        summary_block = html.Ul(
            [
                html.Li(f"Carrier: {carrier}"),
                html.Li(f"Flight: {flight_number}"),
                html.Li(f"Travel date: {travel_date}"),
                html.Li(f"Upgrade type: {upgrade_type}"),
            ],
            style={"paddingLeft": "1.2rem", "margin": "0"},
        )

        if triggered == "restore-snapshot":
            if not baseline_records:
                return (
                    summary_block,
                    "No baseline snapshot available to restore.",
                    snapshot_meta,
                    existing_records,
                    existing_removed,
                    no_update,
                    no_update,
                )
            restored_records = [dict(record) for record in baseline_records]
            restored_meta = dict(baseline_meta) if baseline_meta else dict(snapshot_meta or {})
            restored_meta["num_offers"] = len(restored_records)
            recompute_usd_metrics(restored_records)
            return (
                summary_block,
                "Restored snapshot to original values.",
                restored_meta,
                restored_records,
                [],
                no_update,
                no_update,
            )

        if triggered == "add-bid" and existing_records:
            base = existing_records[0].copy()
            new_bid = {key: base.get(key) for key in base}
            for feature in DISPLAY_FEATURE_ROWS:
                if feature != "Acceptance Probability":
                    new_bid.setdefault(feature, base.get(feature))
            for identifier in BID_IDENTIFIER_COLUMNS:
                if identifier in new_bid:
                    new_bid[identifier] = None
            new_bid["Bid #"] = get_next_bid_label(existing_records)
            new_bid.setdefault("offer_status", "pending")
            prepared_bid = prepare_bid_record(new_bid)
            new_data = sort_records_by_bid(existing_records + [prepared_bid])
            recompute_usd_metrics(new_data)
            new_meta = dict(snapshot_meta or {})
            new_meta["num_offers"] = len(new_data)
            return summary_block, "", new_meta, new_data, existing_removed, no_update, no_update

        if triggered == "restore-bid":
            working_records = list(existing_records or [])
            if not restore_selector:
                return (
                    summary_block,
                    "Select removed bids to restore.",
                    snapshot_meta,
                    working_records,
                    existing_removed,
                    no_update,
                    no_update,
                )
            restore_ids = set(restore_selector)
            restored_records: List[Dict[str, object]] = []
            remaining_removed: List[Dict[str, object]] = []
            for item in existing_removed:
                if item.get("id") in restore_ids:
                    restored_records.append(prepare_bid_record(item.get("record", {})))
                else:
                    remaining_removed.append(item)
            if not restored_records:
                return (
                    summary_block,
                    "No matching removed bids found.",
                    snapshot_meta,
                    working_records,
                    existing_removed,
                    no_update,
                    no_update,
                )
            working = sort_records_by_bid(working_records + restored_records)
            recompute_usd_metrics(working)
            new_meta = dict(snapshot_meta or {})
            new_meta["num_offers"] = len(working)
            return summary_block, "", new_meta, working, remaining_removed, no_update, no_update

        if triggered == "delete-bid" and existing_records:
            selections = set()
            if selected_columns:
                selections.update(
                    int(col_id.replace("bid_", ""))
                    for col_id in selected_columns
                    if col_id.startswith("bid_") and col_id.replace("bid_", "").isdigit()
                )
            if delete_selector:
                selections.update(int(idx) for idx in delete_selector)
            indices_to_remove = sorted(selections, reverse=True)
            if not indices_to_remove:
                return (
                    summary_block,
                    "Select bids to delete.",
                    snapshot_meta,
                    existing_records,
                    existing_removed,
                    no_update,
                    no_update,
                )
            working = list(existing_records)
            removed_entries: List[Dict[str, object]] = []
            for idx in indices_to_remove:
                if 0 <= idx < len(working):
                    removed_record = working.pop(idx)
                    removed_entries.append(
                        {
                            "id": str(uuid4()),
                            "label": f"Bid {removed_record.get('Bid #') or idx + 1}",
                            "record": removed_record,
                        }
                    )
            working = sort_records_by_bid(working)
            recompute_usd_metrics(working)
            new_meta = dict(snapshot_meta or {})
            new_meta["num_offers"] = len(working)
            updated_removed = existing_removed + removed_entries
            return (
                summary_block,
                "",
                new_meta,
                working,
                updated_removed,
                no_update,
                no_update,
            )

        if triggered != "snapshot-dropdown":
            return (
                summary_block,
                "",
                snapshot_meta,
                existing_records,
                existing_removed,
                no_update,
                no_update,
            )

        if snapshot_value is None:
            return (
                summary_block,
                "Select a snapshot to view bids.",
                snapshot_meta,
                existing_records,
                existing_removed,
                no_update,
                no_update,
            )

        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        subset = dataset.loc[mask].copy()

        if subset.empty:
            return (
                summary_block,
                "No rows found for the selected flight.",
                None,
                None,
                [],
                no_update,
                no_update,
            )

        if "snapshot_num" not in subset.columns:
            return (
                summary_block,
                "Snapshot information is unavailable in this dataset.",
                None,
                None,
                [],
                no_update,
                no_update,
            )

        label_map, label_column = compute_bid_label_map(subset)
        snapshot_df = subset.loc[
            subset["snapshot_num"].astype(str) == str(snapshot_value)
        ].copy()

        if snapshot_df.empty:
            return (
                summary_block,
                "No rows found for the selected snapshot.",
                None,
                None,
                [],
                no_update,
                no_update,
            )

        snapshot_df = apply_bid_labels(snapshot_df, label_map, label_column)
        if "Bid #" in snapshot_df.columns:
            snapshot_df = snapshot_df.sort_values("Bid #")

        for column in ["current_timestamp", "departure_timestamp", "travel_date"]:
            if column in snapshot_df.columns:
                snapshot_df[column] = snapshot_df[column].apply(
                    lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x
                )

        seats_available = snapshot_df.get("seats_available")
        seats_value = (
            seats_available.iloc[0]
            if seats_available is not None and not seats_available.empty
            else None
        )

        departure_time = snapshot_df.get("departure_timestamp")
        current_time = snapshot_df.get("current_timestamp")
        departure_ts = (
            pd.to_datetime(departure_time.iloc[0])
            if departure_time is not None and not departure_time.empty
            else None
        )
        current_ts = (
            pd.to_datetime(current_time.iloc[0])
            if current_time is not None and not current_time.empty
            else None
        )

        delta_hours: Optional[float] = None
        if isinstance(departure_ts, pd.Timestamp) and isinstance(current_ts, pd.Timestamp):
            delta = departure_ts - current_ts
            delta_hours = max(delta.total_seconds() / 3600, 0)

        base_records = [prepare_bid_record(record) for record in snapshot_df.to_dict("records")]
        base_data = sort_records_by_bid(base_records)
        recompute_usd_metrics(base_data)

        snapshot_meta = {
            "carrier": carrier,
            "flight_number": flight_number,
            "travel_date": travel_date,
            "upgrade_type": upgrade_type,
            "snapshot": snapshot_value,
            "seats_available": seats_value,
            "num_offers": len(base_data),
            "departure_timestamp": departure_ts.isoformat() if isinstance(departure_ts, pd.Timestamp) else None,
            "current_timestamp": current_ts.isoformat() if isinstance(current_ts, pd.Timestamp) else None,
            "time_before_departure_hours": delta_hours,
        }

        baseline_records = [dict(record) for record in base_data]
        baseline_meta = dict(snapshot_meta)
        return (
            summary_block,
            "",
            snapshot_meta,
            base_data,
            [],
            baseline_records,
            baseline_meta,
        )

    @app.callback(
        Output("seats-available-input", "value"),
        Output("offers-input", "value"),
        Output("time-before-days-input", "value"),
        Output("time-before-hours-input", "value"),
        Input("snapshot-meta-store", "data"),
        Input("bid-records-store", "data"),
    )
    def sync_inputs(meta: Optional[Dict[str, str]], records: Optional[List[Dict[str, str]]]):
        seats_value = meta.get("seats_available") if meta else None
        offers_value = len(records) if records else 0
        delta_hours = meta.get("time_before_departure_hours") if meta else None
        if delta_hours is not None:
            days = int(delta_hours // 24)
            hours = int(round(delta_hours - days * 24))
        else:
            days = None
            hours = None
        return seats_value, offers_value, days, hours

    @app.callback(
        Output("bid-records-store", "data", allow_duplicate=True),
        Output("snapshot-meta-store", "data", allow_duplicate=True),
        Input("seats-available-input", "value"),
        Input("offers-input", "value"),
        Input("time-before-days-input", "value"),
        Input("time-before-hours-input", "value"),
        State("bid-records-store", "data"),
        State("snapshot-meta-store", "data"),
        prevent_initial_call=True,
    )
    def apply_summary_overrides(
        seats_value: Optional[float],
        offers_value: Optional[int],
        days_value: Optional[int],
        hours_value: Optional[int],
        records: Optional[List[Dict[str, str]]],
        meta: Optional[Dict[str, str]],
    ):
        if records is None or meta is None:
            return no_update, no_update

        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

        updated_records = [dict(record) for record in records]
        updated_meta = dict(meta)

        if triggered == "seats-available-input":
            if seats_value is None:
                return no_update, no_update
            for record in updated_records:
                record["seats_available"] = seats_value
            updated_meta["seats_available"] = seats_value
            return updated_records, updated_meta

        if triggered == "offers-input":
            if offers_value is None or offers_value < 0:
                return no_update, no_update
            updated_records = sort_records_by_bid(updated_records)
            current_len = len(updated_records)
            if offers_value == current_len:
                updated_meta["num_offers"] = offers_value
                for record in updated_records:
                    normalize_offer_time(record)
                recompute_usd_metrics(updated_records)
                return updated_records, updated_meta
            if offers_value > current_len and current_len > 0:
                template = updated_records[0]
                for _ in range(offers_value - current_len):
                    new_bid = dict(template)
                    for identifier in BID_IDENTIFIER_COLUMNS:
                        if identifier in new_bid:
                            new_bid[identifier] = None
                    new_bid["Bid #"] = get_next_bid_label(updated_records)
                    new_bid.setdefault("offer_status", "pending")
                    updated_records.append(prepare_bid_record(new_bid))
            elif offers_value < current_len:
                updated_records = updated_records[: offers_value]
            updated_records = sort_records_by_bid(updated_records)
            for record in updated_records:
                normalize_offer_time(record)
            recompute_usd_metrics(updated_records)
            updated_meta["num_offers"] = len(updated_records)
            return updated_records, updated_meta

        if triggered in {"time-before-days-input", "time-before-hours-input"}:
            if days_value is None and hours_value is None:
                return no_update, no_update
            hours_value = hours_value or 0
            days_value = days_value or 0
            total_hours = max(days_value * 24 + hours_value, 0)
            departure_iso = updated_meta.get("departure_timestamp")
            if departure_iso:
                departure_ts = pd.to_datetime(departure_iso)
                new_current = departure_ts - pd.Timedelta(hours=total_hours)
                for record in updated_records:
                    record["current_timestamp"] = new_current.isoformat()
                updated_meta["current_timestamp"] = new_current.isoformat()
            updated_meta["time_before_departure_hours"] = total_hours
            return updated_records, updated_meta

        return no_update, no_update

    @app.callback(
        Output("bid-table", "columns"),
        Output("bid-table", "data"),
        Output("bid-table", "style_data_conditional"),
        Input("bid-records-store", "data"),
        Input("prediction-store", "data"),
    )
    def render_bid_table(
        records: Optional[List[Dict[str, str]]],
        predictions: Optional[Dict[str, float]],
    ):
        return build_bid_table(records, predictions)

    @app.callback(
        Output("bid-delete-selector", "options"),
        Output("bid-delete-selector", "value"),
        Input("bid-records-store", "data"),
    )
    def sync_delete_selector(records: Optional[List[Dict[str, object]]]):
        if not records:
            return [], []
        options = [
            {
                "label": f"Bid {record.get('Bid #') or record.get('bid_number') or idx + 1}",
                "value": idx,
            }
            for idx, record in enumerate(records)
        ]
        return options, []

    @app.callback(
        Output("bid-restore-selector", "options"),
        Output("bid-restore-selector", "value"),
        Input("removed-bids-store", "data"),
    )
    def sync_restore_selector(removed: Optional[List[Dict[str, object]]]):
        if not removed:
            return [], []
        options = [
            {"label": item.get("label") or f"Removed bid {idx + 1}", "value": item.get("id")}
            for idx, item in enumerate(removed)
            if item.get("id") is not None
        ]
        return options, []

    @app.callback(
        Output("bid-records-store", "data", allow_duplicate=True),
        Input("bid-table", "data_timestamp"),
        State("bid-table", "data"),
        State("bid-table", "columns"),
        State("bid-records-store", "data"),
        prevent_initial_call=True,
    )
    def persist_table_edits(
        data_timestamp: Optional[int],
        table_data: Optional[List[Dict[str, object]]],
        columns: Optional[List[Dict[str, object]]],
        records: Optional[List[Dict[str, object]]],
    ):
        if not data_timestamp or not table_data or not columns or not records:
            return no_update

        updated_records = apply_table_edits(records, table_data, columns)
        if updated_records is None:
            return no_update
        return updated_records

    @app.callback(
        Output("prediction-graph", "figure"),
        Output("prediction-store", "data"),
        Output("prediction-warning", "children"),
        Input("bid-records-store", "data"),
        Input("model-uri-store", "data"),
        State("dataset-path-store", "data"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("upgrade-dropdown", "value"),
        State("snapshot-meta-store", "data"),
    )
    def run_predictions(
        records: Optional[List[Dict[str, str]]],
        model_uri: Optional[str],
        dataset_path: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        snapshot_meta: Optional[Dict[str, object]],
    ):
        if not records:
            return build_prediction_plot(pd.DataFrame()), {}, ""

        selected_df = prepare_prediction_dataframe(records)

        if not model_uri:
            empty_fig = build_prediction_plot(pd.DataFrame())
            return empty_fig, {}, "Load a model to generate acceptance probabilities."

        plot_source = pd.DataFrame()
        label_map: Dict[object, int] = {}
        label_column: Optional[str] = None
        if dataset_path and carrier and flight_number and travel_date and upgrade_type:
            try:
                dataset = load_dataset_cached(dataset_path)
                required = {"carrier_code", "flight_number", "travel_date", "upgrade_type"}
                if required.issubset(dataset.columns):
                    travel_date_dt = pd.to_datetime(travel_date).date()
                    mask = (
                        (dataset["carrier_code"] == carrier)
                        & (dataset["flight_number"].astype(str) == str(flight_number))
                        & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
                        & (dataset["upgrade_type"] == upgrade_type)
                    )
                    plot_source = dataset.loc[mask].copy()
                    label_map, label_column = compute_bid_label_map(plot_source)
                    plot_source = apply_bid_labels(plot_source, label_map, label_column)
                    if "Bid #" in plot_source.columns:
                        plot_source = plot_source.sort_values("Bid #")
            except Exception:
                plot_source = pd.DataFrame()

        selected_snapshot = None
        if snapshot_meta:
            selected_snapshot = snapshot_meta.get("snapshot")
        selected_snapshot_value = str(selected_snapshot) if selected_snapshot is not None else None

        if label_map and label_column:
            selected_df = apply_bid_labels(selected_df, label_map, label_column)

        if "snapshot_num" not in selected_df.columns and selected_snapshot_value is not None:
            selected_df["snapshot_num"] = selected_snapshot_value
        elif "snapshot_num" in selected_df.columns:
            selected_df["snapshot_num"] = selected_df["snapshot_num"].astype(str)

        if plot_source.empty:
            combined_df = selected_df.copy()
        else:
            if selected_snapshot_value is not None and "snapshot_num" in plot_source.columns:
                mask = plot_source["snapshot_num"].astype(str) == selected_snapshot_value
                plot_source = plot_source.loc[~mask]
            if "snapshot_num" in plot_source.columns:
                plot_source["snapshot_num"] = plot_source["snapshot_num"].astype(str)
            combined_df = pd.concat([plot_source, selected_df], ignore_index=True, sort=False)

        try:
            plot_pred_df = predict(model_uri, combined_df.copy())
            table_pred_df = predict(model_uri, selected_df.copy())
        except Exception as exc:  # pragma: no cover - user feedback
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Prediction failed: {exc}")
            return empty_fig, {}, str(exc)

        figure = build_prediction_plot(plot_pred_df)
        warning = table_pred_df.attrs.get("model_warning", "") or plot_pred_df.attrs.get("model_warning", "") or ""

        predictions = {}
        for idx, _ in enumerate(records):
            column_id = f"bid_{idx}"
            if idx < len(table_pred_df):
                value = table_pred_df.iloc[idx].get("Acceptance Probability")
                if value is None or pd.isna(value):
                    predictions[column_id] = None
                else:
                    try:
                        predictions[column_id] = round(float(value), 4)
                    except (TypeError, ValueError):
                        predictions[column_id] = value
            else:
                predictions[column_id] = None

        return figure, predictions, warning
    return app


def main():  # pragma: no cover - manual entry point
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
