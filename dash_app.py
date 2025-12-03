"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

from copy import deepcopy
from typing import Optional

import mlflow
from dash import Dash, Input, Output, State, dcc, html
from mlflow.exceptions import MlflowException
from bid_predictor.utils import detect_execution_environment

from bid_predictor_ui import (
    DEFAULT_UI_FEATURE_CONFIG,
    build_ui_feature_config,
    load_dataset_cached,
    load_model_cached,
)
from bid_predictor_ui.feature_sensitivity import (
    build_feature_sensitivity_tab,
    register_feature_sensitivity_callbacks,
)
from bid_predictor_ui.acceptance_explorer import (
    build_acceptance_tab,
    load_acceptance_dataset,
    register_acceptance_callbacks,
)
from bid_predictor_ui.snapshot import (
    build_snapshot_tab,
    register_snapshot_callbacks,
)

DEFAULT_ACCEPTANCE_TABLE = "model_prediction_testing.audit_bid_predictor"

if detect_execution_environment()[0] in (
        "sagemaker_notebook",
        "sagemaker_terminal",
    ):
    arn = os.environ["MLFLOW_AWS_ARN"]
    mlflow.set_tracking_uri(arn)
    default_dataset_path = (
        os.environ.get("S3_BUCKET_DATA")
        + "/data/air_canada_and_lot/evaluation_sets/eval_bid_data_snapshots_v2_3_or_mode_bids.parquet"
    )
else:
    default_dataset_path = (
        "./data/air_canada_and_lot/evaluation_sets/eval_bid_data_snapshots_v2_3_or_mode_bids.parquet"
    )

# -- Dash application --------------------------------------------------------------------------


def create_app() -> Dash:
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "Bid Predictor Playground",
                        style={"margin": "0", "color": "#1b4965"},
                    ),
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
                            html.Div(
                                [
                                    html.Label(
                                        "Dataset path", style={"fontWeight": "600"}
                                    ),
                                    dcc.Input(
                                        id="dataset-path",
                                        type="text",
                                        value=default_dataset_path,
                                        placeholder="Path to bid_data_snapshots_v2.parquet",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
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
                                    dcc.Loading(
                                        id="dataset-loading",
                                        type="circle",
                                        children=html.Div(
                                            id="dataset-status",
                                            className="status-message",
                                            style={"marginTop": "0.5rem"},
                                        ),
                                    ),
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
                                    html.Label(
                                        "MLflow tracking URI",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Input(
                                        id="mlflow-tracking-uri",
                                        type="text",
                                        value=mlflow.get_tracking_uri(),
                                        placeholder="http://localhost:5000",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Label("Model name", style={"fontWeight": "600"}),
                                    dcc.Input(
                                        id="model-name",
                                        type="text",
                                        placeholder="Registered model name",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Label(
                                        "Model stage or version",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Input(
                                        id="model-stage",
                                        type="text",
                                        placeholder="e.g. Production or 5",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
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
                                    dcc.Loading(
                                        id="model-loading",
                                        type="circle",
                                        children=html.Div(
                                            id="model-status",
                                            className="status-message",
                                            style={"marginTop": "0.5rem"},
                                        ),
                                    ),
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
                        id="standard-controls",
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "1.5rem",
                            "marginBottom": "1.5rem",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Data source",
                                                style={"fontWeight": "600"},
                                            ),
                                            dcc.RadioItems(
                                                id="acceptance-source",
                                                options=[
                                                    {
                                                        "label": "Local / S3 file",
                                                        "value": "path",
                                                    },
                                                    {
                                                        "label": "AWS Redshift (ENV)",
                                                        "value": "redshift",
                                                    },
                                                ],
                                                value="path",
                                                labelStyle={
                                                    "display": "block",
                                                    "marginBottom": "0.25rem",
                                                },
                                                style={"marginBottom": "0.5rem"},
                                            ),
                                        ]
                                    ),
                                    html.Label(
                                        "Acceptance dataset path",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Input(
                                        id="acceptance-dataset-path",
                                        type="text",
                                        placeholder="Path to acceptance probability data",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Label(
                                        "Table name (Redshift)",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Input(
                                        id="acceptance-table-name",
                                        type="text",
                                        value=DEFAULT_ACCEPTANCE_TABLE,
                                        placeholder="schema.table",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Label(
                                        "Recent hours (optional)",
                                        style={"fontWeight": "600"},
                                    ),
                                    dcc.Input(
                                        id="acceptance-hours",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=12,
                                        placeholder="e.g. 24",
                                        style={
                                            "width": "100%",
                                            "marginBottom": "0.5rem",
                                        },
                                    ),
                                    html.Button(
                                        "Load acceptance dataset",
                                        id="load-acceptance-dataset",
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
                                    dcc.Loading(
                                        id="acceptance-dataset-loading",
                                        type="circle",
                                        children=html.Div(
                                            id="acceptance-dataset-status",
                                            className="status-message",
                                            style={"marginTop": "0.5rem"},
                                        ),
                                    ),
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
                        id="acceptance-controls",
                        style={
                            "display": "none",
                            "flexWrap": "wrap",
                            "gap": "1.5rem",
                            "marginBottom": "1.5rem",
                        },
                    ),
                ],
            ),
            dcc.Store(id="dataset-path-store"),
            dcc.Store(id="acceptance-dataset-path-store"),
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
            dcc.Store(id="snapshot-selection-request-store"),
            dcc.Store(id="selection-history-store", data=[]),
            dcc.Store(id="scenario-selection-request-store"),
            dcc.Store(id="scenario-selection-history-store", data=[]),
            dcc.Store(
                id="feature-config-store",
                data=deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            ),
            dcc.Tabs(
                id="main-tabs",
                value="snapshot",
                children=[
                    build_snapshot_tab(),
                    build_feature_sensitivity_tab(),
                    build_acceptance_tab(),
                ],
                style={"marginTop": "1rem"},
            ),
        ],
        style={
            "fontFamily": "'Segoe UI', sans-serif",
            "backgroundColor": "#fafafa",
            "padding": "1.5rem",
        },
    )

    register_snapshot_callbacks(app)
    register_feature_sensitivity_callbacks(app)
    register_acceptance_callbacks(app)

    # Callbacks -----------------------------------------------------------------------------

    @app.callback(
        Output("standard-controls", "style"),
        Output("acceptance-controls", "style"),
        Input("main-tabs", "value"),
    )
    def toggle_control_panels(active_tab: str):
        standard_style = {
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "1.5rem",
            "marginBottom": "1.5rem",
        }
        acceptance_style = {
            "display": "none",
            "flexWrap": "wrap",
            "gap": "1.5rem",
            "marginBottom": "1.5rem",
        }
        if active_tab == "acceptance":
            standard_style["display"] = "none"
            acceptance_style["display"] = "flex"
        return standard_style, acceptance_style

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
        Output("acceptance-dataset-status", "children"),
        Output("acceptance-dataset-path-store", "data"),
        Input("load-acceptance-dataset", "n_clicks"),
        State("acceptance-dataset-path", "value"),
        State("acceptance-source", "value"),
        State("acceptance-table-name", "value"),
        State("acceptance-hours", "value"),
        prevent_initial_call=True,
    )
    def load_acceptance_dataset_path(
        n_clicks: int,
        path: str,
        source: str,
        table_name: str,
        hours: Optional[int],
    ):
        dataset_config = None
        if source == "redshift":
            dataset_config = {
                "source": "redshift",
                "table": table_name or DEFAULT_ACCEPTANCE_TABLE,
            }
            if hours not in (None, ""):
                dataset_config["hours"] = hours
        else:
            if not path:
                return "Please provide a dataset path.", None
            dataset_config = {"source": "path", "path": path}
            if hours not in (None, ""):
                dataset_config["hours"] = hours

        try:
            dataset = load_acceptance_dataset(dataset_config)
        except Exception as exc:  # pragma: no cover - user feedback
            return f"Failed to load acceptance dataset: {exc}", None

        if source == "redshift":
            hours_text = (
                f" from last {int(hours)} hours" if hours not in (None, "") else ""
            )
            summary = (
                f"Loaded {len(dataset):,} rows from {dataset_config['table']}{hours_text}."
            )
        else:
            hours_text = (
                f" from last {int(hours)} hours" if hours not in (None, "") else ""
            )
            summary = f"Loaded acceptance dataset{hours_text} with {len(dataset):,} rows."
        return summary, dataset_config

    @app.callback(
        Output("model-status", "children"),
        Output("model-uri-store", "data"),
        Output("feature-config-store", "data"),
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
                model_uri = f"models:/{model_name}/Production"
            model = load_model_cached(model_uri)
            raw_config = getattr(model, "feature_config_", None) or getattr(
                model, "feature_config", None
            )
            ui_feature_config = build_ui_feature_config(raw_config)
        except MlflowException as exc:  # pragma: no cover - user feedback
            return (
                f"Failed to load model: {exc}",
                None,
                deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            )
        except Exception as exc:  # pragma: no cover
            return (
                f"Unexpected error while loading model: {exc}",
                None,
                deepcopy(DEFAULT_UI_FEATURE_CONFIG),
            )

        return f"Loaded model from {model_uri}", model_uri, ui_feature_config

    return app


def main():  # pragma: no cover - manual entry point
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
