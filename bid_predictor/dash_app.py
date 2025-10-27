"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import mlflow
import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html
from mlflow.exceptions import MlflowException
import plotly.graph_objects as go

from .feature_config import _GROUPBY_KEY_FEATURES, load_feature_config
from .tuning.data_access import load_training_data, resolve_train_file


# -- Caching helpers ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_dataset_cached(path: str) -> pd.DataFrame:
    """Load the training dataset and cache it for repeated access."""

    df = load_training_data(path)
    # Ensure the group-by key columns exist; raise a helpful error if missing.
    missing = [col for col in _GROUPBY_KEY_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: {}".format(
                ", ".join(sorted(missing))
            )
        )

    # Normalize timestamp columns for reliable downstream calculations.
    for col in ("current_timestamp", "departure_timestamp", "travel_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


@lru_cache(maxsize=4)
def _load_model_cached(model_uri: str):
    """Load and cache the MLflow model given a model URI."""

    return mlflow.sklearn.load_model(model_uri)


def _get_feature_columns() -> Tuple[List[str], List[str]]:
    feature_config = load_feature_config()
    features = list(feature_config["pre_features"])
    categorical = list(feature_config["cat_features"])
    return features, categorical


def _prepare_prediction_dataframe(table_records: Iterable[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(list(table_records))
    if df.empty:
        return df

    # Attempt to convert numeric columns back to floats/ints when possible.
    for col in df.columns:
        if col in {"carrier_code", "flight_number", "fare_class", "offer_status", "upgrade_type"}:
            continue
        if df[col].isnull().all():
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            try:
                df[col] = pd.to_datetime(df[col])
            except (TypeError, ValueError):
                pass
    if "travel_date" in df.columns:
        df["travel_date"] = pd.to_datetime(df["travel_date"])
    if "current_timestamp" in df.columns:
        df["current_timestamp"] = pd.to_datetime(df["current_timestamp"])
    if "departure_timestamp" in df.columns:
        df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
    return df


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

    # Create traces per bid using time ordering
    for bid_id, grp in work.groupby("Bid #"):
        grp_sorted = grp.sort_values("time_until_departure_hours")
        status = grp_sorted["offer_status"].iloc[-1]
        label = f"Bid {bid_id} - {status}"
        fig.add_trace(
            go.Bar(
                x=grp_sorted["time_until_departure_hours"],
                y=grp_sorted["Acceptance Probability"],
                name=label,
                hovertemplate="Time: %{x}<br>Probability: %{y:.3f}<extra></extra>",
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
        yaxis=dict(title="Acceptance probability", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0.5, xanchor="center"),
        height=760,
    )
    if "seats_available" in work.columns:
        fig.update_layout(yaxis2=dict(title="Seats available", overlaying="y", side="right"))
    return fig


def _predict(model_uri: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    model = _load_model_cached(model_uri)
    feature_df = df.copy()
    model_warning: Optional[str] = None

    expected_columns: Optional[List[str]] = None
    try:
        metadata = getattr(model, "metadata", None)
        if metadata is not None:
            input_schema = metadata.get_input_schema()
            if input_schema is not None:
                names = list(input_schema.input_names())
                expected_columns = names or None
    except AttributeError:
        expected_columns = None

    if expected_columns:
        missing = [col for col in expected_columns if col not in feature_df.columns]
        if missing:
            model_warning = (
                "Added missing model columns with empty values: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        # ensure the dataframe has exactly the schema expected by the model
        feature_df = feature_df.reindex(columns=expected_columns)
    else:
        features, _ = _get_feature_columns()
        missing = [col for col in features if col not in feature_df.columns]
        if missing:
            model_warning = (
                "Added missing feature config columns with empty values: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        # align to feature config order, introducing NaNs for absent columns so
        # downstream transformers receive the expected number of features
        feature_df = feature_df.reindex(columns=features)

    predictions = model.predict_proba(feature_df)
    if isinstance(predictions, pd.DataFrame) and "Acceptance Probability" in predictions.columns:
        df["Acceptance Probability"] = predictions["Acceptance Probability"].values
    else:
        # mlflow.pyfunc returns numpy array; expect probability in second column
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            df["Acceptance Probability"] = predictions[:, 1]
        else:
            df["Acceptance Probability"] = predictions
    if model_warning:
        df.attrs["model_warning"] = model_warning
    return df


# -- Dash application --------------------------------------------------------------------------


def create_app() -> Dash:
    feature_columns, _ = _get_feature_columns()
    default_dataset_path = resolve_train_file(None)

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
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Carrier", style={"fontWeight": "600"}),
                            dcc.Dropdown(id="carrier-dropdown", placeholder="Select carrier", options=[]),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label("Flight number", style={"fontWeight": "600"}),
                            dcc.Dropdown(id="flight-number-dropdown", placeholder="Select flight", options=[]),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label("Travel date", style={"fontWeight": "600"}),
                            dcc.Dropdown(id="travel-date-dropdown", placeholder="Select travel date", options=[]),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label("Upgrade type", style={"fontWeight": "600"}),
                            dcc.Dropdown(id="upgrade-dropdown", placeholder="Select upgrade type", options=[]),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "1rem",
                    "padding": "1.5rem",
                    "backgroundColor": "#edf2fb",
                    "borderRadius": "12px",
                    "boxShadow": "inset 0 0 10px rgba(27, 73, 101, 0.08)",
                    "marginBottom": "1.5rem",
                },
            ),
            html.Div(
                [
                    html.Div(id="flight-summary", style={"flex": "2", "paddingRight": "1rem"}),
                    html.Div(
                        [
                            html.Label("Snapshot", style={"fontWeight": "600"}),
                            dcc.Dropdown(id="snapshot-dropdown", placeholder="Select snapshot", options=[]),
                            html.Div(id="snapshot-details", style={"marginTop": "0.75rem"}),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "1rem",
                    "padding": "1.5rem",
                    "backgroundColor": "#f4f1de",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)",
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
                                },
                            ),
                        ],
                        style={"marginBottom": "0.75rem"},
                    ),
                    dash_table.DataTable(
                        id="bid-table",
                        columns=[{"name": col, "id": col, "editable": True} for col in feature_columns],
                        data=[],
                        editable=True,
                        row_deletable=True,
                        row_selectable="multi",
                        style_table={"overflowX": "auto", "borderRadius": "8px", "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"},
                        style_cell={"textAlign": "center", "padding": "0.6rem", "backgroundColor": "#ffffff"},
                        style_header={"backgroundColor": "#1b4965", "color": "white", "fontWeight": "700"},
                    ),
                ],
                style={
                    "padding": "1.5rem",
                    "backgroundColor": "#ffffff",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
                    "marginBottom": "1.5rem",
                },
            ),
            html.Div(
                [
                    html.H3("Predictions", style={"color": "#1b4965"}),
                    html.Div(id="prediction-warning", className="status-message", style={"marginBottom": "0.75rem"}),
                    dash_table.DataTable(
                        id="prediction-table",
                        columns=[],
                        data=[],
                        style_table={"overflowX": "auto", "borderRadius": "8px", "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"},
                        style_cell={"textAlign": "center", "padding": "0.6rem", "backgroundColor": "#f7fff7"},
                        style_header={"backgroundColor": "#16324f", "color": "white", "fontWeight": "700"},
                    ),
                    dcc.Graph(id="prediction-graph", style={"height": "800px", "marginTop": "1rem"}),
                ],
                style={
                    "padding": "1.5rem",
                    "backgroundColor": "#edf2fb",
                    "borderRadius": "12px",
                    "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.08)",
                    "marginBottom": "2rem",
                },
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
            dataset = _load_dataset_cached(path)
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
            _load_model_cached(model_uri)
        except MlflowException as exc:  # pragma: no cover - user feedback
            return f"Failed to load model: {exc}", None
        except Exception as exc:  # pragma: no cover
            return f"Unexpected error while loading model: {exc}", None

        return f"Loaded model from {model_uri}", model_uri

    @app.callback(
        Output("carrier-dropdown", "options"),
        Output("carrier-dropdown", "value"),
        Input("dataset-path-store", "data"),
    )
    def populate_carriers(dataset_path: Optional[str]):
        if not dataset_path:
            return [], None

        dataset = _load_dataset_cached(dataset_path)
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

        dataset = _load_dataset_cached(dataset_path)
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

        dataset = _load_dataset_cached(dataset_path)
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

        dataset = _load_dataset_cached(dataset_path)
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

        dataset = _load_dataset_cached(dataset_path)
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
        Output("snapshot-details", "children"),
        Output("bid-table", "columns"),
        Output("bid-table", "data"),
        Output("bid-table", "selected_rows"),
        Input("snapshot-dropdown", "value"),
        Input("add-bid", "n_clicks"),
        Input("delete-bid", "n_clicks"),
        State("bid-table", "data"),
        State("bid-table", "selected_rows"),
        State("bid-table", "columns"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("upgrade-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def update_snapshot_view(
        snapshot_value: Optional[str],
        add_clicks: int,
        delete_clicks: int,
        existing_data: Optional[List[Dict[str, str]]],
        selected_rows: Optional[List[int]],
        existing_columns: Optional[List[Dict[str, str]]],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
    ):
        from dash import callback_context

        if not dataset_path:
            message = html.Div("Load a dataset to begin.")
            return message, html.Div(), [], [], []

        dataset = _load_dataset_cached(dataset_path)
        if not carrier or not flight_number or not travel_date or not upgrade_type:
            message = html.Div("Select carrier, flight, date, and upgrade type to continue.")
            return message, html.Div(), [], [], []

        travel_date_dt = pd.to_datetime(travel_date).date()
        subset_mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        subset = dataset.loc[subset_mask].copy()

        if subset.empty:
            message = html.Div("No rows found for selected flight.")
            return message, html.Div(), [], [], []

        snapshot_count = (
            subset["snapshot_num"].nunique() if "snapshot_num" in subset.columns else 0
        )

        flight_summary = html.Div(
            [
                html.H3("Selected flight", style={"marginTop": "0", "color": "#1b4965"}),
                html.Ul(
                    [
                        html.Li(f"Carrier: {carrier}"),
                        html.Li(f"Flight number: {flight_number}"),
                        html.Li(f"Travel date: {travel_date}"),
                        html.Li(f"Upgrade type: {upgrade_type}"),
                        html.Li(f"Snapshots available: {snapshot_count}"),
                    ],
                    style={"paddingLeft": "1.2rem"},
                ),
            ]
        )

        ctx = callback_context
        triggered = getattr(ctx, "triggered_id", None)
        if triggered is None and ctx.triggered:
            triggered = ctx.triggered[0]["prop_id"].split(".")[0]

        if snapshot_value is None:
            message = html.Div("Select a snapshot to view bid details.")
            return flight_summary, message, [], [], []

        if "snapshot_num" not in subset.columns:
            message = html.Div("Snapshot information is unavailable in this dataset.")
            return flight_summary, message, [], [], []

        snapshot_df = subset.loc[
            subset["snapshot_num"].astype(str) == str(snapshot_value)
        ].copy()
        if snapshot_df.empty:
            message = html.Div("No rows found for the selected snapshot.")
            return flight_summary, message, [], [], []

        sort_columns: List[str] = []
        if "offer_time" in snapshot_df.columns:
            sort_columns.append("offer_time")
        if "Bid #" in snapshot_df.columns:
            sort_columns.append("Bid #")
        if sort_columns:
            snapshot_df.sort_values(by=sort_columns, inplace=True, na_position="last")

        # Prepare additional columns
        if "Bid #" not in snapshot_df.columns:
            snapshot_df.insert(0, "Bid #", range(1, len(snapshot_df) + 1))
        if "seats_available" not in snapshot_df.columns:
            seats_val = snapshot_df.get("seats_available", pd.Series(dtype=float))
            if seats_val.empty:
                snapshot_df["seats_available"] = None
        if {
            "departure_timestamp",
            "current_timestamp",
        }.issubset(snapshot_df.columns):
            departure_ts = pd.to_datetime(snapshot_df["departure_timestamp"])
            current_ts = pd.to_datetime(snapshot_df["current_timestamp"])
            delta_hours = (departure_ts - current_ts).dt.total_seconds() / 3600
            snapshot_df["time_before_departure_hours"] = delta_hours.round(2)
            if "time_until_departure_hours" not in snapshot_df.columns:
                snapshot_df["time_until_departure_hours"] = snapshot_df[
                    "time_before_departure_hours"
                ]
        else:
            if "time_before_departure_hours" not in snapshot_df.columns:
                snapshot_df["time_before_departure_hours"] = None
        if "Acceptance Probability" not in snapshot_df.columns:
            snapshot_df["Acceptance Probability"] = None

        required_columns = [
            "Bid #",
            "offer_status",
            "seats_available",
            "time_before_departure_hours",
            "item_count",
            "usd_base_amount",
            "fare_class",
            "offer_time",
            "multiplier_fare_class",
            "multiplier_loyalty",
            "multiplier_success_history",
            "multiplier_payment_type",
            "usd_base_amount_25%",
            "usd_base_amount_50%",
            "usd_base_amount_75%",
            "Acceptance Probability",
            "snapshot_num",
            "current_timestamp",
            "departure_timestamp",
            "carrier_code",
            "flight_number",
            "travel_date",
            "upgrade_type",
        ]
        for column in required_columns:
            if column not in snapshot_df.columns:
                snapshot_df[column] = None

        ordered_columns = required_columns + [
            col for col in snapshot_df.columns if col not in required_columns
        ]
        snapshot_df = snapshot_df[ordered_columns]

        # Format datetimes for display
        for col in ["current_timestamp", "departure_timestamp", "offer_time"]:
            if col in snapshot_df.columns:
                snapshot_df[col] = snapshot_df[col].apply(
                    lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x
                )
        if "travel_date" in snapshot_df.columns:
            snapshot_df["travel_date"] = snapshot_df["travel_date"].apply(
                lambda x: x.date().isoformat() if isinstance(x, pd.Timestamp) else x
            )

        editable_columns = {
            "seats_available",
            "time_before_departure_hours",
            "item_count",
            "usd_base_amount",
            "fare_class",
            "offer_time",
            "multiplier_fare_class",
            "multiplier_loyalty",
            "multiplier_success_history",
            "multiplier_payment_type",
            "usd_base_amount_25%",
            "usd_base_amount_50%",
            "usd_base_amount_75%",
        }

        columns = [
            {
                "name": col,
                "id": col,
                "editable": col in editable_columns,
            }
            for col in snapshot_df.columns
        ]

        def _snapshot_summary_block(df: pd.DataFrame) -> html.Div:
            seats_available = df.get("seats_available")
            seats_value = seats_available.iloc[0] if seats_available is not None else None
            current_time_series = df.get("current_timestamp")
            current_time = current_time_series.iloc[0] if current_time_series is not None else None
            if isinstance(current_time, str):
                current_time_display = current_time
            elif isinstance(current_time, pd.Timestamp):
                current_time_display = current_time.isoformat()
            else:
                current_time_display = str(current_time) if current_time is not None else "N/A"

            departure_series = df.get("departure_timestamp")
            departure_time = departure_series.iloc[0] if departure_series is not None else None
            if isinstance(departure_time, str):
                departure_ts = pd.to_datetime(departure_time)
            else:
                departure_ts = departure_time

            if isinstance(current_time, str):
                current_ts = pd.to_datetime(current_time)
            else:
                current_ts = current_time

            delta_display = "N/A"
            if isinstance(departure_ts, pd.Timestamp) and isinstance(current_ts, pd.Timestamp):
                delta = departure_ts - current_ts
                days = delta.days
                hours = int((delta.total_seconds() - days * 86400) // 3600)
                delta_display = f"{days} days {hours} hours"

            num_offers = len(df)
            offers_series = df.get("offer_time")

            return html.Div(
                [
                    html.P(f"Seats available: {seats_value if seats_value is not None else 'N/A'}"),
                    html.P(f"Number of offers: {num_offers}"),
                    html.P(f"Current timestamp: {current_time_display}"),
                    html.P(f"Time before departure: {delta_display}"),
                    html.P(
                        f"Offer window: {offers_series.min()} – {offers_series.max()}"
                        if offers_series is not None and not pd.Series(offers_series).isnull().all()
                        else ""
                    ),
                ],
                style={"lineHeight": "1.6", "color": "#16324f"},
            )

        snapshot_details = _snapshot_summary_block(snapshot_df)

        base_data = snapshot_df.to_dict("records")

        if triggered == "add-bid":
            working_data = list(existing_data or base_data)
            columns_ids = [col["id"] for col in (existing_columns or columns)]
            template = {col_id: None for col_id in columns_ids}
            template_source = (working_data[0] if working_data else base_data[0]) if (working_data or base_data) else {}
            for key in [
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
                "snapshot_num",
                "current_timestamp",
                "departure_timestamp",
            ]:
                if template_source and key in template_source:
                    template[key] = template_source.get(key)
            existing_ids = [
                row.get("Bid #") for row in working_data if row.get("Bid #") not in (None, "")
            ]
            next_bid = 1
            if existing_ids:
                try:
                    next_bid = int(max(float(bid) for bid in existing_ids)) + 1
                except Exception:
                    next_bid = len(working_data) + 1
            template["Bid #"] = next_bid
            template.setdefault("offer_status", "pending")
            working_data.append(template)
            return flight_summary, snapshot_details, columns, working_data, []

        if triggered == "delete-bid":
            if not existing_data:
                return flight_summary, snapshot_details, columns, base_data, []
            if not selected_rows:
                return flight_summary, snapshot_details, columns, existing_data, selected_rows or []
            new_data = [row for idx, row in enumerate(existing_data) if idx not in selected_rows]
            return flight_summary, snapshot_details, columns, new_data, []

        return flight_summary, snapshot_details, columns, base_data, []

    @app.callback(
        Output("prediction-table", "columns"),
        Output("prediction-table", "data"),
        Output("prediction-graph", "figure"),
        Output("prediction-warning", "children"),
        Input("bid-table", "data"),
        State("model-uri-store", "data"),
    )
    def run_predictions(table_data: List[Dict[str, str]], model_uri: Optional[str]):
        if not table_data:
            return [], [], go.Figure(), ""
        if not model_uri:
            df = pd.DataFrame(table_data)
            columns = [{"name": col, "id": col} for col in df.columns]
            return columns, table_data, _build_prediction_plot(pd.DataFrame()), ""

        df = _prepare_prediction_dataframe(table_data)
        try:
            pred_df = _predict(model_uri, df)
        except Exception as exc:  # pragma: no cover - user feedback
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Prediction failed: {exc}")
            columns = [{"name": col, "id": col} for col in df.columns]
            return columns, table_data, empty_fig, str(exc)

        columns = [
            {"name": col, "id": col}
            if col == "Acceptance Probability"
            else {"name": col, "id": col}
            for col in pred_df.columns
        ]
        figure = _build_prediction_plot(pred_df)
        data = pred_df.to_dict("records")
        warning = pred_df.attrs.get("model_warning", "")
        if warning:
            figure.update_layout(title=f"{figure.layout.title.text} (warning)")
        return columns, data, figure, warning

    return app


def main():  # pragma: no cover - manual entry point
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
