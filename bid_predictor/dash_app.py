"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import mlflow
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
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


DISPLAY_FEATURE_ROWS = [
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
]


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
            dcc.Store(id="bid-records-store"),
            dcc.Store(id="snapshot-meta-store"),
            dcc.Store(id="prediction-store"),
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
                                        ],
                                        style={"marginBottom": "0.75rem"},
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
        Output("snapshot-feedback", "children"),
        Output("snapshot-meta-store", "data"),
        Output("bid-records-store", "data"),
        Output("bid-table", "selected_columns"),
        Input("snapshot-dropdown", "value"),
        Input("add-bid", "n_clicks"),
        Input("delete-bid", "n_clicks"),
        State("bid-records-store", "data"),
        State("bid-table", "selected_columns"),
        State("carrier-dropdown", "value"),
        State("flight-number-dropdown", "value"),
        State("travel-date-dropdown", "value"),
        State("upgrade-dropdown", "value"),
        State("dataset-path-store", "data"),
        State("snapshot-meta-store", "data"),
    )
    def update_snapshot_view(
        snapshot_value: Optional[str],
        add_clicks: int,
        delete_clicks: int,
        existing_records: Optional[List[Dict[str, str]]],
        selected_columns: Optional[List[str]],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_path: Optional[str],
        snapshot_meta: Optional[Dict[str, str]],
    ):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

        if not dataset_path:
            return (
                html.Div("Load a dataset to begin."),
                "",
                None,
                None,
                [],
            )

        dataset = _load_dataset_cached(dataset_path)

        if not carrier or not flight_number or not travel_date or not upgrade_type:
            return (
                html.Div("Select a carrier, flight, travel date, and upgrade type."),
                "",
                snapshot_meta,
                existing_records,
                selected_columns or [],
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

        if triggered == "add-bid" and existing_records:
            base = existing_records[0].copy()
            new_bid = {key: base.get(key) for key in base}
            for feature in DISPLAY_FEATURE_ROWS:
                if feature != "Acceptance Probability":
                    new_bid.setdefault(feature, base.get(feature))
            existing_ids = [
                record.get("Bid #")
                for record in existing_records
                if record.get("Bid #") not in (None, "")
            ]
            next_bid = 1
            if existing_ids:
                try:
                    next_bid = int(max(float(bid) for bid in existing_ids)) + 1
                except Exception:
                    next_bid = len(existing_records) + 1
            new_bid["Bid #"] = next_bid
            new_bid.setdefault("offer_status", "pending")
            new_data = existing_records + [new_bid]
            new_meta = dict(snapshot_meta or {})
            new_meta["num_offers"] = len(new_data)
            return summary_block, "", new_meta, new_data, []

        if triggered == "delete-bid" and existing_records:
            if not selected_columns:
                return summary_block, "Select columns to delete.", snapshot_meta, existing_records, selected_columns or []
            indices_to_remove = sorted(
                {
                    int(col_id.replace("bid_", ""))
                    for col_id in selected_columns
                    if col_id.startswith("bid_")
                },
                reverse=True,
            )
            working = list(existing_records)
            for idx in indices_to_remove:
                if 0 <= idx < len(working):
                    working.pop(idx)
            for pos, record in enumerate(working, start=1):
                record["Bid #"] = pos
            new_meta = dict(snapshot_meta or {})
            new_meta["num_offers"] = len(working)
            return summary_block, "", new_meta, working, []

        if triggered != "snapshot-dropdown":
            return summary_block, "", snapshot_meta, existing_records, selected_columns or []

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
            )

        if "snapshot_num" not in subset.columns:
            return (
                summary_block,
                "Snapshot information is unavailable in this dataset.",
                None,
                None,
                [],
            )

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
            )

        if "Bid #" not in snapshot_df.columns:
            snapshot_df["Bid #"] = range(1, len(snapshot_df) + 1)

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

        base_data = snapshot_df.to_dict("records")

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

        return summary_block, "", snapshot_meta, base_data, []

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
            current_len = len(updated_records)
            if offers_value == current_len:
                updated_meta["num_offers"] = offers_value
                return updated_records, updated_meta
            if offers_value > current_len and current_len > 0:
                template = updated_records[0]
                for _ in range(offers_value - current_len):
                    new_bid = dict(template)
                    new_bid["Bid #"] = len(updated_records) + 1
                    new_bid.setdefault("offer_status", "pending")
                    updated_records.append(new_bid)
            elif offers_value < current_len:
                updated_records = updated_records[: offers_value]
            for pos, record in enumerate(updated_records, start=1):
                record["Bid #"] = pos
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
        if not records:
            columns = [
                {"name": "Feature", "id": "Feature", "editable": False},
            ]
            return columns, [], []

        columns = [
            {"name": "Feature", "id": "Feature", "editable": False},
        ]
        data_rows: List[Dict[str, object]] = []
        style_rules: List[Dict[str, object]] = []

        for idx, record in enumerate(records):
            bid_label = record.get("Bid #") or record.get("bid_number") or idx + 1
            column_id = f"bid_{idx}"
            columns.append({"name": f"Bid {bid_label}", "id": column_id, "editable": True})

        prediction_map = predictions or {}

        for feature in DISPLAY_FEATURE_ROWS:
            row = {"Feature": feature}
            for idx, record in enumerate(records):
                column_id = f"bid_{idx}"
                if feature == "Acceptance Probability":
                    row[column_id] = prediction_map.get(column_id)
                else:
                    row[column_id] = record.get(feature)
            data_rows.append(row)

        style_rules.append(
            {
                "if": {"filter_query": '{Feature} = "Acceptance Probability"'},
                "fontWeight": "700",
                "backgroundColor": "#f1f5f9",
                "pointerEvents": "none",
            }
        )

        return columns, data_rows, style_rules

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

        updated_records = [dict(record) for record in records]
        feature_map = {row.get("Feature"): row for row in table_data}
        bid_columns = [col for col in columns if col["id"] != "Feature"]

        for position, column in enumerate(bid_columns):
            column_id = column["id"]
            if position >= len(updated_records):
                continue
            record = updated_records[position]
            for feature in DISPLAY_FEATURE_ROWS:
                if feature == "Acceptance Probability":
                    continue
                value_row = feature_map.get(feature)
                if value_row is not None and column_id in value_row:
                    record[feature] = value_row[column_id]
        return updated_records

    @app.callback(
        Output("prediction-graph", "figure"),
        Output("prediction-store", "data"),
        Output("prediction-warning", "children"),
        Input("bid-records-store", "data"),
        Input("model-uri-store", "data"),
    )
    def run_predictions(
        records: Optional[List[Dict[str, str]]],
        model_uri: Optional[str],
    ):
        if not records:
            return _build_prediction_plot(pd.DataFrame()), {}, ""

        df = _prepare_prediction_dataframe(records)

        if not model_uri:
            empty_fig = _build_prediction_plot(pd.DataFrame())
            return empty_fig, {}, "Load a model to generate acceptance probabilities."

        try:
            pred_df = _predict(model_uri, df)
        except Exception as exc:  # pragma: no cover - user feedback
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Prediction failed: {exc}")
            return empty_fig, {}, str(exc)

        figure = _build_prediction_plot(pred_df)
        warning = pred_df.attrs.get("model_warning", "")

        predictions = {}
        for idx, _ in enumerate(records):
            column_id = f"bid_{idx}"
            predictions[column_id] = pred_df.iloc[idx].get("Acceptance Probability") if idx < len(pred_df) else None

        return figure, predictions, warning
    return app


def main():  # pragma: no cover - manual entry point
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
