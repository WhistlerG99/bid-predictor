"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

import json
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

    return mlflow.pyfunc.load_model(model_uri)


# -- Utility functions -------------------------------------------------------------------------


def _flight_label(row: pd.Series) -> str:
    travel_date = row.get("travel_date")
    if pd.notna(travel_date):
        if isinstance(travel_date, pd.Timestamp):
            travel_str = travel_date.date().isoformat()
        else:
            travel_str = str(travel_date)
    else:
        travel_str = "?"
    return f"{row['carrier_code']} {row['flight_number']} | {travel_str} | {row['upgrade_type']}"


def _make_flight_options(dataset: pd.DataFrame) -> List[Dict[str, str]]:
    grouped = (
        dataset[_GROUPBY_KEY_FEATURES[:-1]]
        .drop_duplicates()
        .sort_values(["carrier_code", "flight_number", "travel_date", "upgrade_type"])
    )
    options = []
    for _, row in grouped.iterrows():
        label = _flight_label(row)
        value = json.dumps({
            "carrier_code": row["carrier_code"],
            "flight_number": row["flight_number"],
            "travel_date": row["travel_date"].isoformat()
            if isinstance(row["travel_date"], pd.Timestamp)
            else str(row["travel_date"]),
            "upgrade_type": row["upgrade_type"],
        })
        options.append({"label": label, "value": value})
    return options


def _filter_flight(dataset: pd.DataFrame, selector: Dict[str, str]) -> pd.DataFrame:
    mask = pd.Series(True, index=dataset.index)
    for key, value in selector.items():
        if key == "travel_date":
            mask &= dataset[key].dt.date == pd.to_datetime(value).date()
        else:
            mask &= dataset[key] == value
    return dataset.loc[mask].sort_values("snapshot_num")


def _get_feature_columns() -> Tuple[List[str], List[str]]:
    feature_config = load_feature_config()
    features = list(feature_config["features"])
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

    bid_cols = [col for col in ["Bid #", "offer_status", "usd_base_amount", "item_count", "fare_class"] if col in work.columns]
    if bid_cols:
        work["bid_label"] = work[bid_cols].astype(str).agg(" | ".join, axis=1)
    else:
        work["bid_label"] = "Bid"

    # bar traces for each bid label
    for label, grp in work.groupby("bid_label"):
        grp_sorted = grp.sort_values("time_until_departure_hours")
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.5, xanchor="center"),
    )
    if "seats_available" in work.columns:
        fig.update_layout(yaxis2=dict(title="Seats available", overlaying="y", side="right"))
    return fig


def _predict(model_uri: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    model = _load_model_cached(model_uri)
    features, _ = _get_feature_columns()
    feature_df = df[features].copy()
    predictions = model.predict(feature_df)
    if isinstance(predictions, pd.DataFrame) and "Acceptance Probability" in predictions.columns:
        df["Acceptance Probability"] = predictions["Acceptance Probability"].values
    else:
        # mlflow.pyfunc returns numpy array; expect probability in second column
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            df["Acceptance Probability"] = predictions[:, 1]
        else:
            df["Acceptance Probability"] = predictions
    return df


# -- Dash application --------------------------------------------------------------------------


def create_app() -> Dash:
    feature_columns, _ = _get_feature_columns()
    default_dataset_path = resolve_train_file(None)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1("Bid Predictor Playground"),
            html.P(
                "Load a dataset snapshot and an MLflow-registered model to explore acceptance probabilities."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Dataset path"),
                            dcc.Input(
                                id="dataset-path",
                                type="text",
                                value=default_dataset_path,
                                placeholder="Path to bid_data_snapshots_v2.parquet",
                                style={"width": "100%"},
                            ),
                            html.Button("Load dataset", id="load-dataset", n_clicks=0),
                            html.Div(id="dataset-status", className="status-message"),
                        ],
                        style={"flex": "1", "padding": "0 1rem 1rem 0"},
                    ),
                    html.Div(
                        [
                            html.Label("MLflow tracking URI"),
                            dcc.Input(
                                id="mlflow-tracking-uri",
                                type="text",
                                value=mlflow.get_tracking_uri(),
                                placeholder="http://localhost:5000",
                                style={"width": "100%"},
                            ),
                            html.Label("Model name"),
                            dcc.Input(id="model-name", type="text", placeholder="Registered model name", style={"width": "100%"}),
                            html.Label("Model stage or version"),
                            dcc.Input(
                                id="model-stage",
                                type="text",
                                placeholder="e.g. Production or 5",
                                style={"width": "100%"},
                            ),
                            html.Button("Load model", id="load-model", n_clicks=0),
                            html.Div(id="model-status", className="status-message"),
                        ],
                        style={"flex": "1", "padding": "0 0 1rem 1rem"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),
            dcc.Store(id="dataset-path-store"),
            dcc.Store(id="model-uri-store"),
            html.Hr(),
            html.Div(
                [
                    html.Label("Flight selector"),
                    dcc.Dropdown(id="flight-dropdown", placeholder="Select a flight", options=[]),
                ]
            ),
            html.Div(id="flight-summary", style={"marginTop": "1rem"}),
            dash_table.DataTable(
                id="bid-table",
                columns=[{"name": col, "id": col, "editable": True} for col in feature_columns],
                data=[],
                editable=True,
                row_deletable=False,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "0.5rem"},
            ),
            html.Hr(),
            html.H3("Predictions"),
            dash_table.DataTable(
                id="prediction-table",
                columns=[],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "0.5rem"},
            ),
            dcc.Graph(id="prediction-graph"),
        ]
    )

    # Callbacks ---------------------------------------------------------------------------------

    @app.callback(
        Output("dataset-status", "children"),
        Output("dataset-path-store", "data"),
        Output("flight-dropdown", "options"),
        Output("flight-dropdown", "value"),
        Input("load-dataset", "n_clicks"),
        State("dataset-path", "value"),
        prevent_initial_call=True,
    )
    def load_dataset(n_clicks: int, path: str):
        if not path:
            return "Please provide a dataset path.", None, [], None

        try:
            dataset = _load_dataset_cached(path)
        except Exception as exc:  # pragma: no cover - user feedback
            return f"Failed to load dataset: {exc}", None, [], None

        options = _make_flight_options(dataset)
        status = f"Loaded dataset with {len(dataset):,} rows."
        return status, path, options, options[0]["value"] if options else None

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
        Output("bid-table", "columns"),
        Output("bid-table", "data"),
        Output("flight-summary", "children"),
        Input("flight-dropdown", "value"),
        State("dataset-path-store", "data"),
    )
    def update_bid_table(flight_value: Optional[str], dataset_path: Optional[str]):
        if not flight_value or not dataset_path:
            return [], [], "Load a dataset and choose a flight to begin."

        selector = json.loads(flight_value)
        dataset = _load_dataset_cached(dataset_path)
        subset = _filter_flight(dataset, selector)
        if subset.empty:
            return [], [], "No rows found for selected flight."

        columns = [
            {"name": col, "id": col, "editable": col not in _GROUPBY_KEY_FEATURES}
            for col in subset.columns
        ]
        data = subset.to_dict("records")
        summary = f"Loaded {len(subset)} bid snapshots. Edit the table to adjust features."
        return columns, data, summary

    @app.callback(
        Output("prediction-table", "columns"),
        Output("prediction-table", "data"),
        Output("prediction-graph", "figure"),
        Input("bid-table", "data"),
        State("model-uri-store", "data"),
    )
    def run_predictions(table_data: List[Dict[str, str]], model_uri: Optional[str]):
        if not table_data:
            return [], [], go.Figure()
        if not model_uri:
            df = pd.DataFrame(table_data)
            columns = [{"name": col, "id": col} for col in df.columns]
            return columns, table_data, _build_prediction_plot(pd.DataFrame())

        df = _prepare_prediction_dataframe(table_data)
        try:
            pred_df = _predict(model_uri, df)
        except Exception as exc:  # pragma: no cover - user feedback
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Prediction failed: {exc}")
            columns = [{"name": col, "id": col} for col in df.columns]
            return columns, table_data, empty_fig

        columns = [
            {"name": col, "id": col}
            if col == "Acceptance Probability"
            else {"name": col, "id": col}
            for col in pred_df.columns
        ]
        figure = _build_prediction_plot(pred_df)
        data = pred_df.to_dict("records")
        return columns, data, figure

    return app


def main():  # pragma: no cover - manual entry point
    app = create_app()
    app.run_server(debug=True)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
