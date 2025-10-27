"""Interactive Dash UI to explore bid acceptance predictions from an MLflow model."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import mlflow
import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from mlflow.exceptions import MlflowException
import plotly.graph_objects as go
from plotly import colors as plotly_colors

from .feature_config import _GROUPBY_KEY_FEATURES, load_feature_config
from .tuning.data_access import load_training_data


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

_USD_PERCENT_COLUMNS = {
    "usd_base_amount_25%": 0.25,
    "usd_base_amount_50%": 0.50,
    "usd_base_amount_75%": 0.75,
}

_BAR_COLOR_SEQUENCE = (
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


def _safe_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _normalize_offer_time(record: Dict[str, object]) -> None:
    offer_value = _safe_float(record.get("offer_time"))
    if offer_value is None:
        return
    record["offer_time"] = round(offer_value, 4)


def _refresh_usd_quantiles(record: Dict[str, object]) -> None:
    base_amount = _safe_float(record.get("usd_base_amount"))
    if base_amount is None:
        return
    record["usd_base_amount"] = base_amount
    for column, fraction in _USD_PERCENT_COLUMNS.items():
        record[column] = round(base_amount * fraction, 2)


def _prepare_bid_record(record: Dict[str, object]) -> Dict[str, object]:
    prepared = dict(record)
    prepared.pop("Acceptance Probability", None)
    _normalize_offer_time(prepared)
    _refresh_usd_quantiles(prepared)
    return prepared


def _compute_bid_label_map(df: pd.DataFrame) -> Tuple[Dict[object, int], Optional[str]]:
    """Build a mapping from bid identifier to the label index."""

    for column in ("id", "bid_id", "bid_number"):
        if column not in df.columns:
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        try:
            ordered = (
                pd.Series(values.unique())
                .sort_values(kind="mergesort")
                .tolist()
            )
        except Exception:
            ordered = (
                pd.Series(values.astype(str).unique())
                .sort_values(kind="mergesort")
                .tolist()
            )
        label_map = {value: index + 1 for index, value in enumerate(ordered)}
        return label_map, column
    return {}, None


def _apply_bid_labels(
    df: pd.DataFrame,
    label_map: Dict[object, int],
    label_column: Optional[str],
) -> pd.DataFrame:
    """Ensure the Bid # column reflects the provided identifier mapping."""

    if df.empty:
        return df

    working = df.copy()
    existing = working.get("Bid #")

    if label_map and label_column and label_column in working.columns:
        mapped = working[label_column].map(label_map)
        if existing is not None:
            mapped = mapped.fillna(existing)
        working["Bid #"] = mapped
    elif "Bid #" not in working.columns:
        working["Bid #"] = range(1, len(working) + 1)

    return working


def _sort_records_by_bid(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return records ordered by their bid label."""

    def sort_key(record: Dict[str, object]) -> Tuple[int, object]:
        label = record.get("Bid #")
        if label in (None, "") or pd.isna(label) or (
            isinstance(label, float) and math.isnan(label)
        ):
            return (1, "")
        try:
            return (0, float(label))
        except (TypeError, ValueError):
            return (0, str(label))

    return sorted(list(records), key=sort_key)


def _get_next_bid_label(records: Iterable[Dict[str, object]]) -> int:
    """Return the next available bid label given the existing records."""

    max_label = 0
    for record in records:
        label = record.get("Bid #")
        if label in (None, ""):
            continue
        try:
            value = int(float(label))
        except (TypeError, ValueError):
            continue
        max_label = max(max_label, value)
    return max_label + 1 if max_label > 0 else 1


def _prepare_prediction_dataframe(table_records: Iterable[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(list(table_records))
    if df.empty:
        return df

    # Attempt to convert numeric columns back to floats/ints when possible.
    for col in df.columns:
        if col in {
            "carrier_code",
            "flight_number",
            "fare_class",
            "offer_status",
            "upgrade_type",
            "Bid #",
        }:
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
    features, _ = _get_feature_columns()
    feature_df = df.reindex(columns=features)
    extra_columns = [col for col in df.columns if col not in feature_df.columns]
    for column in extra_columns:
        feature_df[column] = df[column]
    return feature_df


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
        snapshot_data = None
        if "snapshot_num" in grp_sorted.columns:
            snapshot_data = grp_sorted["snapshot_num"].astype(str)
        hover_template = "Time: %{x}<br>Probability: %{y:.4f}%"
        if snapshot_data is not None:
            hover_template = "Snapshot: %{customdata[0]}<br>" + hover_template
        fig.add_trace(
            go.Bar(
                x=grp_sorted["time_until_departure_hours"],
                y=grp_sorted["Acceptance Probability"],
                name=label,
                marker=dict(color=marker_color, line=dict(color=border_color, width=1.5)),
                customdata=None
                if snapshot_data is None
                else snapshot_data.to_numpy().reshape(-1, 1),
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
        acceptance = predictions["Acceptance Probability"].astype(float).to_numpy()
    else:
        # mlflow.pyfunc returns numpy array; expect probability in second column
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            acceptance = predictions[:, 1]
        else:
            acceptance = predictions
    acceptance_series = pd.Series(acceptance, index=df.index, dtype="float64") * 100.0
    df["Acceptance Probability"] = acceptance_series.round(4)
    if model_warning:
        df.attrs["model_warning"] = model_warning
    return df


# -- Dash application --------------------------------------------------------------------------


def create_app() -> Dash:
    default_dataset_path = "./data/air_canada_and_lot/evaluation_sets/eval_bid_data_snapshots_v2_testing.parquet"

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

        dataset = _load_dataset_cached(dataset_path)

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
            new_bid["Bid #"] = _get_next_bid_label(existing_records)
            new_bid.setdefault("offer_status", "pending")
            prepared_bid = _prepare_bid_record(new_bid)
            new_data = _sort_records_by_bid(existing_records + [prepared_bid])
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
                    restored_records.append(_prepare_bid_record(item.get("record", {})))
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
            working = _sort_records_by_bid(working_records + restored_records)
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
            working = _sort_records_by_bid(working)
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

        label_map, label_column = _compute_bid_label_map(subset)
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

        snapshot_df = _apply_bid_labels(snapshot_df, label_map, label_column)
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

        base_records = [_prepare_bid_record(record) for record in snapshot_df.to_dict("records")]
        base_data = _sort_records_by_bid(base_records)

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
            updated_records = _sort_records_by_bid(updated_records)
            current_len = len(updated_records)
            if offers_value == current_len:
                updated_meta["num_offers"] = offers_value
                for record in updated_records:
                    _normalize_offer_time(record)
                    _refresh_usd_quantiles(record)
                return updated_records, updated_meta
            if offers_value > current_len and current_len > 0:
                template = updated_records[0]
                for _ in range(offers_value - current_len):
                    new_bid = dict(template)
                    new_bid["Bid #"] = _get_next_bid_label(updated_records)
                    new_bid.setdefault("offer_status", "pending")
                    updated_records.append(_prepare_bid_record(new_bid))
            elif offers_value < current_len:
                updated_records = updated_records[: offers_value]
            updated_records = _sort_records_by_bid(updated_records)
            for record in updated_records:
                _normalize_offer_time(record)
                _refresh_usd_quantiles(record)
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
                    value = prediction_map.get(column_id)
                    if value is None:
                        row[column_id] = value
                    else:
                        try:
                            row[column_id] = round(float(value), 4)
                        except (TypeError, ValueError):
                            row[column_id] = value
                    continue
                value = record.get(feature)
                if feature == "fare_class":
                    row[column_id] = value
                elif feature == "item_count":
                    numeric = _safe_float(value)
                    row[column_id] = int(numeric) if numeric is not None else value
                elif feature == "offer_time":
                    numeric = _safe_float(value)
                    row[column_id] = round(numeric, 4) if numeric is not None else value
                elif feature == "usd_base_amount":
                    numeric = _safe_float(value)
                    row[column_id] = round(numeric, 2) if numeric is not None else value
                elif feature in _USD_PERCENT_COLUMNS:
                    numeric = _safe_float(value)
                    row[column_id] = round(numeric, 2) if numeric is not None else value
                elif feature.startswith("multiplier"):
                    numeric = _safe_float(value)
                    row[column_id] = round(numeric, 4) if numeric is not None else value
                else:
                    numeric = _safe_float(value)
                    row[column_id] = numeric if numeric is not None else value
            data_rows.append(row)

        style_rules.append(
            {
                "if": {"filter_query": '{Feature} = "Acceptance Probability"'},
                "fontWeight": "700",
                "backgroundColor": "#f1f5f9",
                "pointerEvents": "none",
            }
        )

        for percent_column in _USD_PERCENT_COLUMNS:
            style_rules.append(
                {
                    "if": {"filter_query": f'{{Feature}} = "{percent_column}"'},
                    "backgroundColor": "#f8fafc",
                    "pointerEvents": "none",
                }
            )

        return columns, data_rows, style_rules

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
                    value = value_row[column_id]
                    if feature == "fare_class":
                        record[feature] = value
                    elif feature == "item_count":
                        numeric = _safe_float(value)
                        record[feature] = int(numeric) if numeric is not None else value
                    elif feature == "offer_time":
                        numeric = _safe_float(value)
                        record[feature] = round(numeric, 4) if numeric is not None else value
                    elif feature == "usd_base_amount":
                        numeric = _safe_float(value)
                        record[feature] = numeric if numeric is not None else value
                    elif feature in _USD_PERCENT_COLUMNS:
                        # recomputed from usd_base_amount after loop
                        continue
                    elif feature.startswith("multiplier"):
                        numeric = _safe_float(value)
                        record[feature] = round(numeric, 4) if numeric is not None else value
                    else:
                        numeric = _safe_float(value)
                        record[feature] = numeric if numeric is not None else value
            _normalize_offer_time(record)
            _refresh_usd_quantiles(record)
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
            return _build_prediction_plot(pd.DataFrame()), {}, ""

        selected_df = _prepare_prediction_dataframe(records)

        if not model_uri:
            empty_fig = _build_prediction_plot(pd.DataFrame())
            return empty_fig, {}, "Load a model to generate acceptance probabilities."

        plot_source = pd.DataFrame()
        label_map: Dict[object, int] = {}
        label_column: Optional[str] = None
        if dataset_path and carrier and flight_number and travel_date and upgrade_type:
            try:
                dataset = _load_dataset_cached(dataset_path)
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
                    label_map, label_column = _compute_bid_label_map(plot_source)
                    plot_source = _apply_bid_labels(plot_source, label_map, label_column)
                    if "Bid #" in plot_source.columns:
                        plot_source = plot_source.sort_values("Bid #")
            except Exception:
                plot_source = pd.DataFrame()

        selected_snapshot = None
        if snapshot_meta:
            selected_snapshot = snapshot_meta.get("snapshot")
        selected_snapshot_value = str(selected_snapshot) if selected_snapshot is not None else None

        if label_map and label_column:
            selected_df = _apply_bid_labels(selected_df, label_map, label_column)

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
            plot_pred_df = _predict(model_uri, combined_df.copy())
            table_pred_df = _predict(model_uri, selected_df.copy())
        except Exception as exc:  # pragma: no cover - user feedback
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Prediction failed: {exc}")
            return empty_fig, {}, str(exc)

        figure = _build_prediction_plot(plot_pred_df)
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
