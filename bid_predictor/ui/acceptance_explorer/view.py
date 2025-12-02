"""Callbacks and helpers for the acceptance probability explorer tab."""
from __future__ import annotations

import os
import re
from contextlib import closing
from copy import deepcopy
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import psycopg2
from dash import Dash, Input, Output, State, html, no_update
from dotenv import load_dotenv
from pyarrow import fs as pyfs

from ..feature_config import DEFAULT_UI_FEATURE_CONFIG
from ..formatting import prepare_bid_record
from ..plotting import build_prediction_plot
from ..tables import build_bid_table

DEFAULT_ACCEPTANCE_TABLE = "model_prediction_testing.audit_bid_predictor"
load_dotenv()

_ACCEPTANCE_TABLE_FEATURES = [
    "offer_id",
    "conf_num",
    "days_before_departure",
    "hours_before_departure",
    "seats_available",
    "created_timestamp",
    "usd_base_amount_25%",
    "usd_base_amount_50%",
    "usd_base_amount_75%",
    "usd_base_amount_max",
    "num_offers",
    "bid_rank",
    "Current Time",
]


def _acceptance_feature_config() -> Dict[str, List[str]]:
    config = deepcopy(DEFAULT_UI_FEATURE_CONFIG)
    priority_features = ["offer_id", "conf_num"]
    tail_features = ["hours_before_departure", "Current Time", "Acceptance Probability"]

    display_features: List[str] = []
    seen: set[str] = set()

    for feature in priority_features:
        if feature in seen:
            continue
        seen.add(feature)
        display_features.append(feature)

    base_candidates = list(config.get("display_features", [])) + _ACCEPTANCE_TABLE_FEATURES
    for feature in base_candidates:
        if feature in tail_features or feature in seen:
            continue
        seen.add(feature)
        display_features.append(feature)

    for feature in tail_features:
        if feature in seen:
            continue
        seen.add(feature)
        display_features.append(feature)
    config["display_features"] = display_features

    readonly = []
    seen_readonly: set[str] = set()
    for feature in list(config.get("readonly_features", [])) + _ACCEPTANCE_TABLE_FEATURES:
        if feature in seen_readonly:
            continue
        seen_readonly.add(feature)
        readonly.append(feature)
    config["readonly_features"] = readonly
    return config


def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _parse_recent_hours(hours_value: object) -> Optional[int]:
    if hours_value in (None, ""):
        return None
    try:
        hours = int(hours_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Recent hours must be an integer.") from exc
    if hours <= 0:
        raise ValueError("Recent hours must be positive.")
    return hours


def _list_remote_files(filesystem: pyfs.FileSystem, uri: str) -> List[str]:
    relative_path = uri.replace("s3://", "", 1)
    info = filesystem.get_file_info([relative_path])[0]
    if info.type == pyfs.FileType.NotFound:
        raise FileNotFoundError(f"Dataset path does not exist: {uri}")

    if info.type == pyfs.FileType.File:
        return [relative_path]

    if info.type == pyfs.FileType.Directory:
        selector = pyfs.FileSelector(relative_path, recursive=False)
        entries = filesystem.get_file_info(selector)
        files = [
            entry.path
            for entry in entries
            if entry.type == pyfs.FileType.File
            and PurePosixPath(entry.path).suffix.lower() in {".parquet", ".pq", ".csv"}
        ]
        if not files:
            raise ValueError(
                "No parquet or CSV files found in the provided directory."
            )
        return sorted(files)

    raise ValueError(f"Unsupported S3 path type for {uri}")


_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


def _extract_timestamp_from_name(name: str) -> Optional[pd.Timestamp]:
    match = _TIMESTAMP_PATTERN.search(name)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%Y-%m-%dT%H-%M-%S")
    except (TypeError, ValueError):
        return None


def _filter_files_by_recent_hours(files: Sequence[str], hours: Optional[int]) -> List[str]:
    if hours is None:
        return sorted(files)

    dated_files: List[Tuple[str, Optional[pd.Timestamp]]] = []
    for file_path in files:
        timestamp = _extract_timestamp_from_name(PurePosixPath(file_path).name)
        dated_files.append((file_path, timestamp))

    timestamps = [ts for _, ts in dated_files if ts is not None and not pd.isna(ts)]
    if not timestamps:
        return sorted(files)

    latest_timestamp = max(timestamps)
    threshold = latest_timestamp - pd.Timedelta(hours=hours)
    filtered = [path for path, ts in dated_files if ts is not None and ts >= threshold]
    return sorted(filtered or [path for path, _ in dated_files])


def _scale_acceptance_probabilities(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_na = numeric.dropna()
    if not non_na.empty and non_na.max() <= 1:
        numeric = numeric * 100.0
    return numeric


def _validate_table_name(table_name: str) -> str:
    if not table_name or not re.match(r"^[A-Za-z0-9_.]+$", table_name):
        raise ValueError(
            "Table name must include only letters, numbers, underscores, or periods."
        )
    return table_name


def _redshift_credentials() -> Dict[str, str]:
    required = [
        "REDSHIFT_HOST",
        "REDSHIFT_DATABASE",
        "REDSHIFT_USER",
        "REDSHIFT_PASSWORD",
        "REDSHIFT_PORT",
    ]
    values = {name: os.getenv(name) for name in required}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise EnvironmentError(
            "Missing environment variables for Redshift connection: "
            + ", ".join(missing)
        )
    return {name: str(values[name]) for name in required}


def _normalize_acceptance_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        raise ValueError("The loaded dataset is empty.")

    base_amount_aliases = {
        "usd_base_amount_25_percent": "usd_base_amount_25%",
        "usd_base_amount_50_percent": "usd_base_amount_50%",
        "usd_base_amount_75_percent": "usd_base_amount_75%",
    }
    for alias, canonical in base_amount_aliases.items():
        if canonical not in dataset.columns and alias in dataset.columns:
            dataset[canonical] = dataset[alias]

    if "accept_prob_timestamp" in dataset.columns:
        timestamps = pd.to_datetime(dataset["accept_prob_timestamp"], errors="coerce")
        dataset["current_timestamp"] = timestamps
        if "snapshot_num" not in dataset.columns:
            ordered = sorted({ts for ts in timestamps.dropna().unique()})
            mapping = {value: idx + 1 for idx, value in enumerate(ordered)}
            dataset["snapshot_num"] = timestamps.map(mapping)
    if "travel_date" not in dataset.columns and "departure_timestamp" in dataset.columns:
        dataset["travel_date"] = pd.to_datetime(
            dataset["departure_timestamp"], errors="coerce"
        ).dt.date
    if "hours_before_departure" not in dataset.columns and "days_before_departure" in dataset.columns:
        days = pd.to_numeric(dataset["days_before_departure"], errors="coerce")
        dataset["hours_before_departure"] = days * 24
    if "Bid #" not in dataset.columns and "offer_id" in dataset.columns:
        group_keys = [
            key
            for key in [
                "carrier_code",
                "flight_number",
                "travel_date",
                "upgrade_type",
            ]
            if key in dataset.columns
        ]
        if group_keys:
            dataset["Bid #"] = pd.NA
            for _, group in dataset.groupby(group_keys, sort=False):
                offers = group["offer_id"]
                try:
                    ordered_offers = (
                        pd.Series(offers.dropna().unique())
                        .sort_values(kind="mergesort")
                        .tolist()
                    )
                except Exception:
                    ordered_offers = (
                        pd.Series(offers.dropna().astype(str).unique())
                        .sort_values(kind="mergesort")
                        .tolist()
                    )
                label_map = {value: idx + 1 for idx, value in enumerate(ordered_offers)}
                dataset.loc[group.index, "Bid #"] = offers.map(label_map)

            missing = dataset["Bid #"].isna()
            if missing.any():
                dataset.loc[missing, "Bid #"] = pd.factorize(dataset.loc[missing, "offer_id"])[0] + 1
            dataset["Bid #"] = dataset["Bid #"].astype(int)
        else:
            dataset["Bid #"] = pd.factorize(dataset["offer_id"])[0] + 1
    probability_source = None
    if "Acceptance Probability" in dataset.columns:
        probability_source = dataset["Acceptance Probability"]
    elif "acceptance_prob" in dataset.columns:
        probability_source = dataset["acceptance_prob"]
    elif "accept_prob" in dataset.columns:
        probability_source = dataset["accept_prob"]

    if probability_source is not None:
        scaled = _scale_acceptance_probabilities(probability_source)
        dataset["Acceptance Probability"] = scaled
        dataset["acceptance_prob"] = scaled
        dataset["accept_prob"] = scaled

    dataset["offer_status"] = dataset.get("offer_status", "pending")
    return dataset


def _select_first_series(
    dataset: pd.DataFrame, columns: Iterable[str]
) -> Optional[pd.Series]:
    for column in columns:
        if column in dataset.columns:
            return dataset[column]
    return None


@lru_cache(maxsize=4)
def _load_acceptance_dataset_from_path(path: str, hours: Optional[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if _is_s3_path(path):
        filesystem = pyfs.S3FileSystem()
        files = _list_remote_files(filesystem, path)
        filtered_files = _filter_files_by_recent_hours(files, hours)
        for remote_path in filtered_files:
            suffix = PurePosixPath(remote_path).suffix.lower()
            with filesystem.open_input_file(remote_path) as handle:
                if suffix in {".parquet", ".pq"}:
                    frames.append(pd.read_parquet(handle))
                elif suffix == ".csv":
                    frames.append(pd.read_csv(handle))
                else:
                    raise ValueError(
                        f"Unsupported file extension for s3://{remote_path}"
                    )
    else:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")

        if resolved.is_dir():
            parquet_files = sorted(resolved.glob("*.parquet")) + sorted(
                resolved.glob("*.pq")
            )
            csv_files = sorted(resolved.glob("*.csv"))
            files = parquet_files + csv_files
            if not files:
                raise ValueError(
                    "No parquet or CSV files found in the provided directory."
                )
            filtered_files = _filter_files_by_recent_hours(
                [str(path) for path in files], hours
            )
            files = [Path(file_path) for file_path in filtered_files]
        else:
            files = [resolved]

        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                frames.append(pd.read_parquet(file_path))
            elif suffix == ".csv":
                frames.append(pd.read_csv(file_path))
            else:
                raise ValueError(f"Unsupported file extension for {file_path}")

    dataset = pd.concat(frames, ignore_index=True, sort=False)
    return _normalize_acceptance_dataset(dataset)


@lru_cache(maxsize=4)
def _load_acceptance_dataset_from_redshift(
    table_name: str, hours: Optional[int]
) -> pd.DataFrame:
    validated_table = _validate_table_name(table_name)
    credentials = _redshift_credentials()
    query = f"SELECT * FROM {validated_table}"
    params: Tuple[object, ...] = ()
    if hours is not None:
        query += " WHERE accept_prob_timestamp >= DATEADD(hour, -%s, GETDATE())"
        params = (hours,)

    connection = psycopg2.connect(
        host=credentials["REDSHIFT_HOST"],
        dbname=credentials["REDSHIFT_DATABASE"],
        user=credentials["REDSHIFT_USER"],
        password=credentials["REDSHIFT_PASSWORD"],
        port=int(credentials["REDSHIFT_PORT"]),
    )
    with closing(connection) as conn:
        frame = pd.read_sql_query(query, conn, params=params or None)
    return _normalize_acceptance_dataset(frame)


def load_acceptance_dataset(config: str | Mapping[str, object]) -> pd.DataFrame:
    """Load acceptance data from a file path or Redshift table."""

    if isinstance(config, str):
        return _load_acceptance_dataset_from_path(config, None)

    if not config:
        raise ValueError("No acceptance dataset source provided.")

    source = str(config.get("source") or "path").lower()
    if source == "redshift":
        table = str(config.get("table") or DEFAULT_ACCEPTANCE_TABLE)
        hours = _parse_recent_hours(config.get("hours"))
        return _load_acceptance_dataset_from_redshift(table, hours)

    if source != "path":
        raise ValueError(f"Unsupported acceptance dataset source: {source}")

    path_value = config.get("path")
    if not path_value:
        raise ValueError("Please provide a dataset path.")

    hours = _parse_recent_hours(config.get("hours"))

    return _load_acceptance_dataset_from_path(str(path_value), hours)


def _options_from_series(values: pd.Series) -> List[dict]:
    return [
        {"label": str(value), "value": str(value)}
        for value in values.dropna().drop_duplicates().sort_values()
    ]


def _build_summary(
    carrier: Optional[str],
    flight_number: Optional[str],
    travel_date: Optional[str],
    upgrade: Optional[str],
) -> html.Div:
    return html.Ul(
        [
            html.Li(f"Carrier: {carrier}") if carrier else html.Li("Carrier not set"),
            html.Li(f"Flight: {flight_number}")
            if flight_number
            else html.Li("Flight not set"),
            html.Li(f"Travel date: {travel_date}")
            if travel_date
            else html.Li("Travel date not set"),
            html.Li(f"Upgrade type: {upgrade}")
            if upgrade
            else html.Li("Upgrade type not set"),
        ],
        style={"paddingLeft": "1.2rem", "margin": "0"},
    )


def _prepare_records(snapshot_df: pd.DataFrame) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for record in snapshot_df.to_dict("records"):
        cleaned = dict(record)
        if "Acceptance Probability" not in cleaned:
            cleaned["Acceptance Probability"] = cleaned.get("acceptance_prob") or cleaned.get(
                "accept_prob"
            )
        if "Current Time" not in cleaned:
            cleaned["Current Time"] = cleaned.get("current_timestamp") or cleaned.get(
                "accept_prob_timestamp"
            )
        current_value = cleaned.get("Current Time")
        if pd.isna(current_value):
            cleaned["Current Time"] = ""
        elif hasattr(current_value, "isoformat"):
            cleaned["Current Time"] = current_value.isoformat()
        records.append(prepare_bid_record(cleaned))
    return records


def _build_predictions(df: pd.DataFrame) -> Dict[str, object]:
    probability_map: Dict[str, object] = {}
    probabilities = _select_first_series(
        df, ["Acceptance Probability", "acceptance_prob", "accept_prob"]
    )
    if probabilities is None:
        probabilities = []
    for idx, value in enumerate(probabilities):
        probability_map[f"bid_{idx}"] = value
    return {"probabilities": probability_map, "derived_features": []}


def _render_table(
    records: Optional[List[Dict[str, object]]],
    predictions: Dict[str, object],
    feature_config: Optional[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    probability_map: Dict[str, object] = {}
    derived_values = None
    if isinstance(predictions, dict):
        if "probabilities" in predictions or "derived_features" in predictions:
            probability_map = dict(predictions.get("probabilities", {}))
            derived_values = predictions.get("derived_features")
        else:
            probability_map = dict(predictions)

    columns, data_rows, style_rules = build_bid_table(
        records,
        probability_map,
        feature_config=feature_config,
        derived_feature_values=derived_values,
        show_comp_features=True,
    )
    for column in columns:
        column["editable"] = False
    for column in columns:
        style_rules.append(
            {"if": {"column_id": column.get("id")}, "pointerEvents": "none"}
        )
    style_rules.append(
        {"if": {"row_index": "odd"}, "backgroundColor": "#f3f4f6"}
    )
    style_rules.append(
        {"if": {"row_index": "even"}, "backgroundColor": "#ffffff"}
    )
    return columns, data_rows, style_rules


def register_acceptance_callbacks(app: Dash) -> None:
    """Register callbacks for loading and viewing acceptance datasets."""

    @app.callback(
        Output("acceptance-carrier-dropdown", "options"),
        Output("acceptance-carrier-dropdown", "value"),
        Input("acceptance-dataset-path-store", "data"),
    )
    def populate_carriers(dataset_config: Optional[object]):
        if not dataset_config:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "carrier_code" not in dataset.columns:
            return [], None
        options = _options_from_series(dataset["carrier_code"])
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("acceptance-flight-number-dropdown", "options"),
        Output("acceptance-flight-number-dropdown", "value"),
        Input("acceptance-carrier-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def populate_flights(carrier: Optional[str], dataset_config: Optional[object]):
        if not dataset_config or not carrier:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if not {"carrier_code", "flight_number"}.issubset(dataset.columns):
            return [], None
        mask = dataset["carrier_code"] == carrier
        options = _options_from_series(dataset.loc[mask, "flight_number"].astype(str))
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("acceptance-travel-date-dropdown", "options"),
        Output("acceptance-travel-date-dropdown", "value"),
        Input("acceptance-flight-number-dropdown", "value"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def populate_travel_dates(
        flight_number: Optional[str],
        carrier: Optional[str],
        dataset_config: Optional[object],
    ):
        if not dataset_config or not carrier or not flight_number:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "travel_date" not in dataset.columns:
            return [], None
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
        )
        travel_dates = pd.to_datetime(dataset.loc[mask, "travel_date"], errors="coerce")
        options = [
            {"label": date.strftime("%Y-%m-%d"), "value": date.strftime("%Y-%m-%d")}
            for date in travel_dates.dropna().drop_duplicates().sort_values()
        ]
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("acceptance-upgrade-dropdown", "options"),
        Output("acceptance-upgrade-dropdown", "value"),
        Input("acceptance-travel-date-dropdown", "value"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-flight-number-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def populate_upgrades(
        travel_date: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        dataset_config: Optional[object],
    ):
        if not dataset_config or not carrier or not flight_number or not travel_date:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "upgrade_type" not in dataset.columns:
            return [], None
        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
        )
        options = _options_from_series(dataset.loc[mask, "upgrade_type"])
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("acceptance-snapshot-dropdown", "options"),
        Output("acceptance-snapshot-dropdown", "value"),
        Input("acceptance-upgrade-dropdown", "value"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-flight-number-dropdown", "value"),
        State("acceptance-travel-date-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def populate_snapshots(
        upgrade: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        dataset_config: Optional[object],
    ):
        if not dataset_config or not carrier or not flight_number or not travel_date or not upgrade:
            return [], None
        dataset = load_acceptance_dataset(dataset_config)
        if "snapshot_num" not in dataset.columns:
            return [], None
        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade)
        )
        snapshots = dataset.loc[mask, "snapshot_num"].astype(str)
        options = _options_from_series(snapshots)
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("acceptance-departure-timestamp", "children"),
        Output("acceptance-origination-code", "children"),
        Output("acceptance-destination-code", "children"),
        Input("acceptance-dataset-path-store", "data"),
        Input("acceptance-carrier-dropdown", "value"),
        Input("acceptance-flight-number-dropdown", "value"),
        Input("acceptance-travel-date-dropdown", "value"),
        Input("acceptance-upgrade-dropdown", "value"),
    )
    def populate_route_details(
        dataset_config: Optional[object],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade: Optional[str],
    ):
        if not dataset_config or not carrier or not flight_number:
            return "–", "–", "–"

        dataset = load_acceptance_dataset(dataset_config)
        if not {"carrier_code", "flight_number"}.issubset(dataset.columns):
            return "–", "–", "–"

        mask = (dataset["carrier_code"] == carrier) & (
            dataset["flight_number"].astype(str) == str(flight_number)
        )
        if travel_date and "travel_date" in dataset.columns:
            travel_date_dt = pd.to_datetime(travel_date).date()
            mask &= pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt
        if upgrade and "upgrade_type" in dataset.columns:
            mask &= dataset["upgrade_type"] == upgrade

        subset = dataset.loc[mask]
        if subset.empty:
            return "–", "–", "–"

        origin = subset.iloc[0].get("origination_code", "–")
        destination = subset.iloc[0].get("destination_code", "–")
        departure = subset.iloc[0].get("departure_timestamp", "–")
        if hasattr(departure, "isoformat"):
            departure = departure.isoformat()
        return departure or "–", origin or "–", destination or "–"

    @app.callback(
        Output("acceptance-flight-summary", "children"),
        Output("acceptance-prediction-graph", "figure"),
        Output("acceptance-warning", "children"),
        Output("acceptance-bid-table", "columns"),
        Output("acceptance-bid-table", "data"),
        Output("acceptance-bid-table", "style_data_conditional"),
        Output("acceptance-table-feedback", "children"),
        Input("acceptance-snapshot-dropdown", "value"),
        State("acceptance-carrier-dropdown", "value"),
        State("acceptance-flight-number-dropdown", "value"),
        State("acceptance-travel-date-dropdown", "value"),
        State("acceptance-upgrade-dropdown", "value"),
        State("acceptance-dataset-path-store", "data"),
    )
    def render_view(
        snapshot_value: Optional[str],
        carrier: Optional[str],
        flight_number: Optional[str],
        travel_date: Optional[str],
        upgrade_type: Optional[str],
        dataset_config: Optional[object],
    ):
        summary = _build_summary(carrier, flight_number, travel_date, upgrade_type)
        if not dataset_config:
            return summary, build_prediction_plot(pd.DataFrame()), "Load a dataset to begin.", no_update, [], [], ""

        if not all([carrier, flight_number, travel_date, upgrade_type]):
            return summary, build_prediction_plot(pd.DataFrame()), "", no_update, [], [], "Select a carrier, flight, travel date, and upgrade type."

        dataset = load_acceptance_dataset(dataset_config)
        required_columns = {
            "carrier_code",
            "flight_number",
            "travel_date",
            "upgrade_type",
        }
        if not required_columns.issubset(dataset.columns):
            empty_fig = build_prediction_plot(pd.DataFrame())
            return (
                summary,
                empty_fig,
                "Dataset is missing required columns for this view.",
                no_update,
                [],
                [],
                "",
            )

        travel_date_dt = pd.to_datetime(travel_date).date()
        mask = (
            (dataset["carrier_code"] == carrier)
            & (dataset["flight_number"].astype(str) == str(flight_number))
            & (pd.to_datetime(dataset["travel_date"]).dt.date == travel_date_dt)
            & (dataset["upgrade_type"] == upgrade_type)
        )
        subset = dataset.loc[mask].copy()
        if subset.empty:
            empty_fig = build_prediction_plot(pd.DataFrame())
            return summary, empty_fig, "No rows found for the selected flight.", no_update, [], [], ""

        subset["snapshot_num"] = subset.get("snapshot_num").astype(str)
        probability_series = _select_first_series(
            subset, ["Acceptance Probability", "acceptance_prob", "accept_prob"]
        )
        if probability_series is not None:
            subset["Acceptance Probability"] = probability_series
        if "Current Time" not in subset.columns:
            if "accept_prob_timestamp" in subset.columns:
                subset["Current Time"] = pd.to_datetime(
                    subset["accept_prob_timestamp"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
            elif "current_timestamp" in subset.columns:
                subset["Current Time"] = pd.to_datetime(
                    subset["current_timestamp"], errors="coerce"
                ).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        graph_df = subset.copy()
        if "accept_prob_timestamp" in graph_df.columns and "current_timestamp" not in graph_df.columns:
            graph_df["current_timestamp"] = pd.to_datetime(
                graph_df["accept_prob_timestamp"], errors="coerce"
            )

        figure = build_prediction_plot(graph_df)
        warning = "" if snapshot_value else "Select a snapshot to view bid details."

        if not snapshot_value:
            return summary, figure, warning, no_update, [], [], ""

        snapshot_df = subset.loc[subset["snapshot_num"] == str(snapshot_value)].copy()
        snapshot_df = snapshot_df.reset_index(drop=True)
        if "Bid #" in snapshot_df.columns:
            snapshot_df = snapshot_df.sort_values("Bid #", kind="mergesort")
            snapshot_df = snapshot_df.reset_index(drop=True)
        if snapshot_df.empty:
            return summary, figure, "No rows found for the selected snapshot.", no_update, [], [], ""

        predictions = _build_predictions(snapshot_df)
        records = _prepare_records(snapshot_df)
        columns, data_rows, style_rules = _render_table(
            records, predictions, _acceptance_feature_config()
        )

        return summary, figure, warning, columns, data_rows, style_rules, ""


__all__ = ["register_acceptance_callbacks", "load_acceptance_dataset"]
