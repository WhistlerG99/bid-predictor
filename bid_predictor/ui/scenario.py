"""Scenario exploration helpers for the Dash UI."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .formatting import apply_bid_labels, compute_bid_label_map
from .plotting import BAR_COLOR_SEQUENCE

_TIME_TO_DEPARTURE_KEY = "__time_to_departure_hours__"


@dataclass(frozen=True)
class ScenarioFeature:
    """Representation of a feature that can be adjusted in the scenario tab."""

    key: str
    scope: str
    label: str
    bid_label: Optional[int] = None
    is_integer: bool = False
    kind: str = "numeric"

    def encode(self) -> str:
        payload = {
            "key": self.key,
            "scope": self.scope,
            "label": self.label,
            "bid_label": self.bid_label,
            "is_integer": self.is_integer,
            "kind": self.kind,
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def decode(value: Optional[str]) -> Optional["ScenarioFeature"]:
        if not value:
            return None
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return None
        return ScenarioFeature(
            key=payload.get("key", ""),
            scope=payload.get("scope", ""),
            label=payload.get("label", ""),
            bid_label=payload.get("bid_label"),
            is_integer=bool(payload.get("is_integer", False)),
            kind=payload.get("kind", "numeric"),
        )


@dataclass(frozen=True)
class ScenarioRange:
    """Describes the default slider configuration for a scenario feature."""

    min_value: float
    max_value: float
    step: float
    count: int
    base_value: float


def build_carrier_options(dataset: pd.DataFrame) -> List[Dict[str, str]]:
    """Return carrier dropdown options for the scenario explorer."""

    if dataset.empty or "carrier_code" not in dataset.columns:
        return []

    carriers = dataset["carrier_code"].dropna().drop_duplicates().sort_values()
    return [{"label": str(code), "value": str(code)} for code in carriers]


def build_flight_number_options(dataset: pd.DataFrame, carrier: Optional[str]) -> List[Dict[str, str]]:
    """Return flight number options filtered by carrier."""

    if dataset.empty or not carrier:
        return []
    required = {"carrier_code", "flight_number"}
    if not required.issubset(dataset.columns):
        return []

    mask = dataset["carrier_code"] == carrier
    flights = (
        dataset.loc[mask, "flight_number"].dropna().astype(str).drop_duplicates().sort_values()
    )
    return [{"label": value, "value": value} for value in flights]


def build_travel_date_options(
    dataset: pd.DataFrame,
    carrier: Optional[str],
    flight_number: Optional[str],
) -> List[Dict[str, str]]:
    """Return travel date options filtered by carrier and flight number."""

    if dataset.empty or not carrier or not flight_number:
        return []

    required = {"carrier_code", "flight_number", "travel_date"}
    if not required.issubset(dataset.columns):
        return []

    mask = (dataset["carrier_code"] == carrier) & (
        dataset["flight_number"].astype(str) == str(flight_number)
    )
    dates = (
        pd.to_datetime(dataset.loc[mask, "travel_date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return [
        {"label": dt.date().isoformat(), "value": dt.date().isoformat()}
        for dt in dates
    ]


def build_upgrade_options(
    dataset: pd.DataFrame,
    carrier: Optional[str],
    flight_number: Optional[str],
    travel_date: Optional[str],
) -> List[Dict[str, str]]:
    """Return upgrade type options filtered by the selected flight."""

    if dataset.empty or not carrier or not flight_number or not travel_date:
        return []

    required = {"carrier_code", "flight_number", "travel_date", "upgrade_type"}
    if not required.issubset(dataset.columns):
        return []

    travel_date_dt = pd.to_datetime(travel_date, errors="coerce")
    if pd.isna(travel_date_dt):
        return []

    mask = (
        (dataset["carrier_code"] == carrier)
        & (dataset["flight_number"].astype(str) == str(flight_number))
        & (pd.to_datetime(dataset["travel_date"], errors="coerce").dt.date == travel_date_dt.date())
    )
    upgrades = (
        dataset.loc[mask, "upgrade_type"].dropna().drop_duplicates().sort_values()
    )
    return [{"label": str(value), "value": str(value)} for value in upgrades]


def extract_baseline_snapshot(
    dataset: pd.DataFrame,
    carrier: Optional[str],
    flight_number: Optional[str],
    travel_date: Optional[str],
    upgrade_type: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str]]:
    if not carrier or not flight_number or not travel_date or not upgrade_type:
        return pd.DataFrame(), None

    required = {
        "carrier_code",
        "flight_number",
        "travel_date",
        "upgrade_type",
    }
    if not required.issubset(dataset.columns):
        return pd.DataFrame(), None

    travel_date_dt = pd.to_datetime(travel_date, errors="coerce")
    if pd.isna(travel_date_dt):
        return pd.DataFrame(), None

    mask = (
        (dataset["carrier_code"] == carrier)
        & (dataset["flight_number"].astype(str) == str(flight_number))
        & (pd.to_datetime(dataset["travel_date"], errors="coerce").dt.date == travel_date_dt.date())
        & (dataset["upgrade_type"] == upgrade_type)
    )
    subset = dataset.loc[mask].copy()
    if subset.empty:
        return subset, None

    snapshot_label: Optional[str] = None
    if "snapshot_num" in subset.columns:
        snapshot_numbers = pd.to_numeric(subset["snapshot_num"], errors="coerce")
        if snapshot_numbers.notna().any():
            latest = snapshot_numbers.max()
            snapshot_label = str(int(latest)) if float(latest).is_integer() else str(latest)
            subset = subset.loc[snapshot_numbers == latest].copy()

    label_map, label_column = compute_bid_label_map(subset)
    subset = apply_bid_labels(subset, label_map, label_column)
    if "Bid #" in subset.columns:
        subset = subset.sort_values("Bid #")
    else:
        subset = subset.reset_index(drop=True)
        subset["Bid #"] = range(1, len(subset) + 1)
    return subset, snapshot_label


def _coerce_numeric(series: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    return None


def _infer_is_integer(series: pd.Series) -> bool:
    valid = series.dropna()
    if valid.empty:
        return False
    return bool(np.allclose(valid, valid.round()))


def _compute_time_to_departure_hours(df: pd.DataFrame) -> Optional[pd.Series]:
    if "departure_timestamp" not in df.columns or "current_timestamp" not in df.columns:
        return None
    departure = pd.to_datetime(df["departure_timestamp"], errors="coerce")
    current = pd.to_datetime(df["current_timestamp"], errors="coerce")
    if departure.isna().all() or current.isna().all():
        return None
    delta = (departure - current).dt.total_seconds() / 3600.0
    return delta


def build_feature_options(df: pd.DataFrame) -> List[ScenarioFeature]:
    if df.empty:
        return []

    options: List[ScenarioFeature] = []
    numeric_candidates = [
        "item_count",
        "usd_base_amount",
        "offer_time",
        "multiplier_fare_class",
        "multiplier_loyalty",
        "multiplier_success_history",
        "multiplier_payment_type",
        "usd_total_amount",
    ]
    global_candidates = [
        "seats_available",
        "available_inventory",
    ]

    available_columns = set(df.columns)
    label_series = df.get("Bid #")

    # Global numeric features that are consistent across bids
    for column in global_candidates:
        if column not in available_columns:
            continue
        numeric = _coerce_numeric(df[column])
        if numeric is None:
            continue
        label = column.replace("_", " ").title()
        options.append(
            ScenarioFeature(
                key=column,
                scope="global",
                label=label,
                bid_label=None,
                is_integer=_infer_is_integer(numeric),
            )
        )

    # Derived time to departure feature
    time_to_departure = _compute_time_to_departure_hours(df)
    if time_to_departure is not None and not time_to_departure.isna().all():
        options.append(
            ScenarioFeature(
                key=_TIME_TO_DEPARTURE_KEY,
                scope="global",
                label="Time to departure (hours)",
                bid_label=None,
                is_integer=False,
                kind="time_to_departure",
            )
        )

    # Bid specific numeric features
    if label_series is not None and not label_series.dropna().empty:
        for column in numeric_candidates:
            if column not in available_columns:
                continue
            numeric = _coerce_numeric(df[column])
            if numeric is None:
                continue
            is_integer = _infer_is_integer(numeric)
            for bid_value in sorted(label_series.dropna().unique()):
                label = f"Bid {int(bid_value)} – {column.replace('_', ' ')}"
                options.append(
                    ScenarioFeature(
                        key=column,
                        scope="bid",
                        label=label,
                        bid_label=int(bid_value),
                        is_integer=is_integer,
                    )
                )

    return options


def select_feature(options: Sequence[ScenarioFeature], value: Optional[str]) -> Optional[ScenarioFeature]:
    decoded = ScenarioFeature.decode(value)
    if not decoded:
        return None
    for feature in options:
        if feature.encode() == decoded.encode():
            return feature
    return None


def compute_default_range(df: pd.DataFrame, feature: ScenarioFeature) -> Optional[ScenarioRange]:
    if df.empty:
        return None

    if feature.kind == "time_to_departure":
        series = _compute_time_to_departure_hours(df)
        if series is None:
            return None
        series = series.dropna()
        if series.empty:
            return None
        base_value = float(series.iloc[0])
        min_value = float(series.min())
        max_value = float(series.max())
        if math.isclose(min_value, max_value):
            delta = max(abs(base_value) * 0.25, 6.0)
            min_value = max(base_value - delta, 0.0)
            max_value = base_value + delta
        step = max((max_value - min_value) / 20.0, 0.5)
        count = max(int(round((max_value - min_value) / max(step, 1e-6))) + 1, 5)
        return ScenarioRange(min_value=min_value, max_value=max_value, step=step, count=count, base_value=base_value)

    column = feature.key
    if column not in df.columns:
        return None
    numeric = _coerce_numeric(df[column])
    if numeric is None:
        return None

    if feature.scope == "bid" and feature.bid_label is not None and "Bid #" in df.columns:
        mask = df["Bid #"] == feature.bid_label
        numeric = numeric[mask]
    numeric = numeric.dropna()
    if numeric.empty:
        return None

    base_value = float(numeric.iloc[0])
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        span = abs(base_value) * 0.25 or 1.0
        min_value = base_value - span
        max_value = base_value + span
        if feature.is_integer:
            min_value = math.floor(min_value)
            max_value = math.ceil(max_value)
            if min_value == max_value:
                max_value = min_value + 1
        else:
            if math.isclose(min_value, max_value):
                max_value = min_value + 1.0

    if feature.is_integer:
        min_value = math.floor(min_value)
        max_value = math.ceil(max_value)
        if min_value < 0 and column in {"item_count"}:
            min_value = 0
        step = 1.0
        count = int(max_value - min_value) + 1
        count = max(min(count, 25), 5)
    else:
        span = max_value - min_value
        step = span / 30.0 if span > 0 else max(abs(base_value) * 0.05, 0.5)
        step = max(step, 0.01)
        count = int(span / step) + 1 if span > 0 else 15
        count = max(min(count, 60), 10)
    return ScenarioRange(min_value=min_value, max_value=max_value, step=step, count=count, base_value=base_value)


def _linspace_inclusive(start: float, stop: float, count: int, *, integer: bool) -> np.ndarray:
    count = max(count, 2)
    values = np.linspace(start, stop, num=count)
    if integer:
        values = np.round(values).astype(int)
        values = np.unique(values)
    return values


def build_adjustment_grid(
    df: pd.DataFrame,
    feature: ScenarioFeature,
    start: float,
    stop: float,
    count: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    values = _linspace_inclusive(start, stop, count, integer=feature.is_integer)
    frames: List[pd.DataFrame] = []

    for step_index, value in enumerate(values):
        scenario_df = df.copy(deep=True)
        if feature.kind == "time_to_departure":
            _apply_time_to_departure(scenario_df, float(value))
        elif feature.scope == "global":
            scenario_df[feature.key] = float(value)
        elif feature.scope == "bid" and feature.bid_label is not None and "Bid #" in scenario_df.columns:
            mask = scenario_df["Bid #"] == feature.bid_label
            scenario_df.loc[mask, feature.key] = float(value)
        else:
            scenario_df[feature.key] = float(value)
        scenario_df["scenario_feature_value"] = float(value)
        scenario_df["scenario_step"] = step_index
        frames.append(scenario_df)

    return pd.concat(frames, ignore_index=True)


def _apply_time_to_departure(df: pd.DataFrame, hours: float) -> None:
    if "departure_timestamp" not in df.columns:
        return
    departure = pd.to_datetime(df["departure_timestamp"], errors="coerce")
    if departure.isna().all():
        return
    offset = pd.to_timedelta(hours, unit="hour")
    current = departure - offset
    df["current_timestamp"] = current


def build_scenario_line_chart(df: pd.DataFrame, feature_label: str) -> go.Figure:
    fig = go.Figure()
    if df.empty or "Acceptance Probability" not in df.columns:
        fig.update_layout(
            template="plotly_white",
            title="Load a model to see acceptance probability curves",
            xaxis_title=feature_label,
            yaxis_title="Acceptance probability (%)",
        )
        return fig

    if "Bid #" not in df.columns and "bid_number" in df.columns:
        df = df.copy()
        df["Bid #"] = df["bid_number"]

    fig.update_layout(template="plotly_white")
    grouped = df.sort_values(["Bid #", "scenario_feature_value"]).groupby("Bid #")
    for idx, (bid_label, grp) in enumerate(grouped):
        fig.add_trace(
            go.Scatter(
                x=grp["scenario_feature_value"],
                y=grp["Acceptance Probability"],
                mode="lines+markers",
                name=f"Bid {bid_label}",
                line=dict(color=BAR_COLOR_SEQUENCE[idx % len(BAR_COLOR_SEQUENCE)]),
            )
        )

    fig.update_layout(
        title="Acceptance probability sensitivity",
        xaxis_title=feature_label,
        yaxis_title="Acceptance probability (%)",
        yaxis=dict(rangemode="tozero"),
        legend=dict(title="Bid", orientation="h", y=1.02, x=0, xanchor="left"),
        margin=dict(t=60, r=20, l=60, b=60),
        height=600,
    )
    return fig


def records_to_dataframe(records: Optional[Sequence[Dict[str, object]]]) -> pd.DataFrame:
    """Convert serialized scenario records back into a DataFrame.

    Dash stores JSON-serializable data inside ``dcc.Store`` components. When
    ``extract_baseline_snapshot`` exports records for the scenario tab it
    stringifies datetime columns so they can be stored.  Rehydrate those
    columns here so downstream helpers (e.g. prediction pipelines) see the
    expected dtypes again.
    """

    df = pd.DataFrame(records or [])
    if df.empty:
        return df

    for column in df.columns:
        if not isinstance(column, str):
            continue
        lowercase = column.lower()
        if "timestamp" in lowercase or "date" in lowercase:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().any():
                df[column] = parsed
    return df


__all__ = [
    "ScenarioFeature",
    "ScenarioRange",
    "build_adjustment_grid",
    "build_carrier_options",
    "build_feature_options",
    "build_flight_number_options",
    "build_scenario_line_chart",
    "build_travel_date_options",
    "build_upgrade_options",
    "compute_default_range",
    "extract_baseline_snapshot",
    "records_to_dataframe",
    "select_feature",
]
