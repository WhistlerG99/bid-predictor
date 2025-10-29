"""Formatting helpers for bid records displayed in the Dash UI."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .constants import (
    BID_IDENTIFIER_COLUMNS,
    USD_MAX_COLUMN,
    USD_PERCENT_COLUMNS,
)


def safe_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def normalize_offer_time(record: Dict[str, object]) -> None:
    offer_value = safe_float(record.get("offer_time"))
    if offer_value is None:
        return
    record["offer_time"] = round(offer_value, 4)


def prepare_bid_record(record: Dict[str, object]) -> Dict[str, object]:
    prepared = dict(record)
    prepared.pop("Acceptance Probability", None)
    normalize_offer_time(prepared)
    amount_value = safe_float(prepared.get("usd_base_amount"))
    if amount_value is not None:
        prepared["usd_base_amount"] = round(amount_value, 2)
    return prepared


def recompute_usd_metrics(records: List[Dict[str, object]]) -> None:
    if not records:
        return

    amounts: List[Optional[float]] = []
    for record in records:
        amount = safe_float(record.get("usd_base_amount"))
        if amount is not None:
            record["usd_base_amount"] = round(amount, 2)
        amounts.append(amount)

    valid_amounts = [value for value in amounts if value is not None]
    max_amount: Optional[float] = max(valid_amounts) if valid_amounts else None

    for idx, record in enumerate(records):
        peer_values = [
            value
            for peer_idx, value in enumerate(amounts)
            if peer_idx != idx and value is not None
        ]
        if peer_values:
            peer_series = pd.Series(peer_values)
            for column, fraction in USD_PERCENT_COLUMNS.items():
                quantile_value = peer_series.quantile(fraction)
                record[column] = (
                    round(float(quantile_value), 2)
                    if quantile_value is not None and not pd.isna(quantile_value)
                    else None
                )
        else:
            for column in USD_PERCENT_COLUMNS:
                record[column] = None

        record[USD_MAX_COLUMN] = (
            round(float(max_amount), 2)
            if max_amount is not None and not pd.isna(max_amount)
            else None
        )


def compute_bid_label_map(df: pd.DataFrame) -> Tuple[Dict[object, int], Optional[str]]:
    """Build a mapping from bid identifier to the label index."""

    for column in BID_IDENTIFIER_COLUMNS:
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


def apply_bid_labels(
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


def sort_records_by_bid(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
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


def get_next_bid_label(records: Iterable[Dict[str, object]]) -> int:
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


__all__ = [
    "apply_bid_labels",
    "compute_bid_label_map",
    "get_next_bid_label",
    "normalize_offer_time",
    "prepare_bid_record",
    "recompute_usd_metrics",
    "safe_float",
    "sort_records_by_bid",
]
