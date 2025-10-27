"""Helpers for bid labeling and ordering in the Dash UI."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .constants import _BID_IDENTIFIER_COLUMNS


def _compute_bid_label_map(df: pd.DataFrame) -> Tuple[Dict[object, int], Optional[str]]:
    """Build a mapping from bid identifier to the label index."""

    for column in _BID_IDENTIFIER_COLUMNS:
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
