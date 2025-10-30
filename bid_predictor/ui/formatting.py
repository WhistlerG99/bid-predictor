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
    """Convert ``value`` to a floating point number when possible.

    The Dash UI frequently receives mixed types from editable table cells –
    numbers come back as strings, empty cells as ``""`` or ``None`` and
    occasionally complex objects if a component stores metadata.  This helper
    normalises those cases so downstream formatting code can operate on a
    consistent ``float`` representation.  Any value that cannot be interpreted
    as a finite float results in ``None`` so callers can decide whether to keep
    the original input or drop the field altogether.

    Parameters
    ----------
    value:
        The raw value captured from the UI or existing record.

    Returns
    -------
    Optional[float]
        The parsed floating point value, or ``None`` if the input is empty or
        not numeric.
    """

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
    """Round the ``offer_time`` field in-place to keep UI output consistent.

    ``offer_time`` is stored as a fractional number of days in model inputs and
    is expected to be displayed with four decimal places.  The Dash table
    allows arbitrary precision, so this helper ensures the backing record keeps
    a uniform representation that matches the formatting rules applied when the
    data is rendered.
    """

    offer_value = safe_float(record.get("offer_time"))
    if offer_value is None:
        return
    record["offer_time"] = round(offer_value, 4)


def prepare_bid_record(record: Dict[str, object]) -> Dict[str, object]:
    """Return a shallow copy of ``record`` prepared for display.

    The Dash table expects numeric values to be rounded to user-friendly
    precision and should not render the ``Acceptance Probability`` column
    alongside editable bid features.  This helper removes the prediction column
    and normalises the values that require rounding while leaving the source
    dictionary untouched.
    """

    prepared = dict(record)
    prepared.pop("Acceptance Probability", None)
    normalize_offer_time(prepared)
    amount_value = safe_float(prepared.get("usd_base_amount"))
    if amount_value is not None:
        prepared["usd_base_amount"] = round(amount_value, 2)
    return prepared


def recompute_usd_metrics(records: List[Dict[str, object]]) -> None:
    """Update comparison columns derived from ``usd_base_amount``.

    When the base amount of a bid changes, the percent-based comparison columns
    (e.g. ``usd_base_amount_25%``) and the maximum value must be recomputed for
    each bid so that the table reflects the latest context.  This function
    performs the in-place update by first normalising the base amounts, then
    calculating peer-based quantiles for every record.  Records with missing
    values propagate ``None`` to the derived fields which signals the UI to
    leave the cells blank.
    """

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
    """Build a mapping from bid identifier to the label index.

    The UI shows bids in ascending order using an integer label.  Depending on
    the dataset, the identifier can live in one of several columns, so this
    helper scans the known candidates and picks the first column that contains
    data.  The returned dictionary maps the original identifier to its
    positional label, while the second element communicates which column was
    used.  Callers can therefore reuse the mapping when updating or sorting
    records.
    """

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
    """Ensure the ``Bid #`` column reflects the provided identifier mapping.

    The Dash table expects a consecutive label column regardless of how the
    underlying dataset stores bid numbers.  When a ``label_map`` is provided,
    the function remaps the identifiers to the friendly labels, falling back to
    the existing values when a mapping cannot be resolved.  If no mapping is
    available, the helper synthesises sequential labels so the UI still has a
    stable ordering.
    """

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
    """Return records ordered by their bid label.

    The Dash callbacks frequently operate on plain dictionaries.  This helper
    wraps ``sorted`` with a defensive key function that gracefully handles
    missing labels, ``NaN`` values and strings that contain numeric text.  The
    result is a list ordered in the same way the UI renders the bids.
    """

    def sort_key(record: Dict[str, object]) -> Tuple[int, object]:
        """Build a two-part sort key prioritising labelled records."""

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
    """Return the next available bid label given the existing records.

    When the user inserts a new bid row the UI needs the next free integer
    label to keep the ordering consistent.  The function inspects every record,
    extracts any numeric representation of the ``Bid #`` value and returns the
    next integer after the observed maximum.  If no numeric labels exist yet,
    the counter starts at ``1``.
    """

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
