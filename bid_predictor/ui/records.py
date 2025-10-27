"""Record manipulation helpers for the Dash UI."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .constants import _USD_MAX_COLUMN, _USD_PERCENT_COLUMNS


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


def _prepare_bid_record(record: Dict[str, object]) -> Dict[str, object]:
    prepared = dict(record)
    prepared.pop("Acceptance Probability", None)
    _normalize_offer_time(prepared)
    amount_value = _safe_float(prepared.get("usd_base_amount"))
    if amount_value is not None:
        prepared["usd_base_amount"] = round(amount_value, 2)
    return prepared


def _recompute_usd_metrics(records: List[Dict[str, object]]) -> None:
    if not records:
        return

    amounts: List[Optional[float]] = []
    for record in records:
        amount = _safe_float(record.get("usd_base_amount"))
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
            for column, fraction in _USD_PERCENT_COLUMNS.items():
                quantile_value = peer_series.quantile(fraction)
                record[column] = (
                    round(float(quantile_value), 2)
                    if quantile_value is not None and not pd.isna(quantile_value)
                    else None
                )
        else:
            for column in _USD_PERCENT_COLUMNS:
                record[column] = None

        record[_USD_MAX_COLUMN] = (
            round(float(max_amount), 2)
            if max_amount is not None and not pd.isna(max_amount)
            else None
        )
