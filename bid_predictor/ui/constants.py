"""Constants used across the Dash UI helpers."""
from __future__ import annotations

from typing import Tuple

USD_PERCENT_COLUMNS = {
    "usd_base_amount_25%": 0.25,
    "usd_base_amount_50%": 0.50,
    "usd_base_amount_75%": 0.75,
}

USD_MAX_COLUMN = "usd_base_amount_max"

BID_IDENTIFIER_COLUMNS: Tuple[str, ...] = ("id", "bid_id", "bid_number")

__all__ = [
    "BID_IDENTIFIER_COLUMNS",
    "USD_MAX_COLUMN",
    "USD_PERCENT_COLUMNS",
]
