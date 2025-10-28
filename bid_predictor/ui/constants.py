"""Constants used across the Dash UI helpers."""
from __future__ import annotations

from typing import Tuple

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
    "usd_base_amount_max",
    "offer_status",
    "Acceptance Probability",
]

USD_PERCENT_COLUMNS = {
    "usd_base_amount_25%": 0.25,
    "usd_base_amount_50%": 0.50,
    "usd_base_amount_75%": 0.75,
}

USD_MAX_COLUMN = "usd_base_amount_max"

BID_IDENTIFIER_COLUMNS: Tuple[str, ...] = ("id", "bid_id", "bid_number")

__all__ = [
    "BID_IDENTIFIER_COLUMNS",
    "DISPLAY_FEATURE_ROWS",
    "USD_MAX_COLUMN",
    "USD_PERCENT_COLUMNS",
]
