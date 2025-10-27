"""Constants used by the Dash UI helpers."""
from __future__ import annotations

from plotly import colors as plotly_colors

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
    "Acceptance Probability",
]

_USD_PERCENT_COLUMNS = {
    "usd_base_amount_25%": 0.25,
    "usd_base_amount_50%": 0.50,
    "usd_base_amount_75%": 0.75,
}

_USD_MAX_COLUMN = "usd_base_amount_max"

_BID_IDENTIFIER_COLUMNS = ("id", "bid_id", "bid_number")

_FLIGHT_GROUP_COLUMNS = [
    "carrier_code",
    "flight_number",
    "travel_date",
    "upgrade_type",
]

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
