"""Shared helpers for rendering and editing bid tables in the Dash UI."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .constants import DISPLAY_FEATURE_ROWS, USD_MAX_COLUMN, USD_PERCENT_COLUMNS
from .formatting import normalize_offer_time, recompute_usd_metrics, safe_float


def build_bid_table(
    records: Optional[Sequence[Dict[str, object]]],
    predictions: Optional[Dict[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Return Dash DataTable configuration for bid feature editing."""

    if not records:
        columns = [{"name": "Feature", "id": "Feature", "editable": False}]
        return columns, [], []

    columns: List[Dict[str, object]] = [
        {"name": "Feature", "id": "Feature", "editable": False}
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
                numeric = safe_float(value)
                row[column_id] = int(numeric) if numeric is not None else value
            elif feature == "offer_time":
                numeric = safe_float(value)
                row[column_id] = round(numeric, 4) if numeric is not None else value
            elif feature == "usd_base_amount":
                numeric = safe_float(value)
                row[column_id] = round(numeric, 2) if numeric is not None else value
            elif feature in USD_PERCENT_COLUMNS:
                numeric = safe_float(value)
                row[column_id] = round(numeric, 2) if numeric is not None else value
            elif feature == USD_MAX_COLUMN:
                numeric = safe_float(value)
                row[column_id] = round(numeric, 2) if numeric is not None else value
            elif feature.startswith("multiplier"):
                numeric = safe_float(value)
                row[column_id] = round(numeric, 4) if numeric is not None else value
            else:
                numeric = safe_float(value)
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

    for percent_column in USD_PERCENT_COLUMNS:
        style_rules.append(
            {
                "if": {"filter_query": f'{{Feature}} = "{percent_column}"'},
                "backgroundColor": "#f8fafc",
                "pointerEvents": "none",
            }
        )

    style_rules.append(
        {
            "if": {"filter_query": f'{{Feature}} = "{USD_MAX_COLUMN}"'},
            "backgroundColor": "#f8fafc",
            "pointerEvents": "none",
        }
    )

    return columns, data_rows, style_rules


def apply_table_edits(
    records: Optional[Iterable[Dict[str, object]]],
    table_data: Optional[Sequence[Dict[str, object]]],
    columns: Optional[Sequence[Dict[str, object]]],
) -> Optional[List[Dict[str, object]]]:
    """Update bid records based on edited Dash DataTable values."""

    if not records or not table_data or not columns:
        return None

    updated_records = [dict(record) for record in records]
    feature_map = {row.get("Feature"): row for row in table_data}
    bid_columns = [column for column in columns if column.get("id") != "Feature"]

    for position, column in enumerate(bid_columns):
        column_id = column.get("id")
        if column_id is None or position >= len(updated_records):
            continue
        record = updated_records[position]
        for feature in DISPLAY_FEATURE_ROWS:
            if feature == "Acceptance Probability":
                continue
            value_row = feature_map.get(feature)
            if value_row is None or column_id not in value_row:
                continue
            value = value_row[column_id]
            if feature == "fare_class":
                record[feature] = value
            elif feature == "item_count":
                numeric = safe_float(value)
                record[feature] = int(numeric) if numeric is not None else value
            elif feature == "offer_time":
                numeric = safe_float(value)
                record[feature] = round(numeric, 4) if numeric is not None else value
            elif feature == "usd_base_amount":
                numeric = safe_float(value)
                record[feature] = numeric if numeric is not None else value
            elif feature in USD_PERCENT_COLUMNS:
                # recomputed from usd_base_amount after loop
                continue
            elif feature == USD_MAX_COLUMN:
                continue
            elif feature.startswith("multiplier"):
                numeric = safe_float(value)
                record[feature] = round(numeric, 4) if numeric is not None else value
            else:
                numeric = safe_float(value)
                record[feature] = numeric if numeric is not None else value
        normalize_offer_time(record)

    recompute_usd_metrics(updated_records)
    return updated_records


__all__ = ["apply_table_edits", "build_bid_table"]
