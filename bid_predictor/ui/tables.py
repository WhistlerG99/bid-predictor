"""Shared helpers for rendering and editing bid tables in the Dash UI."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .constants import USD_MAX_COLUMN, USD_PERCENT_COLUMNS
from .feature_config import DEFAULT_UI_FEATURE_CONFIG
from .formatting import normalize_offer_time, recompute_usd_metrics, safe_float


def build_bid_table(
    records: Optional[Sequence[Dict[str, object]]],
    predictions: Optional[Dict[str, object]],
    *,
    feature_config: Optional[Mapping[str, Sequence[str]]] = None,
    locked_cells: Optional[Mapping[str, Sequence[str]]] = None,
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

    config = feature_config or DEFAULT_UI_FEATURE_CONFIG
    display_features = list(config.get("display_features", []))
    if not display_features:
        display_features = list(DEFAULT_UI_FEATURE_CONFIG.get("display_features", []))
    if "Acceptance Probability" not in display_features:
        display_features.append("Acceptance Probability")

    editable_features = set(config.get("bid_features", []))
    readonly_features = set(config.get("readonly_features", []))
    readonly_features.update(config.get("comp_features", []))
    readonly_features.discard("Acceptance Probability")

    locked_map: Dict[str, set[str]] = {}
    if locked_cells:
        for column_id, features in locked_cells.items():
            locked_map[column_id] = {str(feature) for feature in features}

    for idx, record in enumerate(records):
        bid_label = record.get("Bid #") or record.get("bid_number") or idx + 1
        column_id = f"bid_{idx}"
        column_config: Dict[str, object] = {
            "name": f"Bid {bid_label}",
            "id": column_id,
            "editable": True,
        }
        columns.append(column_config)

    prediction_map = predictions or {}

    for feature in display_features:
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

    style_rules.append(
        {
            "if": {"filter_query": '{Feature} = "offer_status"'},
            "backgroundColor": "#f8fafc",
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

    for readonly_feature in readonly_features:
        if readonly_feature in {"Acceptance Probability"}:
            continue
        style_rules.append(
            {
                "if": {"filter_query": f'{{Feature}} = "{readonly_feature}"'},
                "backgroundColor": "#f8fafc",
                "pointerEvents": "none",
            }
        )

    for column_id, features in locked_map.items():
        for feature in features:
            style_rules.append(
                {
                    "if": {
                        "filter_query": f'{{Feature}} = "{feature}"',
                        "column_id": column_id,
                    },
                    "pointerEvents": "none",
                    "backgroundColor": "#f8fafc",
                    "color": "#94a3b8",
                }
            )

    return columns, data_rows, style_rules


def apply_table_edits(
    records: Optional[Iterable[Dict[str, object]]],
    table_data: Optional[Sequence[Dict[str, object]]],
    columns: Optional[Sequence[Dict[str, object]]],
    *,
    feature_config: Optional[Mapping[str, Sequence[str]]] = None,
    locked_cells: Optional[Mapping[str, Sequence[str]]] = None,
) -> Optional[List[Dict[str, object]]]:
    """Update bid records based on edited Dash DataTable values."""

    if not records or not table_data or not columns:
        return None

    updated_records = [dict(record) for record in records]
    feature_map = {row.get("Feature"): row for row in table_data}
    bid_columns = [column for column in columns if column.get("id") != "Feature"]

    locked_map: Dict[str, set[str]] = {}
    if locked_cells:
        for column_id, features in locked_cells.items():
            locked_map[column_id] = {str(feature) for feature in features}

    config = feature_config or DEFAULT_UI_FEATURE_CONFIG
    display_features = list(config.get("display_features", []))
    if not display_features:
        display_features = list(DEFAULT_UI_FEATURE_CONFIG.get("display_features", []))
    editable_features = set(config.get("bid_features", []))

    for position, column in enumerate(bid_columns):
        column_id = column.get("id")
        if column_id is None or position >= len(updated_records):
            continue
        record = updated_records[position]
        locked_features = locked_map.get(str(column_id), set())

        for feature in display_features:
            if feature == "Acceptance Probability":
                continue
            if feature in locked_features:
                continue
            if feature not in editable_features:
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
            elif feature == "offer_status":
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
