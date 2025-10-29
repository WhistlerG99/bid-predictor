"""Shared helpers for rendering and editing bid tables in the Dash UI."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .constants import USD_MAX_COLUMN, USD_PERCENT_COLUMNS
from .feature_roles import FeatureRoles, infer_feature_roles
from .formatting import normalize_offer_time, recompute_usd_metrics, safe_float


def _format_numeric(feature: str, value: object, roles: FeatureRoles) -> object:
    numeric = safe_float(value)
    if numeric is None:
        return value
    if feature in roles.integer_features:
        return int(round(numeric))
    if feature.startswith("multiplier"):
        return round(numeric, 4)
    lowered = feature.lower()
    if feature in USD_PERCENT_COLUMNS or feature == USD_MAX_COLUMN:
        return round(numeric, 2)
    if "amount" in lowered or "usd" in lowered:
        return round(numeric, 2)
    if "time" in lowered or lowered.endswith("_hours"):
        return round(numeric, 4)
    return numeric


def build_bid_table(
    records: Optional[Sequence[Dict[str, object]]],
    predictions: Optional[Dict[str, object]],
    *,
    locked_cells: Optional[Mapping[str, Sequence[str]]] = None,
    feature_roles: Optional[FeatureRoles] = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Return Dash DataTable configuration for bid feature editing."""

    if not records:
        columns = [{"name": "Feature", "id": "Feature", "editable": False}]
        return columns, [], []

    roles = feature_roles or infer_feature_roles(records)
    display_features = roles.display_features

    columns: List[Dict[str, object]] = [
        {"name": "Feature", "id": "Feature", "editable": False}
    ]
    data_rows: List[Dict[str, object]] = []
    style_rules: List[Dict[str, object]] = []

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
            if feature in roles.numeric_features:
                row[column_id] = _format_numeric(feature, value, roles)
            else:
                row[column_id] = value
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
    locked_cells: Optional[Mapping[str, Sequence[str]]] = None,
) -> Optional[List[Dict[str, object]]]:
    """Update bid records based on edited Dash DataTable values."""

    if not records or not table_data or not columns:
        return None

    updated_records = [dict(record) for record in records]
    feature_map = {row.get("Feature"): row for row in table_data}
    bid_columns = [column for column in columns if column.get("id") != "Feature"]

    roles = infer_feature_roles(records)
    display_features = roles.display_features

    locked_map: Dict[str, set[str]] = {}
    if locked_cells:
        for column_id, features in locked_cells.items():
            locked_map[column_id] = {str(feature) for feature in features}

    global_updates: Dict[str, object] = {}

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
            value_row = feature_map.get(feature)
            if value_row is None or column_id not in value_row:
                continue
            value = value_row[column_id]
            if feature == "offer_status":
                continue
            if feature in USD_PERCENT_COLUMNS:
                # recomputed from usd_base_amount after loop
                continue
            if feature == USD_MAX_COLUMN:
                continue

            if feature in roles.global_features:
                global_updates[feature] = value
                continue

            if feature in roles.numeric_features:
                record[feature] = _format_numeric(feature, value, roles)
            else:
                record[feature] = value
        normalize_offer_time(record)

    if global_updates:
        for feature, value in global_updates.items():
            for record in updated_records:
                if feature in roles.numeric_features:
                    record[feature] = _format_numeric(feature, value, roles)
                else:
                    record[feature] = value

    recompute_usd_metrics(updated_records)
    return updated_records


__all__ = ["apply_table_edits", "build_bid_table"]
