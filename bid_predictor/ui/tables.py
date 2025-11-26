"""Shared helpers for rendering and editing bid tables in the Dash UI."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .feature_config import DEFAULT_UI_FEATURE_CONFIG
from .formatting import clear_derived_features, normalize_offer_time, safe_float


def build_bid_table(
    records: Optional[Sequence[Dict[str, object]]],
    predictions: Optional[Dict[str, object]],
    *,
    feature_config: Optional[Mapping[str, Sequence[str]]] = None,
    locked_cells: Optional[Mapping[str, Sequence[str]]] = None,
    derived_feature_values: Optional[Sequence[Mapping[str, object]]] = None,
    show_comp_features: bool = True,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Return Dash DataTable configuration for bid feature editing.

    The Dash UI represents bids as a transposed table where each column is a
    bid and each row is a feature.  This helper converts the list of bid
    records into the three structures ``dash_table.DataTable`` expects: column
    definitions, cell data, and per-cell styling rules.  It respects the model's
    feature configuration, locks read-only features, injects prediction values
    and applies rounding so the displayed values match the formatting logic used
    elsewhere in the UI.

    Parameters
    ----------
    records:
        Original bid dictionaries in the order they should appear in the table.
    predictions:
        Mapping from generated column ids (``bid_<index>``) to acceptance
        probability predictions.
    feature_config:
        UI-specific feature configuration which controls editable and read-only
        features.  When absent the ``DEFAULT_UI_FEATURE_CONFIG`` fallback is
        used.
    locked_cells:
        Optional mapping of column ids to feature names that the caller wants to
        lock in the UI (e.g. while a scenario slider is active).
    derived_feature_values:
        Optional sequence mirroring ``records`` where each entry contains the
        derived features produced by the model's preprocessing pipeline.
    show_comp_features:
        When ``False``, columns listed under ``comp_features`` in the feature
        configuration are hidden from the rendered table.

    Returns
    -------
    tuple
        A ``(columns, data_rows, style_rules)`` tuple ready to be fed into a
        Dash DataTable.
    """

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
    comp_features = list(config.get("comp_features", []))
    if not show_comp_features and comp_features:
        display_features = [
            feature for feature in display_features if feature not in comp_features
        ]
    if "Acceptance Probability" not in display_features:
        display_features.append("Acceptance Probability")

    editable_features = set(config.get("bid_features", []))
    readonly_features = set(config.get("readonly_features", []))
    if show_comp_features:
        readonly_features.update(comp_features)
    else:
        readonly_features.difference_update(comp_features)
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
    derived_lookup = list(derived_feature_values or [])

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
            if feature == "Current Time":
                timestamp_value = record.get("Current Time")
                if timestamp_value is None:
                    timestamp_value = record.get("current_timestamp") or record.get(
                        "accept_prob_timestamp"
                    )
                row[column_id] = timestamp_value
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
            elif feature == "days_before_departure":
                numeric = safe_float(value)
                row[column_id] = round(numeric, 4) if numeric is not None else value
            elif feature == "hours_before_departure":
                numeric = safe_float(value)
                row[column_id] = round(numeric, 4) if numeric is not None else value
            elif feature == "usd_base_amount":
                numeric = safe_float(value)
                row[column_id] = round(numeric, 2) if numeric is not None else value
            elif feature in comp_features:
                derived_value = None
                if idx < len(derived_lookup):
                    derived_value = derived_lookup[idx].get(feature)
                if derived_value is None:
                    derived_value = value
                numeric = safe_float(derived_value)
                row[column_id] = (
                    round(numeric, 2) if numeric is not None else derived_value
                )
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
    """Update bid records based on edited Dash DataTable values.

    Dash sends back edited data as rows keyed by ``Feature`` with each bid
    stored under a ``bid_<index>`` column.  This helper rehydrates the list of
    bid dictionaries by walking the edited cells, respecting the feature
    configuration (only ``bid_features`` are mutable) and skipping cells that
    are explicitly locked.  Numeric values are coerced and rounded using the
    same rules as :func:`build_bid_table` to avoid drift between consecutive
    edits.  Derived comparison columns are cleared after the update so their
    values can be repopulated from the model pipeline.

    Returns
    -------
    Optional[List[Dict[str, object]]]
        ``None`` when the payload is incomplete, otherwise a new list of bid
        records mirroring ``records`` with the edits applied.
    """

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
    comp_features = set(config.get("comp_features", []))

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
            elif feature in comp_features:
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

    clear_derived_features(updated_records, config)
    return updated_records


__all__ = ["apply_table_edits", "build_bid_table"]
