"""Shared helpers for rendering and editing bid tables in the Dash UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd

from ..data import load_model_cached
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


def _unique(sequence: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in sequence:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _looks_like_flight_feature(name: str) -> bool:
    lowercase = name.lower()
    if any(token in lowercase for token in ("seat", "inventory", "departure")):
        return True
    if lowercase.endswith("_hours") or lowercase.endswith("_days"):
        return True
    if "flight_" in lowercase:
        return True
    if "time_to_departure" in lowercase or "time_until" in lowercase:
        return True
    return False


def _collect_model_feature_names(model: object) -> List[str]:
    """Best-effort extraction of feature names from a fitted pipeline."""

    features: List[str] = []
    if model is None:
        return features

    steps = getattr(model, "steps", None)
    if not steps:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return list(names)
        return features

    def _normalize(names: Iterable[str]) -> List[str]:
        return [str(name) for name in names]

    for _, transformer in reversed(list(steps)):
        candidates: Optional[Iterable[str]] = None

        getter = getattr(transformer, "get_feature_names_out", None)
        if callable(getter):
            try:
                output = getter()
            except TypeError:
                # Some implementations expect an input array; fall back to
                # feature_names_in_ if available.
                output = getattr(transformer, "feature_names_in_", None)
            if output is not None:
                candidates = output

        if candidates is None:
            for attr in (
                "feature_names_out_",
                "feature_names_in_",
                "columns",
                "selected_features",
                "_existing_features",
            ):
                value = getattr(transformer, attr, None)
                if value:
                    candidates = value
                    break

        if candidates is not None:
            normalized = _normalize(candidates)
            if normalized:
                features = normalized
                break

        if not hasattr(transformer, "transform"):
            continue

    return features


def get_model_feature_names(model_uri: Optional[str]) -> List[str]:
    """Return the ordered feature names exposed by a fitted model."""

    if not model_uri:
        return []

    try:
        model = load_model_cached(model_uri)
    except Exception:
        return []

    return _unique(_collect_model_feature_names(model))


def _transform_records_with_model(
    model_uri: Optional[str],
    df: pd.DataFrame,
) -> tuple[Optional[pd.DataFrame], List[str]]:
    if not model_uri or df.empty:
        return None, []

    try:
        model = load_model_cached(model_uri)
    except Exception:
        return None, []

    model_features = _collect_model_feature_names(model)
    steps = getattr(model, "steps", None)
    if not steps:
        return None, model_features

    current: object = df.copy()
    try:
        for _, transformer in steps:
            if not hasattr(transformer, "transform"):
                break
            current = transformer.transform(current)
    except Exception:
        return None, model_features

    frame: Optional[pd.DataFrame]
    if isinstance(current, pd.DataFrame):
        frame = current.copy()
    else:
        frame = None
        if model_features:
            try:
                frame = pd.DataFrame(current, columns=model_features)
            except Exception:
                frame = None

    if frame is not None and model_features:
        frame = frame.reindex(columns=_unique(model_features))

    if frame is not None:
        frame = frame.reset_index(drop=True)

    return frame, model_features


@dataclass(frozen=True)
class _TableFeaturePlan:
    roles: FeatureRoles
    display_features: List[str]
    competitor_features: Set[str]
    model_features: List[str]
    feature_frame: Optional[pd.DataFrame]


def _prepare_feature_plan(
    records: Sequence[Dict[str, object]],
    model_uri: Optional[str],
    feature_roles: Optional[FeatureRoles] = None,
) -> Optional[_TableFeaturePlan]:
    if not records:
        return None

    df = pd.DataFrame(list(records))
    feature_frame, model_columns = _transform_records_with_model(model_uri, df)
    if not model_columns and feature_frame is not None:
        model_columns = list(feature_frame.columns)
    model_columns = _unique(model_columns)

    frame_records: List[Mapping[str, object]]
    if feature_frame is not None:
        frame_records = feature_frame.to_dict("records")
    else:
        frame_records = [{} for _ in records]

    enriched: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        merged = dict(record)
        supplemental = frame_records[idx] if idx < len(frame_records) else {}
        for key, value in supplemental.items():
            if key not in merged or merged[key] in (None, ""):
                merged[key] = value
        for feature in model_columns:
            merged.setdefault(feature, supplemental.get(feature))
        enriched.append(merged)

    roles = feature_roles or infer_feature_roles(enriched)

    model_feature_set = set(model_columns) if model_columns else None

    bid_features = [
        feature
        for feature in roles.bid_features
        if model_feature_set is None or feature in model_feature_set
    ]
    competitor_features = [
        feature
        for feature in roles.competitor_features
        if model_feature_set is None or feature in model_feature_set
    ]

    if model_feature_set is not None:
        supplemental_bid = [
            feature
            for feature in model_columns
            if feature not in bid_features
            and feature not in competitor_features
        ]
        if supplemental_bid:
            bid_features = _unique(list(bid_features) + supplemental_bid)

    if not bid_features:
        fallback_candidates = [
            feature
            for feature in roles.display_features
            if feature not in roles.competitor_features
            and feature not in {"offer_status", "Acceptance Probability"}
            and not _looks_like_flight_feature(str(feature))
        ]
        bid_features = _unique(fallback_candidates)

    if model_columns:
        ordered = [
            feature
            for feature in model_columns
            if feature in set(bid_features) | set(competitor_features)
        ]
    else:
        ordered = bid_features + competitor_features

    display_features = _unique(ordered)
    display_features = [feature for feature in display_features if feature != "num_offers"]

    return _TableFeaturePlan(
        roles=roles,
        display_features=display_features,
        competitor_features=set(competitor_features),
        model_features=model_columns,
        feature_frame=feature_frame,
    )


def build_bid_table(
    records: Optional[Sequence[Dict[str, object]]],
    predictions: Optional[Dict[str, object]],
    *,
    locked_cells: Optional[Mapping[str, Sequence[str]]] = None,
    feature_roles: Optional[FeatureRoles] = None,
    model_uri: Optional[str] = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    """Return Dash DataTable configuration for bid feature editing."""

    if not records:
        columns = [{"name": "Feature", "id": "Feature", "editable": False}]
        return columns, [], []

    plan = _prepare_feature_plan(records, model_uri, feature_roles)
    if plan is None:
        roles = feature_roles or infer_feature_roles(records)
        display_features: List[str] = []
        competitor_features: Set[str] = set()
        feature_rows: List[Dict[str, object]] = []
    else:
        roles = plan.roles
        display_features = list(plan.display_features)
        competitor_features = plan.competitor_features
        if plan.feature_frame is not None:
            feature_rows = plan.feature_frame.to_dict("records")
        else:
            feature_rows = []

    if "offer_status" in roles.display_features and "offer_status" not in display_features:
        display_features.append("offer_status")

    if "Acceptance Probability" not in display_features:
        display_features.append("Acceptance Probability")

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
            if feature in competitor_features:
                if idx < len(feature_rows):
                    value = feature_rows[idx].get(feature, value)
            elif value is None and idx < len(feature_rows):
                value = feature_rows[idx].get(feature, value)
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

    for feature in sorted(competitor_features):
        style_rules.append(
            {
                "if": {"filter_query": f'{{Feature}} = "{feature}"'},
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
    model_uri: Optional[str] = None,
    feature_roles: Optional[FeatureRoles] = None,
) -> Optional[List[Dict[str, object]]]:
    """Update bid records based on edited Dash DataTable values."""

    if not records or not table_data or not columns:
        return None

    plan = _prepare_feature_plan(records, model_uri, feature_roles)
    if plan is None:
        return None

    roles = plan.roles
    display_features = list(plan.display_features)
    if "offer_status" in roles.display_features and "offer_status" not in display_features:
        display_features.append("offer_status")

    updated_records = [dict(record) for record in records]
    feature_map = {row.get("Feature"): row for row in table_data}
    bid_columns = [column for column in columns if column.get("id") != "Feature"]

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

            if feature in roles.flight_features:
                global_updates[feature] = value
                continue

            if feature in plan.competitor_features:
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

    updated_plan = _prepare_feature_plan(updated_records, model_uri, roles)
    if updated_plan and updated_plan.feature_frame is not None:
        frame_rows = updated_plan.feature_frame.to_dict("records")
        for idx, record in enumerate(updated_records):
            if idx >= len(frame_rows):
                break
            row_values = frame_rows[idx]
            for feature in updated_plan.competitor_features:
                if feature not in row_values:
                    continue
                if feature in roles.numeric_features:
                    record[feature] = _format_numeric(
                        feature, row_values.get(feature), roles
                    )
                else:
                    record[feature] = row_values.get(feature)
    else:
        recompute_usd_metrics(updated_records)

    return updated_records


__all__ = ["apply_table_edits", "build_bid_table", "get_model_feature_names"]
