"""Helpers for adapting model feature configuration to the Dash UI."""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping, MutableMapping, Sequence


# Feature keys that should not surface as adjustable flight-level controls even if
# they appear in the model configuration.
_SNAPSHOT_CONTROL_EXCLUSIONS = {
    "carrier_code",
    "flight_code",
    "flight_number",
    "travel_date",
}

# Fallback feature lists that mirror the historical static UI behaviour.  These
# defaults are used whenever a loaded model does not expose a feature_config
# attribute so the UI can still render predictable controls.
_DEFAULT_ADJUSTABLE_BID_FEATURES = [
    "item_count",
    "usd_base_amount",
    "fare_class",
    "from_cabin",
    "offer_time",
    "multiplier_fare_class",
    "multiplier_loyalty",
    "multiplier_success_history",
    "multiplier_payment_type",
]

_DEFAULT_COMP_BID_FEATURES = [
    "usd_base_amount_25%",
    "usd_base_amount_50%",
    "usd_base_amount_75%",
    "usd_base_amount_max",
]

_DEFAULT_FLIGHT_FEATURES = [
    "seats_available",
    "num_offers",
    "days_before_departure",
]

_DEFAULT_DISPLAY_FEATURES = (
    _DEFAULT_ADJUSTABLE_BID_FEATURES
    + _DEFAULT_COMP_BID_FEATURES
    + ["offer_status", "Acceptance Probability"]
)

_DEFAULT_READONLY_FEATURES = _DEFAULT_COMP_BID_FEATURES + ["offer_status"]

DEFAULT_UI_FEATURE_CONFIG: MutableMapping[str, list[str]] = {
    "pre_features": _DEFAULT_ADJUSTABLE_BID_FEATURES + _DEFAULT_COMP_BID_FEATURES,
    "flight_features": _DEFAULT_FLIGHT_FEATURES,
    "snapshot_control_features": [
        feature
        for feature in _DEFAULT_FLIGHT_FEATURES
        if feature not in _SNAPSHOT_CONTROL_EXCLUSIONS
    ],
    "bid_features": _DEFAULT_ADJUSTABLE_BID_FEATURES,
    "comp_features": _DEFAULT_COMP_BID_FEATURES,
    "display_features": _DEFAULT_DISPLAY_FEATURES,
    "readonly_features": _DEFAULT_READONLY_FEATURES,
}


def _unique(sequence: Iterable[str]) -> list[str]:
    """Return ``sequence`` without duplicates while preserving the order.

    Feature definitions coming from CatBoost metadata are often lists with
    repeated entries (for example when categorical encodings expand a feature).
    The UI treats each feature as a distinct column, so duplicates would render
    redundant controls.  This helper normalises any iterable to an ordered list
    of unique string keys.
    """

    seen: set[str] = set()
    ordered: list[str] = []
    for item in sequence:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def build_ui_feature_config(
    raw_config: Mapping[str, object] | None,
) -> MutableMapping[str, list[str]]:
    """Return a UI-oriented view of the model feature configuration.

    Models expose their feature metadata in several shapes (lists, dictionaries
    and single strings).  The Dash UI expects a consistent mapping that splits
    features into categories such as ``bid_features`` or
    ``snapshot_control_features``.  This function standardises the structure by
    coercing each section to a unique, ordered list of strings and by applying
    sensible defaults when the model provides no configuration at all.  It also
    guarantees that UI-only fields (``offer_status`` and ``Acceptance
    Probability``) are present in the display list and flags competitor features
    as read-only so they cannot be edited from the table.
    """

    if raw_config is None:
        return deepcopy(DEFAULT_UI_FEATURE_CONFIG)

    def _coerce_list(key: str) -> list[str]:
        """Normalise ``raw_config[key]`` to a list of string feature names."""

        value = raw_config.get(key, [])
        if isinstance(value, Mapping):
            value = list(value.keys())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
        return [str(value)] if value else []

    pre_features = _unique(_coerce_list("pre_features"))
    flight_features = _unique(_coerce_list("flight_features"))
    bid_features = _unique(_coerce_list("bid_features"))
    comp_features = _unique(_coerce_list("comp_features"))

    snapshot_controls = [
        feature
        for feature in flight_features
        if feature not in _SNAPSHOT_CONTROL_EXCLUSIONS
    ]

    display_features = _unique(bid_features + comp_features)
    extras: list[str] = []
    if "offer_status" not in display_features:
        extras.append("offer_status")
    if "Acceptance Probability" not in display_features:
        extras.append("Acceptance Probability")
    display_features.extend(extras)

    readonly_features = _unique(comp_features + [feature for feature in extras if feature != "Acceptance Probability"])

    return {
        "pre_features": pre_features,
        "flight_features": flight_features,
        "snapshot_control_features": snapshot_controls,
        "bid_features": bid_features,
        "comp_features": comp_features,
        "display_features": display_features,
        "readonly_features": readonly_features,
    }


__all__ = [
    "DEFAULT_UI_FEATURE_CONFIG",
    "build_ui_feature_config",
]
