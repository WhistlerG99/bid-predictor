"""Utilities for inferring feature roles from bid records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .constants import BID_IDENTIFIER_COLUMNS


_EXCLUDED_COLUMNS = {
    "Acceptance Probability",
    "Bid #",
    "scenario_feature_value",
    "scenario_step",
}


def _normalize_order(columns: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def _is_competitor_feature(name: str) -> bool:
    lowercase = name.lower()
    if "%" in name:
        return True
    if any(token in lowercase for token in ("percent", "quantile", "pct")):
        return True
    if any(lowercase.endswith(suffix) for suffix in ("_max", "_min", "_mean", "_median")):
        return True
    if lowercase.startswith(("num_", "count_", "competitor_")):
        return True
    return False


def _is_flight_feature(name: str) -> bool:
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


def _is_integer_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return False
    return bool(np.allclose(numeric, np.round(numeric)))


def _is_integer_feature_name(name: str) -> bool:
    lowercase = name.lower()
    if any(token in lowercase for token in ("count", "num", "inventory")):
        return True
    if lowercase.endswith("_id"):
        return True
    if lowercase in {"seats_available", "item_count"}:
        return True
    return False


@dataclass(frozen=True)
class FeatureRoles:
    """Represents inferred feature groupings for the UI."""

    bid_features: List[str]
    flight_features: List[str]
    competitor_features: List[str]
    display_features: List[str]
    numeric_features: Set[str]
    integer_features: Set[str]
    categorical_features: Set[str]

    @property
    def global_features(self) -> Set[str]:
        return set(self.flight_features) | set(self.competitor_features)


def infer_feature_roles(
    records: Optional[Sequence[Mapping[str, object]]]
) -> FeatureRoles:
    """Infer feature groupings from serialized bid records."""

    df = pd.DataFrame(list(records or []))
    if df.empty:
        return FeatureRoles([], [], [], [], set(), set(), set())

    candidate_columns = [
        column
        for column in df.columns
        if isinstance(column, str)
        and column not in _EXCLUDED_COLUMNS
        and column not in BID_IDENTIFIER_COLUMNS
    ]

    bid_features: List[str] = []
    flight_features: List[str] = []
    competitor_features: List[str] = []
    numeric_features: Set[str] = set()
    integer_features: Set[str] = set()
    categorical_features: Set[str] = set()

    for column in candidate_columns:
        series = df[column]
        non_null = series.dropna()
        unique_count = non_null.nunique(dropna=True)

        numeric = pd.to_numeric(non_null, errors="coerce")
        is_numeric = not numeric.isna().all()
        if is_numeric:
            numeric_features.add(column)
            if _is_integer_series(non_null) and _is_integer_feature_name(column):
                integer_features.add(column)
        else:
            categorical_features.add(column)

        if unique_count <= 1:
            if _is_competitor_feature(column):
                competitor_features.append(column)
            else:
                flight_features.append(column)
            continue

        if _is_competitor_feature(column):
            competitor_features.append(column)
            continue

        if _is_flight_feature(column) and column not in flight_features:
            flight_features.append(column)
        bid_features.append(column)

    display_seed = bid_features + flight_features + competitor_features
    if "offer_status" in df.columns:
        display_seed.append("offer_status")
    display_features = _normalize_order(display_seed)
    if "Acceptance Probability" not in display_features:
        display_features.append("Acceptance Probability")

    return FeatureRoles(
        bid_features=_normalize_order(bid_features),
        flight_features=_normalize_order(flight_features),
        competitor_features=_normalize_order(competitor_features),
        display_features=display_features,
        numeric_features=numeric_features,
        integer_features=integer_features,
        categorical_features=categorical_features,
    )


__all__ = ["FeatureRoles", "infer_feature_roles"]
