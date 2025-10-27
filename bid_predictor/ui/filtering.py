"""Dataset filtering helpers for the Dash UI."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .constants import _BID_IDENTIFIER_COLUMNS, _FLIGHT_GROUP_COLUMNS


def _normalize_threshold(value: Optional[object]) -> Optional[int]:
    """Convert user-specified numeric inputs into non-negative integers."""

    if value in (None, ""):
        return None
    try:
        normalized = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(normalized, 0)


def _resolve_bid_identifier_column(df: pd.DataFrame) -> Optional[str]:
    for column in _BID_IDENTIFIER_COLUMNS:
        if column in df.columns:
            return column
    return None


def _filter_dataset_by_combo_counts(
    dataset: pd.DataFrame,
    min_unique_bids: Optional[int],
    min_snapshot_count: Optional[int],
) -> pd.DataFrame:
    """Limit the dataset to flight/upgrade combos that satisfy count filters."""

    if dataset.empty:
        return dataset

    require_bids = bool(min_unique_bids) and min_unique_bids > 0
    require_snapshots = bool(min_snapshot_count) and min_snapshot_count > 0
    if not (require_bids or require_snapshots):
        return dataset

    missing_keys = [key for key in _FLIGHT_GROUP_COLUMNS if key not in dataset.columns]
    if missing_keys:
        return dataset

    working = dataset.copy()
    group_index = pd.MultiIndex.from_frame(working[_FLIGHT_GROUP_COLUMNS])

    summary = pd.DataFrame(index=group_index.unique())
    summary.index.names = _FLIGHT_GROUP_COLUMNS

    if require_bids:
        bid_column = _resolve_bid_identifier_column(working)
        if bid_column is None:
            return dataset
        summary["unique_bids"] = (
            working.groupby(_FLIGHT_GROUP_COLUMNS, dropna=False)[bid_column]
            .nunique(dropna=True)
            .reindex(summary.index, fill_value=0)
        )

    if require_snapshots:
        if "snapshot_num" not in working.columns:
            return dataset
        summary["snapshot_count"] = (
            working.groupby(_FLIGHT_GROUP_COLUMNS, dropna=False)["snapshot_num"]
            .nunique(dropna=True)
            .reindex(summary.index, fill_value=0)
        )

    if require_bids:
        summary = summary[summary["unique_bids"] >= int(min_unique_bids)]
    if require_snapshots:
        summary = summary[summary["snapshot_count"] >= int(min_snapshot_count)]

    if summary.empty:
        return working.iloc[0:0]

    mask = pd.MultiIndex.from_frame(working[_FLIGHT_GROUP_COLUMNS]).isin(summary.index)
    return working.loc[mask].copy()
