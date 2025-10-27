"""Helper utilities for the Dash-based user interface."""

from .constants import (
    DISPLAY_FEATURE_ROWS,
    _BAR_COLOR_SEQUENCE,
    _BID_IDENTIFIER_COLUMNS,
    _FLIGHT_GROUP_COLUMNS,
    _USD_MAX_COLUMN,
    _USD_PERCENT_COLUMNS,
)
from .data_loading import _get_feature_columns, _load_dataset_cached, _load_model_cached
from .filtering import _filter_dataset_by_combo_counts, _normalize_threshold, _resolve_bid_identifier_column
from .labeling import _apply_bid_labels, _compute_bid_label_map, _get_next_bid_label, _sort_records_by_bid
from .plotting import _build_prediction_plot
from .prediction import _predict, _prepare_prediction_dataframe
from .records import _normalize_offer_time, _prepare_bid_record, _recompute_usd_metrics, _safe_float

__all__ = [
    "DISPLAY_FEATURE_ROWS",
    "_BAR_COLOR_SEQUENCE",
    "_BID_IDENTIFIER_COLUMNS",
    "_FLIGHT_GROUP_COLUMNS",
    "_USD_MAX_COLUMN",
    "_USD_PERCENT_COLUMNS",
    "_get_feature_columns",
    "_load_dataset_cached",
    "_load_model_cached",
    "_filter_dataset_by_combo_counts",
    "_normalize_threshold",
    "_resolve_bid_identifier_column",
    "_apply_bid_labels",
    "_compute_bid_label_map",
    "_get_next_bid_label",
    "_sort_records_by_bid",
    "_build_prediction_plot",
    "_predict",
    "_prepare_prediction_dataframe",
    "_normalize_offer_time",
    "_prepare_bid_record",
    "_recompute_usd_metrics",
    "_safe_float",
]
