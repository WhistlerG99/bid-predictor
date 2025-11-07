import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from feature_engine.outliers import ArbitraryOutlierCapper
from feature_engine.discretisation import ArbitraryDiscretiser
from feature_engine.imputation import (
    ArbitraryNumberImputer,
    AddMissingIndicator,
    MeanMedianImputer,
)
from .feature_config import _GROUPBY_KEY_FEATURES


class AddMissingIndicatorCustom(AddMissingIndicator):

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Drop absent variables before delegating to the base implementation."""
        if self.variables:
            self.variables = [v for v in self.variables if v in X]

        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return transformed data when given pandas or ndarray inputs."""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class ArbitraryNumberImputerCustom(ArbitraryNumberImputer):

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Prune imputation entries for columns that are absent in ``X``."""
        if self.imputer_dict:
            self.imputer_dict = {k: v for k, v in self.imputer_dict.items() if k in X}

        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the imputation only when ``X`` is an array-like object."""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class MeanMedianImputerCustom(MeanMedianImputer):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Restrict the learned statistics to features present in ``X``."""
        if self.variables:
            self.variables = [v for v in self.variables if v in X]

        super().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the mean/median replacements when applicable."""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class ArbitraryOutlierCapperCustom(ArbitraryOutlierCapper):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Remove capping thresholds for columns that are not in the data."""
        if self.min_capping_dict:
            self.min_capping_dict = {
                k: v for k, v in self.min_capping_dict.items() if k in X
            }
            if not self.min_capping_dict:
                self.min_capping_dict = None
        if self.max_capping_dict:
            self.max_capping_dict = {
                k: v for k, v in self.max_capping_dict.items() if k in X
            }
            if not self.max_capping_dict:
                self.max_capping_dict = None

        if self.min_capping_dict or self.max_capping_dict:
            super().fit(X, y)
        else:
            self.variables_ = []
            self.right_tail_caps_ = {}
            self.left_tail_caps_ = {}
            self.feature_names_in_ = X.columns.to_list()
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier capping when provided with tabular inputs."""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            return super().transform(X)
        else:
            return X


class ArbitraryDiscretiserCustom(ArbitraryDiscretiser):
    def __init__(
        self,
        binning_dict: Optional[dict] = None,
        bin_names_dict: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the discretiser with optional renaming for bin labels."""
        super().__init__(binning_dict=binning_dict, **kwargs)
        self.bin_names_dict = bin_names_dict

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Retain only binning rules whose columns exist in ``X``."""
        if self.binning_dict:
            self.binning_dict = {k: v for k, v in self.binning_dict.items() if k in X}
            if not self.binning_dict:
                self.binning_dict = None

        if self.binning_dict:
            super().fit(X, y)
        else:
            self.variables_ = []
            self.feature_names_in_ = X.columns.to_list()
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Discretize the configured columns and optionally rename the bins."""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            X_tf = super().transform(X)
            if self.bin_names_dict:
                X_tf = X_tf.replace(self.bin_names_dict)
            return X_tf
        else:
            return X


def add_flight_code(data):
    """Create a combined carrier/flight categorical code when possible."""
    required = {"carrier_code", "flight_number"}
    if not required.issubset(data.columns):
        return data
    result = data.copy()
    result.loc[:, "flight_code"] = (
        result["carrier_code"].astype(str) + result["flight_number"].astype(str)
    ).astype("category")
    return result


def add_days_b4_depart(data):
    """Compute days between the current timestamp and departure."""
    required = {"departure_timestamp", "current_timestamp"}
    if not required.issubset(data.columns):
        return data
    result = data.copy()
    result.loc[:, "days_before_departure"] = (
        result.departure_timestamp - result.current_timestamp
    ).apply(lambda y: y.total_seconds()) / (60 * 60 * 24)
    return result


def add_bid_rank(data):
    """Create a combined carrier/flight categorical code when possible."""
    required = {
        "usd_base_amount",
        "multiplier_fare_class",
        "multiplier_loyalty",
        "multiplier_success_history",
        "multiplier_payment_type",
    }
    if not required.issubset(data.columns):
        return data
    result = data.copy().set_index(_GROUPBY_KEY_FEATURES)

    result["bid_rank"] = (
        result[list(required)]
        .prod(axis=1)
        .groupby(level=_GROUPBY_KEY_FEATURES, observed=False)
        .rank(method="dense", ascending=False)
        .astype(int)
        .astype("category")
    )
    return result.reset_index()[data.columns.tolist() + ["bid_rank"]]


def add_group_features(data):
    """Add group-level aggregate features such as offer counts and max price."""
    if "usd_base_amount" not in data.columns:
        return data
    if any(key not in data.columns for key in _GROUPBY_KEY_FEATURES):
        return data
    num_offers_col_name = "num_offers"
    usd_base_amount_max_name = "usd_base_amount_max"
    data = data.drop(
        columns=[num_offers_col_name, usd_base_amount_max_name], errors="ignore"
    )
    data = data.merge(
        data.groupby(
            _GROUPBY_KEY_FEATURES,
            observed=True,
        )
        .size()
        .rename(num_offers_col_name)
        .reset_index(),
        on=_GROUPBY_KEY_FEATURES,
    )
    data = data.merge(
        data.groupby(
            _GROUPBY_KEY_FEATURES,
            observed=True,
        )["usd_base_amount"]
        .max()
        .rename(usd_base_amount_max_name)
        .reset_index(),
        on=_GROUPBY_KEY_FEATURES,
    )
    return data


def quantiles_vectorized(vals, qs=(0.25, 0.50, 0.75)):
    """Vectorized helper that returns leave-one-out quantiles for each value."""
    vals = np.asarray(vals)
    n = vals.size
    data = np.full((len(qs), n), np.nan, dtype=float)
    if n <= 1:
        return data

    # Stable sort to keep deterministic ranks with ties
    order = np.argsort(vals, kind="mergesort")
    S = vals[order]  # sorted values
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)  # rank of each original element

    m = n - 1
    for qi, q in enumerate(qs):
        h = q * (m - 1)  # NumPy's 'linear' method position
        j = int(np.floor(h))
        gamma = h - j
        j2 = min(j + 1, m - 1)  # clamp upper neighbor inside [0, m-1]

        # Map indices from T (length m) back to S (length n), accounting for the removed item
        idx_a = j + (j >= ranks)
        idx_b = j2 + (j2 >= ranks)

        data[qi, :] = (1.0 - gamma) * S[idx_a] + gamma * S[idx_b]

    return data  # shape (len(qs), n)


def quantiles_group(group, col):
    """Compute leave-one-out quantiles for a grouped pandas Series."""
    arr = group[col].to_numpy()
    q25, q50, q75 = quantiles_vectorized(arr, (0.25, 0.50, 0.75))
    return pd.DataFrame(
        {
            "q25_excl_self": q25,
            "median_excl_self": q50,
            "q75_excl_self": q75,
        },
        index=group.index,
    )


def add_quantiles(data):
    """Attach quantile-based summary statistics per flight group."""
    if "usd_base_amount" not in data.columns:
        return data
    if any(key not in data.columns for key in _GROUPBY_KEY_FEATURES):
        return data
    data[["usd_base_amount_25%", "usd_base_amount_50%", "usd_base_amount_75%"]] = (
        data.groupby(
            _GROUPBY_KEY_FEATURES,
            group_keys=False,
            observed=True,
            sort=False,
        ).apply(quantiles_group, col="usd_base_amount", include_groups=False)
    )
    return data


def _lossless_numeric(series: pd.Series):
    """Attempt to coerce a series to numeric without introducing NaNs."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_na_mask = series.notna()
    if not non_na_mask.any():
        return numeric, True
    return (numeric, not numeric[non_na_mask].isna().any())


def _align_merge_key_types(
    active_lookup: pd.DataFrame, X_inactive: pd.DataFrame, key: str
):
    """Ensure merge keys share a dtype between active and inactive slices."""
    if key not in active_lookup.columns or key not in X_inactive.columns:
        return

    active_numeric, active_ok = _lossless_numeric(active_lookup[key])
    inactive_numeric, inactive_ok = _lossless_numeric(X_inactive[key])

    if active_ok and inactive_ok:
        active_lookup[key] = active_numeric
        X_inactive[key] = inactive_numeric
    else:
        active_lookup[key] = active_lookup[key].astype("string")
        X_inactive[key] = X_inactive[key].astype("string")


def eval_transforms(X, func, cols):
    """Fills the inactive offers <cols> with the <func> using the values at the time it was a last active bid"""
    X_active = X[X.active == True].copy()  # noqa: E712 - intentional identity comparison
    X_inactive = X[X.active == False].copy()  # noqa: E712 - intentional identity comparison

    X_active = func(X_active)

    lookup_cols = ["id"] + _GROUPBY_KEY_FEATURES + cols
    existing_cols = [column for column in lookup_cols if column in X_active.columns]
    active_lookup = (
        X_active[existing_cols]
        .rename(columns={"snapshot_num": "last_snapshot"})
        .copy()
    )

    _align_merge_key_types(active_lookup, X_inactive, "last_snapshot")

    merge_keys = ["id"] + _GROUPBY_KEY_FEATURES[:-1] + ["last_snapshot"]
    available_keys = [key for key in merge_keys if key in active_lookup.columns]
    if available_keys:
        X_inactive = X_inactive.merge(
            active_lookup,
            on=available_keys,
            how="left",
        )

    X = pd.concat((X_active, X_inactive)).sort_values(
        _GROUPBY_KEY_FEATURES + ["active", "id"]
    )
    return X


# --- Wrappers around your existing funcs so they can be used in pipelines ---
def add_flight_code_wrapper(X):
    """Safely apply :func:`add_flight_code` within sklearn pipelines."""
    if isinstance(X, pd.DataFrame):
        return add_flight_code(X)
    else:
        return X


def add_days_b4_depart_wrapper(X):
    """Safely apply :func:`add_days_b4_depart` within sklearn pipelines."""
    if isinstance(X, pd.DataFrame):
        return add_days_b4_depart(X)
    else:
        return X


# --- Wrappers around your existing funcs so they can be used in pipelines ---
def add_bid_rank_wrapper(X):
    """Safely apply :func:`add_flight_code` within sklearn pipelines."""
    if isinstance(X, pd.DataFrame):
        X = eval_transforms(X, add_bid_rank, ["bid_rank"])
    return X


def add_group_features_wrapper(X):
    """Safely apply :func:`add_group_features` within sklearn pipelines."""
    if isinstance(X, pd.DataFrame):
        X = eval_transforms(X, add_group_features, ["num_offers", "usd_base_amount_max"])
    return X


def add_quantiles_wrapper(X):
    """Safely apply :func:`add_quantiles` within sklearn pipelines."""
    if isinstance(X, pd.DataFrame):
        X = eval_transforms(X, add_quantiles, ["usd_base_amount_25%", "usd_base_amount_50%", "usd_base_amount_75%"])
    return X


class ColumnReducer(BaseEstimator, TransformerMixin):
    """Selects the configured columns from a pandas DataFrame."""

    def __init__(self, selected_features):
        # Store the parameter without modification so the estimator remains
        # cloneable by scikit-learn utilities such as ``clone`` or
        # ``cross_validate``. Any derived lists are created lazily in ``fit``.
        self.selected_features = selected_features
        self._existing_features = None

    def fit(self, X, y=None):
        selected_features = list(self.selected_features)

        if isinstance(X, pd.DataFrame):
            self._existing_features = [
                feature for feature in selected_features if feature in X.columns
            ]
            self._existing_features += [
                feature + "_na"
                for feature in selected_features
                if feature + "_na" in X.columns
            ]
        else:
            self._existing_features = selected_features
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        features = self._existing_features
        if features is None:
            features = [
                feature
                for feature in list(self.selected_features)
                if feature in X.columns
            ]
        else:
            features = [feature for feature in features if feature in X.columns]

        return X.reindex(columns=features)


# FunctionTransformer allows arbitrary pandas-based functions
add_flight_code_transformer = FunctionTransformer(add_flight_code_wrapper)
add_days_b4_depart_transformer = FunctionTransformer(add_days_b4_depart_wrapper)
add_bid_rank_transformer = FunctionTransformer(add_bid_rank_wrapper)
group_features_transformer = FunctionTransformer(add_group_features_wrapper)
quantiles_transformer = FunctionTransformer(add_quantiles_wrapper)
