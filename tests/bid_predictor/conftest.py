import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn import set_config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_feature_engine_stub():
    if "feature_engine" in sys.modules:
        return

    feature_engine = types.ModuleType("feature_engine")
    imputation = types.ModuleType("feature_engine.imputation")
    outliers = types.ModuleType("feature_engine.outliers")
    discretisation = types.ModuleType("feature_engine.discretisation")

    class AddMissingIndicator:
        def __init__(self, variables=None):
            self.variables = variables
            self.variables_ = None

        def fit(self, X, y=None):
            if self.variables is None:
                self.variables_ = list(X.columns)
            else:
                self.variables_ = [var for var in self.variables if var in X]
            return self

        def transform(self, X):
            data = X.copy()
            for var in self.variables_ or []:
                data[f"{var}_na"] = data[var].isna().astype(int)
            return data

    class ArbitraryNumberImputer:
        def __init__(self, imputer_dict=None):
            self.imputer_dict = imputer_dict or {}

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            data = X.copy()
            for var, value in self.imputer_dict.items():
                if var in data:
                    data[var] = data[var].fillna(value)
            return data

    class MeanMedianImputer:
        def __init__(self, variables=None, imputation_method="median"):
            self.variables = variables or []
            self.imputation_method = imputation_method
            self.statistics_ = {}

        def fit(self, X, y=None):
            for var in self.variables:
                if var in X:
                    series = X[var]
                    if self.imputation_method == "mean":
                        self.statistics_[var] = series.mean()
                    else:
                        self.statistics_[var] = series.median()
            return self

        def transform(self, X):
            data = X.copy()
            for var, value in self.statistics_.items():
                data[var] = data[var].fillna(value)
            return data

    class ArbitraryOutlierCapper:
        def __init__(self, min_capping_dict=None, max_capping_dict=None):
            self.min_capping_dict = min_capping_dict or {}
            self.max_capping_dict = max_capping_dict or {}

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            data = X.copy()
            if self.min_capping_dict:
                for var, min_val in self.min_capping_dict.items():
                    if var in data:
                        data[var] = data[var].clip(lower=min_val)
            if self.max_capping_dict:
                for var, max_val in self.max_capping_dict.items():
                    if var in data:
                        data[var] = data[var].clip(upper=max_val)
            return data

    class ArbitraryDiscretiser:
        def __init__(self, binning_dict=None, return_boundaries=False):
            self.binning_dict = binning_dict or {}
            self.return_boundaries = return_boundaries

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            data = X.copy()
            for var, bins in self.binning_dict.items():
                if var in data:
                    data[var] = pd.cut(data[var], bins=bins, include_lowest=True)
            return data

    imputation.AddMissingIndicator = AddMissingIndicator
    imputation.ArbitraryNumberImputer = ArbitraryNumberImputer
    imputation.MeanMedianImputer = MeanMedianImputer
    outliers.ArbitraryOutlierCapper = ArbitraryOutlierCapper
    discretisation.ArbitraryDiscretiser = ArbitraryDiscretiser

    feature_engine.imputation = imputation
    feature_engine.outliers = outliers
    feature_engine.discretisation = discretisation

    sys.modules["feature_engine"] = feature_engine
    sys.modules["feature_engine.imputation"] = imputation
    sys.modules["feature_engine.outliers"] = outliers
    sys.modules["feature_engine.discretisation"] = discretisation


_install_feature_engine_stub()

set_config(enable_metadata_routing=True)

from bid_predictor.feature_config import _GROUPBY_KEY_FEATURES


@dataclass
class MlflowCalls:
    params: List[Dict[str, object]] = field(default_factory=list)
    metrics: List[Dict[str, float]] = field(default_factory=list)
    metric_calls: List[Dict[str, object]] = field(default_factory=list)
    dicts: List[Dict[str, object]] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    tags: List[Dict[str, str]] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)


class DummyRun:
    def __init__(self, calls: MlflowCalls):
        self.calls = calls
        self.info = types.SimpleNamespace(run_id="dummy-run")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class StubMlflow:
    def __init__(self):
        self.calls = MlflowCalls()
        self._experiments: List[str] = []
        self.sklearn = types.SimpleNamespace(log_model=lambda *args, **kwargs: None)
        self.tracking = types.SimpleNamespace(
            MlflowClient=lambda: types.SimpleNamespace(
                log_metric=lambda *args, **kwargs: None
            )
        )

    def set_tracking_uri(self, *args, **kwargs):
        return None

    def set_experiment(self, name: str):
        self._experiments.append(name)

    def start_run(self, *args, **kwargs):
        return DummyRun(self.calls)

    def active_run(self):
        return None

    def log_params(self, params: Dict[str, object]):
        self.calls.params.append(dict(params))

    def log_metrics(self, metrics: Dict[str, float]):
        self.calls.metrics.append(dict(metrics))

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        self.calls.metric_calls.append({"key": key, "value": value, "step": step})

    def log_dict(self, payload, artifact_file: str):
        self.calls.dicts.append({"payload": payload, "artifact": artifact_file})

    def log_text(self, text: str, artifact_file: str):
        self.calls.texts.append(f"{artifact_file}:{text}")

    def set_tags(self, tags: Dict[str, str]):
        self.calls.tags.append(dict(tags))

    def log_figure(self, figure, artifact_file: str):
        self.calls.figures.append(artifact_file)


@pytest.fixture()
def stub_mlflow(monkeypatch):
    stub = StubMlflow()

    import bid_predictor.tracking as tracking
    import train as train_module

    monkeypatch.setattr(tracking, "mlflow", stub)
    monkeypatch.setattr(train_module, "mlflow", stub)

    return stub


@pytest.fixture()
def sample_feature_config():
    feature_metadata = {
        "feature_num": {
            "categorical": False,
            "include_in_model": True,
            "derived": False,
            "flight_feature": False,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "feature_cat": {
            "categorical": True,
            "include_in_model": True,
            "derived": False,
            "flight_feature": False,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "item_count": {
            "categorical": False,
            "include_in_model": True,
            "derived": False,
            "flight_feature": False,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "seats_available": {
            "categorical": False,
            "include_in_model": True,
            "derived": False,
            "flight_feature": False,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
            "monotonicity": 1,
        },
        "usd_base_amount": {
            "categorical": False,
            "include_in_model": False,
            "derived": False,
            "flight_feature": False,
            "bid_feature": True,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "departure_timestamp": {
            "categorical": False,
            "include_in_model": False,
            "derived": False,
            "flight_feature": True,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "current_timestamp": {
            "categorical": False,
            "include_in_model": False,
            "derived": False,
            "flight_feature": True,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "carrier_code": {
            "categorical": True,
            "include_in_model": False,
            "derived": False,
            "flight_feature": True,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "flight_number": {
            "categorical": True,
            "include_in_model": False,
            "derived": False,
            "flight_feature": True,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "travel_date": {
            "categorical": False,
            "include_in_model": False,
            "derived": False,
            "flight_feature": True,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "upgrade_type": {
            "categorical": True,
            "include_in_model": False,
            "derived": False,
            "flight_feature": False,
            "bid_feature": True,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
        "snapshot_num": {
            "categorical": False,
            "include_in_model": False,
            "derived": False,
            "flight_feature": False,
            "bid_feature": False,
            "comp_feature": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        },
    }

    for metadata in feature_metadata.values():
        metadata["monotonicity"] = 0

    pre_features = [
        "feature_num",
        "feature_cat",
        "item_count",
        "seats_available",
        "usd_base_amount",
        "departure_timestamp",
        "current_timestamp",
    ] + _GROUPBY_KEY_FEATURES

    return {
        "pre_features": pre_features,
        "features": ["feature_num", "feature_cat", "item_count", "seats_available"],
        "cat_features": ["feature_cat"],
        "flight_features": [],
        "bid_features": [],
        "comp_features": [],
        "feature_metadata": feature_metadata,
        "impute_value": [],
        "impute_median": [],
        "outlier": [],
        "bins": [],
        "monotone_constraints": [0,0,0,0],
    }


@pytest.fixture()
def sample_training_dataframe():
    rng = np.random.default_rng(0)

    rows = []
    for i in range(60):
        base_date = pd.Timestamp("2023-07-10") + pd.Timedelta(days=i)
        if i >= 40:
            base_date = pd.Timestamp("2023-08-02") + pd.Timedelta(days=i - 40)
        current_ts = base_date - pd.Timedelta(hours=12)
        departure_ts = base_date + pd.Timedelta(hours=2)
        decision_ts = current_ts + pd.Timedelta(hours=6)
        rows.append(
            {
                "carrier_code": "AC" if i % 2 == 0 else "UA",
                "flight_number": f"{100 + (i % 5)}",
                "travel_date": base_date.normalize(),
                "upgrade_type": "BUS",
                "snapshot_num": i % 3,
                "departure_timestamp": departure_ts,
                "current_timestamp": current_ts,
                "decision_timestamp": decision_ts,
                "offer_status": "TICKETED" if i % 3 == 0 else "EXPIRED",
                "id": i,
                "feature_num": rng.normal(),
                "feature_cat": "A" if i % 2 == 0 else "B",
                "item_count": int(rng.integers(1, 4)),
                "seats_available": int(rng.integers(0, 5)),
                "usd_base_amount": float(rng.uniform(100, 400)),
            }
        )

    df = pd.DataFrame(rows)
    df["feature_cat"] = df["feature_cat"].astype("category")
    df["flight_number"] = df["flight_number"].astype("category")
    df["carrier_code"] = df["carrier_code"].astype("category")
    df["travel_date"] = pd.to_datetime(df["travel_date"])
    df["current_timestamp"] = pd.to_datetime(df["current_timestamp"])
    df["departure_timestamp"] = pd.to_datetime(df["departure_timestamp"])
    df["decision_timestamp"] = pd.to_datetime(df["decision_timestamp"])

    return df
