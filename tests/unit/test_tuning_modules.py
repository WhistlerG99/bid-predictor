import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

skopt = pytest.importorskip("skopt")
from skopt.space import Categorical, Integer, Real  # type: ignore  # noqa: E402

from bid_predictor.tuning import (
    cross_validation,
    data_access,
    feature_tuning,
    result_writing,
    search_config,
    search_grid,
)
from tune_catboost import _normalize_structure, normalize_search_value, stringify_param_value


class SimpleEstimator(BaseEstimator):
    def __init__(self, constant: float = 0.5):
        self.constant = constant

    def fit(self, X, y, eval_set=None):
        self.constant_ = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return self

    def predict(self, X):
        return (np.full(len(X), self.constant_) >= 0.5).astype(int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.full((n, 2), 0.0)
        proba[:, 1] = self.constant_
        proba[:, 0] = 1 - self.constant_
        return proba


@pytest.fixture()
def simple_data():
    X = pd.DataFrame({"x": np.linspace(0, 1, 20)})
    y = (X["x"] > 0.5).astype(int).values
    return X, y


def test_split_indices_pandas(simple_data):
    X, y = simple_data
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    train_idx, val_idx = next(cv.split(X, y))
    X_tr, X_val, y_tr, y_val = cross_validation.split_indices(X, y, (train_idx, val_idx))
    assert len(X_tr) + len(X_val) == len(X)
    assert len(y_tr) + len(y_val) == len(y)


def test_score_fold_uses_predict_proba(simple_data):
    X, y = simple_data
    estimator = SimpleEstimator()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    train_idx, val_idx = next(cv.split(X, y))
    X_tr, X_val, y_tr, y_val = cross_validation.split_indices(X, y, (train_idx, val_idx))
    model = estimator.fit(X_tr, y_tr)
    score = cross_validation.score_fold(model, cross_validation.get_scorer("roc_auc"), X_val, y_val)
    assert 0 <= score <= 1


def test_cross_validate_with_eval(simple_data):
    X, y = simple_data
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    estimator = SimpleEstimator()
    scores = cross_validation.cross_validate_with_eval(estimator, X, y, cv, "roc_auc")
    assert len(scores) == 2


def test_split_combination_and_apply_overrides():
    combo = {
        "catboost__iterations": 10,
        "transform__impute_value__feature_cat": 1,
    }
    cat_params, overrides = feature_tuning.split_combination(combo)
    assert cat_params == {"iterations": 10}
    assert overrides["impute_value"]["feature_cat"] == 1

    metadata = {
        "feature_cat": {"impute_value": None, "include_in_model": True, "derived": False, "categorical": True},
    }
    feature_tuning.apply_transform_overrides(metadata, overrides)
    assert metadata["feature_cat"]["impute_value"] == 1


def test_rebuild_feature_config_respects_overrides():
    metadata = {
        "feature_num": {
            "include_in_model": True,
            "derived": False,
            "categorical": False,
            "impute_value": None,
            "impute_median": False,
            "outlier": None,
            "bins": None,
        }
    }
    rebuilt = feature_tuning.rebuild_feature_config(metadata)
    assert "feature_num" in rebuilt["features"]


def test_summarize_transform_params_serializes_dicts():
    overrides = {
        "outlier": {"feature": {"min": 0, "max": 1}},
        "impute_median": {"feature": np.bool_(True)},
    }
    summary = feature_tuning.summarize_transform_params(overrides)
    assert summary["outlier.feature"] == "{\"max\": 1, \"min\": 0}"
    assert summary["impute_median.feature"] is True


def test_normalize_and_stringify_handle_numpy_bool():
    raw = np.bool_(True)
    normalized = normalize_search_value(raw)
    assert isinstance(normalized, bool)
    assert normalized is True

    stringified = stringify_param_value(np.bool_(False))
    assert isinstance(stringified, bool)
    assert stringified is False


def test_normalize_structure_recurses_through_collections():
    payload = {
        "a": np.int64(5),
        "b": [np.float32(1.2), {"flag": np.bool_(True)}],
        "c": {np.int64(1), np.int64(2)},
    }

    normalized = _normalize_structure(payload)

    assert normalized["a"] == 5 and isinstance(normalized["a"], int)
    assert isinstance(normalized["b"][0], float)
    assert normalized["b"][1]["flag"] is True
    assert normalized["c"] == [1, 2]


def test_load_search_config_defaults(tmp_path):
    cfg = search_config.load_search_config(None)
    assert "catboost" in cfg

    path = tmp_path / "search.yaml"
    path.write_text("catboost: {iterations: [10]}\n")
    cfg_file = search_config.load_search_config(str(path))
    assert cfg_file["catboost"]["iterations"] == [10]


def test_build_parameter_grid_handles_empty_sections():
    cfg = {"catboost": {"iterations": [10]}, "transform": {}}
    grid = search_grid.build_parameter_grid(cfg)
    assert grid["catboost__iterations"] == [10]


def test_build_search_space_infers_dimensions():
    cfg = {
        "catboost": {
            "iterations": [50, 100],
            "learning_rate": [0.01, 0.1],
            "depth": [6],
            "bootstrap_type": ["Bernoulli", "Bayesian"],
        },
        "transform": {"outlier": {"feature": [{"max": 5}, {"max": 8}]}}
    }
    dimensions = search_grid.build_search_space(cfg)
    dim_map = {name: dim for name, dim in dimensions}

    assert isinstance(dim_map["catboost__iterations"], Integer)
    assert isinstance(dim_map["catboost__learning_rate"], Real)
    assert isinstance(dim_map["catboost__depth"], Categorical)
    assert dim_map["catboost__depth"].is_constant
    assert isinstance(dim_map["catboost__bootstrap_type"], Categorical)
    assert isinstance(dim_map["transform__outlier__feature"], Categorical)
    outlier_categories = dim_map["transform__outlier__feature"].categories
    assert all(isinstance(cat, search_grid.FrozenSearchValue) for cat in outlier_categories)
    assert search_grid.unwrap_search_value(outlier_categories[0]) == {"max": 5}


def test_write_best_catboost_params(tmp_path):
    path = tmp_path / "params.yaml"
    params = {"iterations": 120, "learning_rate": 0.15}

    written = result_writing.write_best_catboost_params(path, params)
    assert written == path

    loaded = yaml.safe_load(path.read_text())
    assert loaded["catboost"]["iterations"] == 120
    assert pytest.approx(loaded["catboost"]["learning_rate"], rel=1e-6) == 0.15


def test_resolve_train_file_uses_environment(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_DATA", "s3://bucket")
    monkeypatch.setattr(data_access, "detect_execution_environment", lambda: ("sagemaker_notebook", ""))
    path = data_access.resolve_train_file(None)
    assert path.startswith("s3://bucket")


def test_load_training_data_uses_pyarrow(monkeypatch):
    class DummyTable:
        def to_table(self):
            raise AssertionError("not called")

    class DummyDataset:
        def __init__(self, path, format):
            self.path = path
            self.format = format

        def to_table(self):
            return types.SimpleNamespace(
                to_pandas=lambda: pd.DataFrame({"carrier_code": ["AC"], "fare_class": ["Y"], "current_available_seats": [1]})
            )

    import types

    monkeypatch.setattr(data_access.ds, "dataset", lambda path, format: DummyDataset(path, format))
    df = data_access.load_training_data("dummy")
    assert list(df.columns) == ["carrier_code", "fare_class", "seats_available"]


def test_result_writing_helpers(tmp_path):
    df = pd.DataFrame({"mean_score": [0.9], "std_score": [0.01]})
    csv_path = tmp_path / "results" / "scores.csv"
    saved_csv = result_writing.write_results_csv(csv_path, df)
    assert saved_csv.exists()

    feature_config = {
        "feature_metadata": {"feature": {"include_in_model": True, "derived": False}},
    }
    yaml_path = tmp_path / "results" / "best.yaml"
    saved_yaml = result_writing.write_best_feature_config(yaml_path, feature_config)
    assert saved_yaml.exists()
    content = yaml.safe_load(saved_yaml.read_text())
    assert "features" in content and "feature" in content["features"]

    payload = {"score": 0.9}
    json_path = tmp_path / "results" / "best.json"
    saved_json = result_writing.write_best_result_json(json_path, payload)
    assert saved_json.exists()
    assert json.loads(saved_json.read_text())["score"] == 0.9
