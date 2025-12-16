import os
from pathlib import Path

import pytest

from bid_predictor import feature_config


def test_load_feature_config_from_file(tmp_path, monkeypatch):
    config_path = tmp_path / "features.yaml"
    config_path.write_text(
        """
features:
  feature_num:
    include_in_model: true
  feature_cat:
    categorical: true
    include_in_model: true
  derived_feature:
    include_in_model: false
    derived: true
"""
    )

    loaded = feature_config.load_feature_config(str(config_path))
    assert loaded["features"] == ["feature_num", "feature_cat"]
    for key in feature_config._GROUPBY_KEY_FEATURES:
        assert key in loaded["pre_features"]
    assert "feature_cat" in loaded["cat_features"]


def test_load_feature_config_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        feature_config.load_feature_config(str(missing))


def test_feature_config_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("features: ['x']\n")
    feature_config.load_feature_config.cache_clear()
    monkeypatch.setenv(feature_config._FEATURE_CONFIG_ENV, str(cfg))
    loaded = feature_config.load_feature_config()
    assert loaded["features"] == ["x"]
    monkeypatch.delenv(feature_config._FEATURE_CONFIG_ENV)
    feature_config.load_feature_config.cache_clear()


def test_parse_feature_spec_rejects_invalid_type():
    with pytest.raises(TypeError):
        feature_config._parse_feature_spec(123)  # type: ignore[arg-type]


def test_ensure_groupby_keys_appends_missing():
    base = ["feature_a"]
    ensured = feature_config._ensure_groupby_keys(base)
    for key in feature_config._GROUPBY_KEY_FEATURES:
        assert key in ensured
