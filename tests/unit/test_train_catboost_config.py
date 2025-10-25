import pytest

import train


def test_merge_prefers_file_when_flag_missing():
    cli_params = {"iterations": 200, "depth": 4}
    file_params = {"iterations": 120, "depth": 6}
    merged = train._merge_catboost_params(cli_params, file_params, set())
    assert merged == {"iterations": 120, "depth": 6}


def test_merge_respects_explicit_flag():
    cli_params = {"iterations": 200, "depth": 4}
    file_params = {"iterations": 120, "depth": 6}
    merged = train._merge_catboost_params(cli_params, file_params, {"iterations"})
    assert merged == {"iterations": 200, "depth": 6}


def test_load_catboost_config_supports_nested_section(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text("catboost:\n  iterations: 111\n  depth: 3\n")
    params = train._load_catboost_config(str(path))
    assert params == {"iterations": 111, "depth": 3}


def test_load_catboost_config_returns_empty_for_missing():
    params = train._load_catboost_config(None)
    assert params == {}


def test_load_catboost_config_requires_mapping(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError):
        train._load_catboost_config(str(path))
