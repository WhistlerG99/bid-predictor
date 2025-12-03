import os
from pathlib import Path

import pytest

from bid_predictor import utils


def test_detect_execution_environment_local(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SM_") or key.startswith("SAGEMAKER"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)
    monkeypatch.setattr(utils, "_in_jupyter", lambda: False)
    monkeypatch.setattr(utils, "_any_file_exists", lambda paths: False)
    env, reason = utils.detect_execution_environment()
    assert env == "local"
    assert "No SageMaker" in reason


def test_detect_execution_environment_job(monkeypatch):
    monkeypatch.setattr(utils, "_any_file_exists", lambda paths: True)
    env, _ = utils.detect_execution_environment()
    assert env == "sagemaker_job"


def test_get_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CATBOOST_TRAIN_DIR", str(tmp_path / "cat"))
    output_dir = utils.get_output_dir()
    assert Path(output_dir).exists()
