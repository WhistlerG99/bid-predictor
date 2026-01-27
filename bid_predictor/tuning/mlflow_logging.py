"""Utilities for optional MLflow logging during hyperparameter tuning."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Optional


class _NullMlflowLogger:
    """No-op fallback when MLflow is unavailable."""

    enabled = False

    def log_params(self, params: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        return

    def log_param(self, key: str, value: Any) -> None:  # pragma: no cover - trivial
        return

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:  # pragma: no cover - trivial
        return

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:  # pragma: no cover - trivial
        return

    def log_artifact(self, path: str) -> None:  # pragma: no cover - trivial
        return


class MlflowLogger:
    """Thin wrapper around the MLflow module used inside a run."""

    def __init__(self, mlflow_module: Any):
        self._mlflow = mlflow_module
        self.enabled = True

    def log_params(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self._mlflow.log_param(key, value)

    def log_param(self, key: str, value: Any) -> None:
        self._mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if step is None:
            self._mlflow.log_metric(key, value)
        else:
            self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)


def _import_mlflow() -> Any:
    """Attempt to import MLflow, returning ``None`` if unavailable."""
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:  # pragma: no cover - optional dependency
        return None


def _patch_mlflow_metric_logging(mlflow_module: Any) -> None:
    """Wrap MLflow's ``log_metric`` to gracefully handle permission failures."""
    log_metric = getattr(mlflow_module, "log_metric", None)
    if log_metric is None:
        return
    if getattr(log_metric, "_bid_predictor_wrapped", False):
        return

    permission_state = {"warned": False}

    def safe_log_metric(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - depends on environment
        try:
            return log_metric(*args, **kwargs)
        except PermissionError as exc:
            if not permission_state["warned"]:
                permission_state["warned"] = True
                print(
                    "Warning: Skipping MLflow metric logging due to permission error. "
                    f"Details: {exc}",
                    flush=True,
                )
            return None

    safe_log_metric._bid_predictor_wrapped = True  # type: ignore[attr-defined]
    mlflow_module.log_metric = safe_log_metric  # type: ignore[assignment]


@contextmanager
def mlflow_run(
    experiment_name: Optional[str],
    run_name: str = "catboost_tuning",
):
    """Context manager yielding an MLflow logger when available."""

    mlflow_module = _import_mlflow()
    if mlflow_module is None:
        yield _NullMlflowLogger()
        return

    _patch_mlflow_metric_logging(mlflow_module)

    try:
        if experiment_name:
            mlflow_module.set_experiment(experiment_name)

        with mlflow_module.start_run(run_name=run_name):
            yield MlflowLogger(mlflow_module)
        return
    except Exception as exc:  # pragma: no cover - external system interaction
        print(
            "Warning: Failed to initialise MLflow logging. "
            f"Continuing without MLflow. Details: {exc}",
            flush=True,
        )

    yield _NullMlflowLogger()
