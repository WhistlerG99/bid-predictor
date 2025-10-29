"""Ensure key UI modules import without raising errors."""

from importlib import import_module
import sys
import types

import pytest


MODULE_NAMES = (
    "dash_app",
    "bid_predictor.ui",
    "bid_predictor.ui.snapshot",
    "bid_predictor.ui.snapshot.layout",
    "bid_predictor.ui.snapshot.filters",
    "bid_predictor.ui.snapshot.predictions",
    "bid_predictor.ui.snapshot.view",
    "bid_predictor.ui.feature_sensitivity",
    "bid_predictor.ui.feature_sensitivity.layout",
    "bid_predictor.ui.feature_sensitivity.filters",
    "bid_predictor.ui.feature_sensitivity.baseline",
    "bid_predictor.ui.feature_sensitivity.table",
)


class _DummyComponent:
    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return {"component": self._name, "args": args, "kwargs": kwargs}

    def __getattr__(self, item: str) -> "_DummyComponent":
        return _DummyComponent(f"{self._name}.{item}")


class _DummyModule(types.ModuleType):
    def __getattr__(self, item: str) -> _DummyComponent:  # type: ignore[override]
        return _DummyComponent(item)


class _DummyDash:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.layout = None

    def callback(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        def decorator(func):
            return func

        return decorator


@pytest.fixture(autouse=True)
def _stub_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a lightweight Dash stub when the dependency is missing."""

    try:
        import_module("dash")
        return
    except ModuleNotFoundError:
        pass

    dash_stub = types.ModuleType("dash")
    dash_stub.Dash = _DummyDash
    dash_stub.Input = type("Input", (), {})
    dash_stub.Output = type("Output", (), {})
    dash_stub.State = type("State", (), {})
    dash_stub.callback_context = None
    dash_stub.no_update = object()
    dash_stub.dcc = _DummyModule("dash.dcc")
    dash_stub.html = _DummyModule("dash.html")
    dash_stub.dash_table = _DummyModule("dash.dash_table")

    monkeypatch.setitem(sys.modules, "dash", dash_stub)
    monkeypatch.setitem(sys.modules, "dash.dcc", dash_stub.dcc)
    monkeypatch.setitem(sys.modules, "dash.html", dash_stub.html)
    monkeypatch.setitem(sys.modules, "dash.dash_table", dash_stub.dash_table)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_import(module_name: str) -> None:
    """Verify the module can be imported without raising ImportError."""

    module = import_module(module_name)
    assert module is not None
