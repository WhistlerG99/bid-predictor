from importlib import import_module


def test_package_exports():
    package = import_module("bid_predictor")
    assert "bid_predictor" in package.__all__
    assert package.__version__ == "0.1.0"
