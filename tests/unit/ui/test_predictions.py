import pandas as pd

from bid_predictor.ui.predictions import predict


class DummyModel:
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"Acceptance Probability": [0.25, 0.75]})


def test_predict_adds_probability_and_warning(monkeypatch):
    def fake_loader(model_uri: str):
        return DummyModel()

    monkeypatch.setattr("bid_predictor.ui.predictions.load_model_cached", fake_loader)
    monkeypatch.setattr(
        "bid_predictor.ui.predictions.get_feature_columns",
        lambda: (["feature_a", "feature_b"], []),
    )

    df = pd.DataFrame({"feature_a": [1.0, 2.0]})
    result = predict("model://dummy", df)

    assert result is df
    assert list(result["Acceptance Probability"]) == [25.0, 75.0]
    assert result.attrs["model_warning"].startswith("Added missing feature config")
