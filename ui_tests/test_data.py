import pandas as pd
import pytest

from bid_predictor_ui.data import load_dataset_cached, prepare_prediction_dataframe


@pytest.fixture(autouse=True)
def clear_caches():
    load_dataset_cached.cache_clear()
    yield
    load_dataset_cached.cache_clear()


def test_load_dataset_cached_normalizes_and_caches(monkeypatch):
    calls = []

    def fake_loader(path: str, **kwargs) -> pd.DataFrame:
        calls.append(path)
        return pd.DataFrame(
            {
                "carrier_code": ["AC"],
                "flight_number": ["123"],
                "travel_date": ["2024-01-01"],
                "upgrade_type": ["biz"],
                "snapshot_num": [1],
                "current_timestamp": ["2024-01-01T00:00:00"],
                "departure_timestamp": ["2024-01-02T00:00:00"],
            }
        )

    monkeypatch.setattr(
        "bid_predictor_ui.data.load_training_data",
        fake_loader,
    )

    df = load_dataset_cached("/tmp/data.parquet")
    assert calls == ["/tmp/data.parquet"]
    assert pd.api.types.is_datetime64_dtype(df["current_timestamp"])
    assert pd.api.types.is_datetime64_dtype(df["departure_timestamp"])
    assert pd.api.types.is_datetime64_dtype(df["travel_date"])

    _ = load_dataset_cached("/tmp/data.parquet")
    assert calls == ["/tmp/data.parquet"], "Result should be cached"


def test_load_dataset_cached_missing_columns(monkeypatch):
    def fake_loader(path: str, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"carrier_code": ["AC"]})

    monkeypatch.setattr(
        "bid_predictor_ui.data.load_training_data",
        fake_loader,
    )

    with pytest.raises(ValueError):
        load_dataset_cached("/tmp/missing.parquet")


def test_prepare_prediction_dataframe_converts_types():
    records = [
        {
            "usd_base_amount": "120.567",
            "travel_date": "2024-01-05",
            "extra_col": "keep me",
        }
    ]

    df = prepare_prediction_dataframe(
        records,
        feature_config={"pre_features": ["usd_base_amount", "travel_date"]},
    )
    assert df["usd_base_amount"].iloc[0] == pytest.approx(120.567)
    assert pd.api.types.is_datetime64_dtype(df["travel_date"])
    assert "extra_col" in df.columns
