from types import SimpleNamespace

import numpy as np
import pandas as pd

from bid_predictor.ui.tables import apply_table_edits, build_bid_table


def _to_frame(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


class _IdentityTransformer:
    def transform(self, data):
        return _to_frame(data)


class _CompetitorTransformer:
    def transform(self, data):
        df = _to_frame(data)
        if "usd_base_amount" not in df.columns:
            df["usd_base_amount"] = np.nan
        values = df["usd_base_amount"].astype(float).to_numpy()
        results = []
        for idx in range(len(df)):
            peers = [
                values[j]
                for j in range(len(values))
                if j != idx and not np.isnan(values[j])
            ]
            results.append(float(np.median(peers)) if peers else None)
        df["usd_base_amount_50%"] = results
        return df


class _ReduceTransformer:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
        df = _to_frame(data)
        for column in self.columns:
            if column not in df.columns:
                df[column] = None
        return df[self.columns]


class _NumpyReduceTransformer:
    def __init__(self, columns):
        self.columns = columns

    def transform(self, data):
        df = _to_frame(data)
        for column in self.columns:
            if column not in df.columns:
                df[column] = None
        return df[self.columns].to_numpy()


def _fake_model():
    return SimpleNamespace(
        steps=[
            ("identity", _IdentityTransformer()),
            ("competitor", _CompetitorTransformer()),
            (
                "reduce",
                _ReduceTransformer(
                    [
                        "usd_base_amount",
                        "item_count",
                        "multiplier_loyalty",
                        "usd_base_amount_50%",
                        "num_offers",
                    ]
                ),
            ),
            ("clf", object()),
        ]
    )


def test_build_bid_table_formats_predictions():
    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
            "fare_class": "M",
            "offer_time": 1.0,
            "multiplier_fare_class": 1.0,
            "multiplier_loyalty": 1.0,
            "multiplier_success_history": 1.0,
            "multiplier_payment_type": 1.0,
            "offer_status": "pending",
        }
    ]
    predictions = {"bid_0": 0.87654}

    columns, data_rows, styles = build_bid_table(records, predictions)

    assert columns[0]["id"] == "Feature"
    assert columns[1]["name"] == "Bid 1"
    acceptance_row = next(row for row in data_rows if row["Feature"] == "Acceptance Probability")
    assert acceptance_row["bid_0"] == 0.8765
    assert any(rule["if"]["filter_query"].endswith("Acceptance Probability\"") for rule in styles)


def test_apply_table_edits_updates_numeric_fields():
    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
            "fare_class": "M",
            "offer_time": 1.0,
            "multiplier_fare_class": 1.0,
            "multiplier_loyalty": 1.0,
            "multiplier_success_history": 1.0,
            "multiplier_payment_type": 1.0,
            "offer_status": "pending",
        },
        {
            "Bid #": 2,
            "item_count": 3,
            "usd_base_amount": 80.0,
            "fare_class": "N",
            "offer_time": 2.0,
            "multiplier_fare_class": 1.0,
            "multiplier_loyalty": 1.0,
            "multiplier_success_history": 1.0,
            "multiplier_payment_type": 1.0,
            "offer_status": "pending",
        },
    ]

    table_data = [
        {"Feature": "item_count", "bid_0": 4, "bid_1": 5},
        {"Feature": "usd_base_amount", "bid_0": 150.0, "bid_1": 90.0},
        {"Feature": "fare_class", "bid_0": "Q", "bid_1": "R"},
        {"Feature": "offer_time", "bid_0": 1.25, "bid_1": 2.5},
    ]
    columns = [
        {"id": "Feature", "name": "Feature"},
        {"id": "bid_0", "name": "Bid 1"},
        {"id": "bid_1", "name": "Bid 2"},
    ]

    updated = apply_table_edits(records, table_data, columns)

    assert updated is not None
    assert updated[0]["item_count"] == 4
    assert updated[0]["usd_base_amount"] == 150.0
    assert updated[1]["fare_class"] == "R"
    assert updated[1]["offer_time"] == 2.5


def test_build_bid_table_disables_offer_status_editing():
    records = [
        {
            "Bid #": 1,
            "offer_status": "pending",
        }
    ]

    columns, _, styles = build_bid_table(records, {})

    assert any(column.get("editable", True) for column in columns if column["id"] != "Feature")
    offer_status_rule = next(
        rule
        for rule in styles
        if rule.get("if", {}).get("filter_query") == '{Feature} = "offer_status"'
    )
    assert offer_status_rule["pointerEvents"] == "none"


def test_apply_table_edits_ignores_offer_status_changes():
    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "offer_status": "pending",
        }
    ]

    table_data = [
        {"Feature": "item_count", "bid_0": 4},
        {"Feature": "offer_status", "bid_0": "accepted"},
    ]
    columns = [
        {"id": "Feature", "name": "Feature"},
        {"id": "bid_0", "name": "Bid 1"},
    ]

    updated = apply_table_edits(records, table_data, columns)

    assert updated is not None
    assert updated[0]["item_count"] == 4
    assert updated[0]["offer_status"] == "pending"


def test_build_bid_table_locks_bid_specific_cells():
    records = [
        {"Bid #": 1, "item_count": 2},
        {"Bid #": 2, "item_count": 3},
    ]

    columns, _, styles = build_bid_table(
        records,
        {},
        locked_cells={"bid_1": ["item_count"]},
    )

    assert any(
        rule.get("if", {}).get("column_id") == "bid_1" and
        rule.get("if", {}).get("filter_query") == '{Feature} = "item_count"'
        for rule in styles
    )


def test_apply_table_edits_skips_locked_features():
    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
        }
    ]

    table_data = [
        {"Feature": "item_count", "bid_0": 5},
        {"Feature": "usd_base_amount", "bid_0": 150.0},
    ]

    columns = [
        {"id": "Feature", "name": "Feature"},
        {"id": "bid_0", "name": "Bid 1"},
    ]

    updated = apply_table_edits(
        records,
        table_data,
        columns,
        locked_cells={"bid_0": ["item_count"]},
    )

    assert updated is not None
    assert updated[0]["item_count"] == 2
    assert updated[0]["usd_base_amount"] == 150.0


def test_build_bid_table_uses_model_features_and_locks_competitors(monkeypatch):
    monkeypatch.setattr("bid_predictor.ui.tables.load_model_cached", lambda uri: _fake_model())
    model_uri = "model://fake"

    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
            "multiplier_loyalty": 1.0,
            "num_offers": 2,
            "conf_num": "ABC",
        },
        {
            "Bid #": 2,
            "item_count": 3,
            "usd_base_amount": 80.0,
            "multiplier_loyalty": 1.0,
            "num_offers": 2,
            "conf_num": "DEF",
        },
    ]

    columns, data_rows, styles = build_bid_table(records, {}, model_uri=model_uri)

    features = [row["Feature"] for row in data_rows]
    assert "conf_num" not in features
    assert "usd_base_amount_50%" in features
    assert "multiplier_loyalty" in features
    assert "num_offers" not in features

    competitor_rule = next(
        rule
        for rule in styles
        if rule.get("if", {}).get("filter_query") == '{Feature} = "usd_base_amount_50%"'
    )
    assert competitor_rule["pointerEvents"] == "none"

    competitor_row = next(row for row in data_rows if row["Feature"] == "usd_base_amount_50%")
    assert competitor_row["bid_0"] == 80.0
    assert competitor_row["bid_1"] == 100.0


def test_build_bid_table_keeps_uniform_model_features(monkeypatch):
    monkeypatch.setattr("bid_predictor.ui.tables.load_model_cached", lambda uri: _fake_model())
    model_uri = "model://fake"

    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
            "multiplier_loyalty": 1.0,
            "num_offers": 2,
        },
        {
            "Bid #": 2,
            "item_count": 3,
            "usd_base_amount": 90.0,
            "multiplier_loyalty": 1.0,
            "num_offers": 2,
        },
    ]

    columns, data_rows, _ = build_bid_table(records, {}, model_uri=model_uri)

    loyalty_row = next(row for row in data_rows if row["Feature"] == "multiplier_loyalty")
    assert loyalty_row["bid_0"] == 1.0
    assert loyalty_row["bid_1"] == 1.0

    table_data = [{"Feature": "multiplier_loyalty", "bid_0": 1.5, "bid_1": 1.5}]

    updated = apply_table_edits(records, table_data, columns, model_uri=model_uri)

    assert updated is not None
    assert updated[0]["multiplier_loyalty"] == 1.5
    assert updated[1]["multiplier_loyalty"] == 1.5


def test_apply_table_edits_recomputes_competitor_features(monkeypatch):
    monkeypatch.setattr("bid_predictor.ui.tables.load_model_cached", lambda uri: _fake_model())
    model_uri = "model://fake"

    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
            "usd_base_amount_50%": 80.0,
        },
        {
            "Bid #": 2,
            "item_count": 3,
            "usd_base_amount": 80.0,
            "usd_base_amount_50%": 100.0,
        },
    ]

    table_data = [
        {"Feature": "usd_base_amount", "bid_0": 120.0, "bid_1": 90.0},
        {"Feature": "usd_base_amount_50%", "bid_0": 0.0, "bid_1": 0.0},
        {"Feature": "item_count", "bid_0": 2, "bid_1": 3},
    ]
    columns = [
        {"id": "Feature", "name": "Feature"},
        {"id": "bid_0", "name": "Bid 1"},
        {"id": "bid_1", "name": "Bid 2"},
    ]

    updated = apply_table_edits(
        records,
        table_data,
        columns,
        model_uri=model_uri,
    )

    assert updated is not None
    assert updated[0]["usd_base_amount"] == 120.0
    assert updated[1]["usd_base_amount"] == 90.0
    # competitor feature recomputed from updated amounts (other bid's median)
    assert updated[0]["usd_base_amount_50%"] == 90.0
    assert updated[1]["usd_base_amount_50%"] == 120.0


def test_build_bid_table_uses_model_feature_names_when_array_output(monkeypatch):
    def _array_model():
        return SimpleNamespace(
            steps=[
                ("identity", _IdentityTransformer()),
                ("reduce", _NumpyReduceTransformer(["usd_base_amount", "item_count"])),
                ("clf", object()),
            ]
        )

    monkeypatch.setattr(
        "bid_predictor.ui.tables.load_model_cached", lambda uri: _array_model()
    )

    records = [
        {
            "Bid #": 1,
            "item_count": 2,
            "usd_base_amount": 100.0,
            "conf_num": "ABC",
        },
        {
            "Bid #": 2,
            "item_count": 3,
            "usd_base_amount": 80.0,
            "conf_num": "DEF",
        },
    ]

    columns, data_rows, _ = build_bid_table(records, {}, model_uri="model://array")

    assert [column["id"] for column in columns] == ["Feature", "bid_0", "bid_1"]
    features = [row["Feature"] for row in data_rows]
    assert "conf_num" not in features
    assert features[:2] == ["usd_base_amount", "item_count"]
