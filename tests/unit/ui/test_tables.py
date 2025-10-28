from bid_predictor.ui.tables import apply_table_edits, build_bid_table


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
