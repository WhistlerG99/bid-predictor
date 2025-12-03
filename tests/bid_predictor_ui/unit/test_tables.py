from bid_predictor_ui.tables import apply_table_edits, build_bid_table


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
