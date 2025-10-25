from bid_predictor import bid_predictor


def test_cbc_filters_missing_cat_features(sample_feature_config, sample_training_dataframe, stub_mlflow):
    pipeline = bid_predictor.build_pipeline(
        feature_config=sample_feature_config,
        iterations=5,
        depth=2,
        learning_rate=0.2,
        l2_leaf_reg=1.0,
        random_seed=42,
        logging_level="Silent",
        task_type="CPU",
        devices="0",
        custom_metric=["AUC"],
    )
    X = sample_training_dataframe[sample_feature_config["pre_features"]]
    y = (sample_training_dataframe["offer_status"] == "TICKETED").astype(int)
    pipeline.fit(X, y, eval_set=(X, y))
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape[0] == X.shape[0]


def test_build_pipeline_returns_expected_steps(sample_feature_config):
    pipeline = bid_predictor.build_pipeline(feature_config=sample_feature_config)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names[0:3] == ["flight_code", "depart", "group"]
    assert step_names[-1] == "clf"
