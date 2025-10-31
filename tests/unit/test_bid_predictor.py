import cloudpickle
import yaml

from bid_predictor import bid_predictor
from bid_predictor.feature_config import load_feature_config
from bid_predictor.bid_predictor import _CATBOOST_PARAM_DEFAULTS


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
    assert step_names[0:4] == ["flight_code", "depart", "bid_rank", "group"]
    assert step_names[-1] == "clf"


def test_pipeline_omits_monotone_constraints_when_not_configured(sample_feature_config):
    pipeline = bid_predictor.build_pipeline(feature_config=sample_feature_config)

    assert "monotone_constraints" not in pipeline.named_steps["clf"].cb_params


def test_pipeline_applies_monotone_constraints(sample_feature_config):
    feature_metadata = sample_feature_config["feature_metadata"]
    feature_metadata["feature_num"]["monotonicity"] = 1
    feature_metadata["item_count"]["monotonicity"] = -1

    pipeline = bid_predictor.build_pipeline(feature_config=sample_feature_config)

    assert pipeline.named_steps["clf"].cb_params["monotone_constraints"] == [1, 0, -1, 0]


def test_pipeline_persists_feature_config(sample_feature_config, tmp_path):
    pipeline = bid_predictor.build_pipeline(feature_config=sample_feature_config)

    assert pipeline.feature_config_ == sample_feature_config
    assert pipeline.__class__.feature_config == sample_feature_config

    model_path = tmp_path / "pipeline.pkl"
    with model_path.open("wb") as handle:
        cloudpickle.dump(pipeline, handle)

    with model_path.open("rb") as handle:
        loaded = cloudpickle.load(handle)

    assert loaded.feature_config_ == sample_feature_config
    assert loaded.__class__.feature_config == sample_feature_config


def test_dump_feature_config_roundtrip(tmp_path, sample_feature_config):
    pipeline = bid_predictor.build_pipeline(feature_config=sample_feature_config)

    output = tmp_path / "feature_config.yaml"
    pipeline.dump_feature_config(output)

    with output.open("r", encoding="utf-8") as handle:
        dumped_payload = yaml.safe_load(handle)

    features_yaml = dumped_payload.get("features", {})
    for feature_name, metadata in sample_feature_config["feature_metadata"].items():
        emitted_metadata = features_yaml.get(feature_name, {})
        for key, value in metadata.items():
            if value is None:
                assert key not in emitted_metadata

    reloaded = load_feature_config(output)

    assert reloaded == sample_feature_config


def test_dump_catboost_params_roundtrip(tmp_path, sample_feature_config):
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

    clf_params = pipeline.named_steps["clf"].cb_params
    assert pipeline.__class__.catboost_params == clf_params

    output = tmp_path / "catboost_params.yaml"
    pipeline.dump_catboost_params(output)

    with output.open("r", encoding="utf-8") as handle:
        dumped_payload = yaml.safe_load(handle)

    catboost_yaml = dumped_payload.get("catboost", {})
    assert isinstance(catboost_yaml, dict)

    sentinel = object()
    for key, value in clf_params.items():
        default_value = _CATBOOST_PARAM_DEFAULTS.get(key, sentinel)
        if default_value is not sentinel and default_value == value:
            assert key not in catboost_yaml
        else:
            assert key in catboost_yaml
            assert catboost_yaml[key] == value

    for key in catboost_yaml:
        default_value = _CATBOOST_PARAM_DEFAULTS.get(key, sentinel)
        if default_value is not sentinel:
            assert default_value != catboost_yaml[key]

    # Ensure the emitted YAML can serve as a train.py configuration input
    assert dumped_payload == {"catboost": catboost_yaml}
