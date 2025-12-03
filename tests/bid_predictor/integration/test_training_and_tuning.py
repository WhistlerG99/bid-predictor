from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
import train
from bid_predictor.bid_predictor import build_pipeline
from bid_predictor.tuning.cross_validation import (
    RandomDateSplitter,
    cross_validate_with_eval,
)


@pytest.mark.integration
def test_train_pipeline_is_deterministic(sample_training_dataframe, sample_feature_config, stub_mlflow, monkeypatch):
    captured = []

    def capture_prob_examples(df):
        df_sorted = df.sort_index()["Acceptance Probability"].copy()
        captured.append(df_sorted)

    monkeypatch.setattr(train, "log_prob_examples", capture_prob_examples)

    def make_args():
        return SimpleNamespace(
            task_type="CPU",
            devices="0",
            iterations=15,
            depth=2,
            learning_rate=0.1,
            l2_leaf_reg=1.0,
            eval_metric="AUC",
            random_state=42,
            feature_config=None,
            experiment_name="tests",
            testing=True,
        )

    data = sample_training_dataframe.copy()

    for _ in range(2):
        train.train_and_log_model(data.copy(), sample_feature_config, make_args())

    assert len(captured) == 2
    assert_series_equal(captured[0], captured[1])


@pytest.mark.integration
def test_cross_validation_scores_are_reproducible(sample_training_dataframe, sample_feature_config, stub_mlflow):
    X_train, _, y_train, _, _ = train.prepare_features(
        sample_training_dataframe.copy(), sample_feature_config["pre_features"], testing=True
    )
    pipeline = build_pipeline(
        feature_config=sample_feature_config,
        iterations=15,
        depth=2,
        learning_rate=0.1,
        l2_leaf_reg=1.0,
        random_seed=42,
        logging_level="Silent",
        task_type="CPU",
        devices="0",
        custom_metric=["AUC"],
    )
    travel_dates = pd.to_datetime(
        sample_training_dataframe.loc[X_train.index, "travel_date"]
    ).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    splitter = RandomDateSplitter(
        travel_dates=travel_dates,
        n_splits=3,
        random_state=42,
        eval_ratio=1.0,
        total_size=None,
    )

    scores_first = cross_validate_with_eval(
        pipeline, X_train, y_train, splitter, "roc_auc"
    )
    scores_second = cross_validate_with_eval(
        pipeline, X_train, y_train, splitter, "roc_auc"
    )

    np.testing.assert_allclose(scores_first, scores_second)
