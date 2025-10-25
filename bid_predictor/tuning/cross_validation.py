"""Custom cross-validation helpers for CatBoost tuning."""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold


def split_indices(
    X: Any, y: Any, indices: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Any, Any, Any, Any]:
    """Split arrays or pandas objects by the provided fold indices."""

    train_idx, val_idx = indices

    if hasattr(X, "iloc"):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
    else:
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]

    if hasattr(y, "iloc"):
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
    else:
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

    return X_train_fold, X_val_fold, y_train_fold, y_val_fold


def call_prediction_interface(model, method: str, X_val):
    """Invoke the requested prediction interface on a fitted estimator."""

    if hasattr(model, method):
        predictor = getattr(model, method)
        return predictor(X_val)

    if hasattr(model, "steps"):
        Xt = X_val
        for _, transformer in model.steps[:-1]:
            if transformer in (None, "passthrough"):
                continue
            if hasattr(transformer, "transform"):
                Xt = transformer.transform(Xt)
            elif callable(transformer):
                Xt = transformer(Xt)
            else:
                raise AttributeError(
                    "Pipeline step does not expose a transform method"
                )

        final_estimator = model.steps[-1][1]
        predictor = getattr(final_estimator, method)
        return predictor(Xt)

    predictor = getattr(model, method)
    return predictor(X_val)


def score_fold(model, scorer, X_val_fold, y_val_fold) -> float:
    """Evaluate a fitted estimator on the validation fold.

    Mirrors scikit-learn's scorer invocation while bypassing the response-method
    validation that misidentifies the CatBoost pipeline. The scorer's factory
    arguments determine the appropriate prediction interface to call.
    """

    factory_args = getattr(scorer, "_factory_args", {})
    needs_proba = factory_args.get("needs_proba", False)
    needs_threshold = factory_args.get("needs_threshold", False)

    if needs_proba:
        try:
            y_pred = call_prediction_interface(model, "predict_proba", X_val_fold)
        except AttributeError as exc:
            raise ValueError(
                "Scorer requires predict_proba but estimator lacks it"
            ) from exc
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    elif needs_threshold:
        try:
            y_pred = call_prediction_interface(model, "predict_proba", X_val_fold)
            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
        except AttributeError:
            try:
                y_pred = call_prediction_interface(
                    model, "decision_function", X_val_fold
                )
            except AttributeError:
                y_pred = call_prediction_interface(model, "predict", X_val_fold)
    else:
        y_pred = call_prediction_interface(model, "predict", X_val_fold)

    return float(
        scorer._sign * scorer._score_func(y_val_fold, y_pred, **scorer._kwargs)
    )


def cross_validate_with_eval(
    estimator, X: Any, y: Any, cv: StratifiedKFold, scoring: str
) -> List[float]:
    """Run cross-validation with CatBoost evaluation sets."""

    scorer = get_scorer(scoring)
    scores: List[float] = []

    for split in cv.split(X, y):
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = split_indices(X, y, split)

        model = clone(estimator)
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))
        fold_score = score_fold(model, scorer, X_val_fold, y_val_fold)
        scores.append(fold_score)

    return scores
