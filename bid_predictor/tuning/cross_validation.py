"""Custom cross-validation helpers for CatBoost tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer


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


@dataclass
class RandomDateSplitter:
    """Generate contiguous train/eval splits based on travel dates.

    For each split the splitter samples three ordered dates ``t_begin``,
    ``t_cut`` and ``t_end`` such that:

    * The training set contains rows where ``t_begin <= travel_date < t_cut``
    * The evaluation set contains rows where ``t_cut <= travel_date <= t_end``
    * The number of rows in the train/eval sets approximates the requested
      ratio while covering roughly ``total_size`` rows overall.
    """

    travel_dates: Iterable[Any]
    n_splits: int = 3
    total_size: int | None = None
    eval_ratio: float = 1.0
    random_state: int | None = None
    max_attempts: int = 100
    ratio_tolerance: float = 0.35

    def __post_init__(self) -> None:
        dates = pd.to_datetime(pd.Series(self.travel_dates), utc=False)
        if dates.empty:
            raise ValueError("travel_dates must contain at least one entry")

        unique_dates = np.unique(dates.values)
        if unique_dates.size < 2:
            raise ValueError(
                "travel_dates must contain at least two distinct dates"
            )

        if self.n_splits <= 0:
            raise ValueError("n_splits must be positive")

        if self.total_size is not None and self.total_size <= 0:
            raise ValueError("total_size must be a positive integer or None")

        if self.eval_ratio <= 0:
            raise ValueError("eval_ratio must be a positive value")

        self._dates = dates.reset_index(drop=True)
        self._unique_dates = unique_dates
        inverse = np.searchsorted(unique_dates, dates.values)
        self._counts = np.bincount(inverse, minlength=unique_dates.size)
        self._rng_state = self.random_state

    @property
    def _targets(self) -> Tuple[int, int, int]:
        total_rows = len(self._dates)
        total_size = self.total_size if self.total_size is not None else total_rows
        total_size = min(total_size, total_rows)

        train_target = max(1, int(round(total_size / (1 + self.eval_ratio))))
        eval_target = max(1, total_size - train_target)
        total_size = train_target + eval_target
        return train_target, eval_target, total_size

    def _candidate_window(
        self,
        rng: np.random.RandomState,
        train_target: int,
        eval_target: int,
        total_target: int,
    ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None:
        """Sample a contiguous train/eval window that approximates target sizes."""
        counts = self._counts
        unique_dates = self._unique_dates
        n_unique = unique_dates.size
        if n_unique < 2:
            return None

        # When sampling start indices ensure there remains at least one
        # evaluation date after the training range.
        max_start = n_unique - 1
        for _ in range(self.max_attempts):
            start_idx = int(rng.randint(0, max_start))
            if counts[start_idx:].sum() < total_target:
                continue

            train_rows = 0
            cut_idx = start_idx
            while cut_idx < n_unique and train_rows < train_target:
                train_rows += counts[cut_idx]
                cut_idx += 1

            if cut_idx >= n_unique:
                continue

            eval_rows = 0
            end_idx = cut_idx
            while end_idx < n_unique and eval_rows < eval_target:
                eval_rows += counts[end_idx]
                end_idx += 1

            if eval_rows == 0:
                continue

            t_begin = pd.Timestamp(unique_dates[start_idx])
            t_cut = pd.Timestamp(unique_dates[cut_idx])
            t_end = pd.Timestamp(unique_dates[min(end_idx, n_unique) - 1])

            train_mask = (self._dates >= t_begin) & (self._dates < t_cut)
            eval_mask = (self._dates >= t_cut) & (self._dates <= t_end)
            train_count = int(train_mask.sum())
            eval_count = int(eval_mask.sum())

            if train_count == 0 or eval_count == 0:
                continue

            ratio = eval_count / max(train_count, 1)
            lower = self.eval_ratio * (1 - self.ratio_tolerance)
            upper = self.eval_ratio * (1 + self.ratio_tolerance)
            if lower <= ratio <= upper:
                return t_begin, t_cut, t_end

        return None

    def _deterministic_window(
        self, train_target: int, eval_target: int
    ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """Construct a fallback window when random sampling cannot meet targets."""
        counts = self._counts
        unique_dates = self._unique_dates
        n_unique = unique_dates.size

        start_idx = 0
        total_target = train_target + eval_target
        while start_idx < n_unique - 1 and counts[start_idx:].sum() < total_target:
            start_idx += 1

        cut_idx = max(start_idx + 1, start_idx)
        train_rows = 0
        while cut_idx < n_unique - 1 and train_rows < train_target:
            train_rows += counts[cut_idx - 1]
            cut_idx += 1

        cut_idx = min(cut_idx, n_unique - 1)

        end_idx = cut_idx
        eval_rows = 0
        while end_idx < n_unique and eval_rows < eval_target:
            eval_rows += counts[end_idx]
            if eval_rows >= eval_target or end_idx == n_unique - 1:
                break
            end_idx += 1

        t_begin = pd.Timestamp(unique_dates[start_idx])
        t_cut = pd.Timestamp(unique_dates[min(cut_idx, n_unique - 1)])
        if t_cut <= t_begin and n_unique >= 2:
            t_cut = pd.Timestamp(unique_dates[min(start_idx + 1, n_unique - 1)])

        t_end = pd.Timestamp(unique_dates[min(end_idx, n_unique - 1)])
        return t_begin, t_cut, t_end

    def split(self, X=None, y=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield train/eval indices honoring the sampled date window."""
        rng = (
            np.random.RandomState(self._rng_state)
            if self._rng_state is not None
            else np.random.RandomState()
        )
        train_target, eval_target, total_target = self._targets

        for _ in range(self.n_splits):
            window = self._candidate_window(rng, train_target, eval_target, total_target)
            if window is None:
                window = self._deterministic_window(train_target, eval_target)

            t_begin, t_cut, t_end = window
            train_mask = (self._dates >= t_begin) & (self._dates < t_cut)
            eval_mask = (self._dates >= t_cut) & (self._dates <= t_end)

            train_idx = np.flatnonzero(train_mask.to_numpy())
            eval_idx = np.flatnonzero(eval_mask.to_numpy())

            if train_idx.size == 0 or eval_idx.size == 0:
                window = self._deterministic_window(train_target, eval_target)
                t_begin, t_cut, t_end = window
                train_mask = (self._dates >= t_begin) & (self._dates < t_cut)
                eval_mask = (self._dates >= t_cut) & (self._dates <= t_end)
                train_idx = np.flatnonzero(train_mask.to_numpy())
                eval_idx = np.flatnonzero(eval_mask.to_numpy())
                if train_idx.size == 0 or eval_idx.size == 0:
                    raise ValueError(
                        "Unable to construct a non-empty train/eval split from travel dates"
                    )

            yield train_idx, eval_idx


def cross_validate_with_eval(
    estimator, X: Any, y: Any, cv: Any, scoring: str
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
