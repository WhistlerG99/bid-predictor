"""Utilities for constructing hyperparameter search grids.

This module centralizes logic for expanding search configurations into the
`sklearn.model_selection.ParameterGrid` compatible dictionaries that power the
CatBoost tuning script. Keeping the functionality in a standalone module makes
it easier to reuse in future experiments or command-line tools.
"""
from __future__ import annotations

import math
import numbers
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from skopt.space import Categorical, Dimension, Integer, Real


def _ensure_list(values: Any) -> List[Any]:
    """Return *values* as a list, preserving simple values as singletons."""

    if isinstance(values, (list, tuple)):
        return list(values)
    if hasattr(values, "__array__"):
        # numpy.ndarray implements the array protocol but behaves like a list here
        return list(values)  # type: ignore[arg-type]
    return [values]


def build_parameter_grid(search_cfg: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Construct a flattened parameter grid from the search configuration.

    Parameters
    ----------
    search_cfg:
        Mapping describing CatBoost and feature transformation overrides. The
        structure matches the configuration files consumed by ``tune_catboost``.

    Returns
    -------
    Dict[str, List[Any]]
        Dictionary suitable for instantiating ``ParameterGrid`` where keys are
        flattened pipeline parameter names and values are candidate settings.
    """

    grid: Dict[str, List[Any]] = {}

    for param, values in search_cfg.get("catboost", {}).items():
        values_list = _ensure_list(values)
        if not values_list:
            continue
        grid[f"catboost__{param}"] = values_list

    transform_cfg: Mapping[str, Mapping[str, Iterable[Any]]] = search_cfg.get(
        "transform", {}
    )
    for section in ("impute_value", "impute_median", "outlier", "bins"):
        section_cfg = transform_cfg.get(section, {}) or {}
        for feature, values in section_cfg.items():
            values_list = _ensure_list(values)
            if not values_list:
                continue
            grid[f"transform__{section}__{feature}"] = values_list

    if not grid:
        grid["__noop__"] = [None]

    return grid


def _values_are_bools(values: List[Any]) -> bool:
    return all(isinstance(value, bool) for value in values)


def _values_are_ints(values: List[Any]) -> bool:
    return all(isinstance(value, numbers.Integral) and not isinstance(value, bool) for value in values)


def _values_are_reals(values: List[Any]) -> bool:
    return all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in values)


def _make_dimension(values: List[Any]) -> Dimension:
    if not values:
        raise ValueError("Values must be non-empty to create a search dimension")

    if _values_are_bools(values):
        return Categorical(values)

    if _values_are_ints(values):
        min_val = int(min(values))
        max_val = int(max(values))
        if min_val == max_val:
            return Categorical([min_val])
        return Integer(min_val, max_val)

    if _values_are_reals(values):
        min_val = float(min(values))
        max_val = float(max(values))
        if math.isclose(min_val, max_val):
            return Categorical([float(min_val)])
        return Real(min_val, max_val)

    if len(values) == 1:
        return Categorical(values)

    return Categorical(values)


def build_search_space(search_cfg: Mapping[str, Any]) -> List[Tuple[str, Dimension]]:
    """Create Bayesian optimization dimensions from the search configuration."""

    dimensions: List[Tuple[str, Dimension]] = []

    for param, values in search_cfg.get("catboost", {}).items():
        values_list = _ensure_list(values)
        if not values_list:
            continue
        dimensions.append((f"catboost__{param}", _make_dimension(values_list)))

    transform_cfg: Mapping[str, Mapping[str, Iterable[Any]]] = search_cfg.get(
        "transform", {}
    )
    for section in ("impute_value", "impute_median", "outlier", "bins"):
        section_cfg = transform_cfg.get(section, {}) or {}
        for feature, values in section_cfg.items():
            values_list = _ensure_list(values)
            if not values_list:
                continue
            dimensions.append((f"transform__{section}__{feature}", _make_dimension(values_list)))

    if not dimensions:
        dimensions.append(("__noop__", Categorical([None])))

    return dimensions

