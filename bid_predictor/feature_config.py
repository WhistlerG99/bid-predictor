from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

# default path: repo_root/config/features.yaml
_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config", "features.yaml")

@dataclass
class FeatureSpec:
    all_features: List[str]
    pre_features: List[str]
    cat_features: List[str]
    active_features: List[str]

    def features(self) -> List[str]:
        keep = set(self.active_features)
        return [f for f in self.all_features if f in keep]

    def pre(self) -> List[str]:
        keep = set(self.active_features)
        return [f for f in self.pre_features if f in keep]

    def cat(self) -> List[str]:
        keep = set(self.active_features)
        return [f for f in self.cat_features if f in keep]


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_feature_spec(path: str | None = None) -> FeatureSpec:
    y = _read_yaml(path or _DEFAULT_PATH)
    missing = [k for k in ("all_features", "pre_features", "cat_features", "active_features") if k not in y]
    if missing:
        raise ValueError(f"features.yaml is missing keys: {missing}")

    def _uniq(xs):
        seen = set(); out=[]
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return FeatureSpec(
        all_features=_uniq(y["all_features"]),
        pre_features=_uniq(y["pre_features"]),
        cat_features=_uniq(y["cat_features"]),
        active_features=_uniq(y["active_features"]),
    )


def validate_against_df(spec: FeatureSpec, df_columns: List[str]) -> None:
    cols = set(df_columns)
    requested = set(spec.features())
    missing = sorted(requested - cols)
    if missing:
        raise ValueError(f"Active features not in dataframe: {missing}")
