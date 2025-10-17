#!/usr/bin/env python3
import os, sys, io

APPEND_BP = '''

# --- AUTO-GENERATED (yaml features) ---
try:
    import os as _os
    if _os.getenv("USE_YAML_FEATURES", "1") == "1":
        from bid_predictor.feature_config import load_feature_spec as _lfs
        _spec = _lfs(_os.getenv("FEATURES_YAML"))
        if 'features' in globals():
            globals()['features'] = _spec.features()
        if 'FEATURES' in globals():
            globals()['FEATURES'] = _spec.features()
        if 'pre_features' in globals():
            globals()['pre_features'] = _spec.pre()
        if 'PRE_FEATURES' in globals():
            globals()['PRE_FEATURES'] = _spec.pre()
        if 'cat_features' in globals():
            globals()['cat_features'] = _spec.cat()
        if 'CAT_FEATURES' in globals():
            globals()['CAT_FEATURES'] = _spec.cat()
except Exception as _e:
    print(f"[feature-config] warning: {_e}")
'''

APPEND_TRAIN = '''

# --- AUTO-GENERATED (yaml features helpers) ---
try:
    import os as _os
    from bid_predictor.feature_config import load_feature_spec as _lfs, validate_against_df as _vdf
    _spec = _lfs(_os.getenv("FEATURES_YAML"))
    if 'df' in globals():
        _vdf(_spec, list(df.columns))
        if 'X' not in globals():
            X = df[_spec.features()]
    if 'cat_indices' not in globals() and 'X' in globals():
        cat_indices = [i for i, c in enumerate(X.columns) if c in set(_spec.cat())]
except Exception as _e:
    print(f"[feature-config] warning: {_e}")
'''

FILES = [
    ("bid_predictor/bid_predictor.py", APPEND_BP),
    ("train.py", APPEND_TRAIN),
]

for path, snippet in FILES:
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        continue
    with open(path, 'a', encoding='utf-8') as f:
        f.write(snippet)
        print(f"[ok] appended to {path}")
