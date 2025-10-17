# Bid Predictor

## YAML-driven feature selection
This branch introduces a single source of truth for feature subsets: `config/features.yaml`.

### Edit features
Change only `active_features` to experiment with different subsets. The other lists remain the full catalogs.

```yaml
all_features:   # full catalog
  - f_clicks
  - f_impressions
  - f_ctr
  - f_bid
  - f_region
  - f_device

pre_features:   # go through sklearn ColumnTransformer
  - f_clicks
  - f_impressions
  - f_ctr
  - f_region
  - f_device

cat_features:   # categorical names for CatBoost
  - f_region
  - f_device

active_features:  # 👉 edit me only
  - f_clicks
  - f_impressions
  - f_ctr
  - f_region
  - f_device
```

### Use in code
```python
from bid_predictor.feature_config import load_feature_spec, validate_against_df

spec = load_feature_spec()          # reads config/features.yaml
FEATURES = spec.features()
PRE_FEATURES = spec.pre()
CAT_FEATURES = spec.cat()

validate_against_df(spec, list(df.columns))
X = df[FEATURES]
cat_indices = [i for i, c in enumerate(X.columns) if c in set(CAT_FEATURES)]
model.fit(X, y, cat_features=cat_indices)
```

### Alternate configs per run
Set `FEATURES_YAML=/path/to/other.yaml` and modify `load_feature_spec(os.getenv("FEATURES_YAML"))` if desired.
