# bid-predictor

Small utilities for training and tuning the CatBoost bid prediction pipeline.

## Installation

Create a virtual environment with Python 3.9+ and install the package (editable
mode is convenient while iterating):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The training and tuning scripts rely on the optional dependencies listed in
`requirements.txt`. Install them if you plan to run the CLI utilities directly:

```bash
pip install -r requirements.txt
```

## Data expectations

Both `train.py` and `tune_catboost.py` expect a parquet dataset containing the
bid snapshots. By default the scripts look for the file at
`./data/air_canada_and_lot/bid_data_snapshots_v2.parquet`. The path can be
overridden with the `--train-file` flag (for the tuner) or by setting the
`SM_CHANNEL_TRAIN` environment variable in SageMaker environments. Columns such
as `carrier_code`, `flight_number`, and `fare_class` should be present and will
be cast to categorical dtypes automatically.

## Running `train.py`

The training script fits a single CatBoost model and logs metrics to MLflow.
Typical usage on a local workstation looks like:

```bash
python train.py \
  --iterations 400 \
  --depth 8 \
  --learning-rate 0.08 \
  --feature-config path/to/feature_config.yaml \
  --experiment-name "catboost-local"
```

Key options:

- `--task-type` / `--devices`: set to `GPU` and specify device IDs to leverage
  GPU acceleration.
- `--feature-config`: optional path to a custom feature configuration YAML.
- `--testing`: use the small evaluation window from August 2023 to reproduce the
  lightweight test split used in CI.

The script automatically handles MLflow tracking configuration when running
inside SageMaker by reading the `MLFLOW_AWS_ARN` environment variable.

## Running `tune_catboost.py`

`tune_catboost.py` performs cross-validated grid search over CatBoost
hyperparameters and feature transformation toggles. You can run it with the
built-in defaults:

```bash
python tune_catboost.py --feature-config path/to/feature_config.yaml
```

Provide a custom search space by passing a YAML or JSON file via
`--search-config`. Several ready-to-use examples live in `search_configs/`:

- `baseline.yaml`: lightweight sweep over depth, learning rate, and basic
  seat-availability imputation and `item_count` caps.
- `regularization_sweep.yaml`: emphasizes regularization knobs such as
  `l2_leaf_reg`, subsampling, and bagging temperature alongside alternative
  `usd_base_amount` binning and `num_offers` limits.
- `feature_transform.yaml`: coordinates CatBoost hyperparameters with
  `seats_available` imputation, plus `item_count`/`num_offers` outlier settings
  and different bin widths for `usd_base_amount`.
- `gpu_quickstart.yaml`: slim GPU-friendly grid that experiments with
  `seats_available` binning for rapid graphics-hardware iterations.
- `wide_grid.yaml`: broader search including class weighting, border counts, and
  several feature transformation combinations for `seats_available`,
  `item_count`, and `num_offers`.

Example invocation using one of the presets and exporting the results to CSV:

```bash
python tune_catboost.py \
  --feature-config path/to/feature_config.yaml \
  --search-config search_configs/regularization_sweep.yaml \
  --cv-splits 5 \
  --output-csv tuning_results.csv \
  --best-config-out best_features.yaml
```

The tuner respects additional options such as `--task-type`, `--devices`, and
`--testing`, mirroring the behaviour of `train.py`. Results can also be tracked
with MLflow if the service is configured in the environment.
