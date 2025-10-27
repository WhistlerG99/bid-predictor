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

## Using the Dash bid prediction playground

An interactive Dash UI ships with the package to help explore model
predictions for individual flights and bids. After installing the optional
dependencies from `requirements.txt`, launch the app with:

```bash
python dash_app.py
```

Open http://127.0.0.1:8050/ in your browser. The interface guides you through
the following steps:

1. **Load a dataset snapshot.** Enter the path to the parquet snapshots file
   (the default resolves to the same location used by the training scripts) and
   click **Load dataset**. This populates the flight dropdown with all available
   flight keys.
2. **Load an MLflow model.** Provide the tracking URI for the MLflow server, the
   registered model name, and the stage or version to score. Press **Load
   model** to cache the model locally.
3. **Select and edit bids.** Choose a flight from the dropdown to display every
   snapshot for that itinerary. You can tweak feature values directly in the
   table to simulate alternative scenarios.
4. **Review predictions.** When a model is loaded, the table of bids updates
   with acceptance probabilities and the accompanying chart mirrors the
   `log_prob_examples` visualization from `bid_predictor/tracking.py`, including
   the seats-available trend when present.

Any changes you make in the bid table automatically refresh both the prediction
grid and the plot so you can iterate quickly on counterfactual inputs.

## Running the tests

The project uses `pytest` for both unit and integration coverage. After
installing the package in a virtual environment, install the test dependencies
from `requirements.txt` (they include the stubs used by the suites) and then run
pytest from the repository root:

```bash
pip install -r requirements.txt
pytest
```

To run only the integration scenarios—for example, to verify the deterministic
outputs of `train.py` and `tune_catboost.py`—use the dedicated marker:

```bash
pytest -m integration
```
