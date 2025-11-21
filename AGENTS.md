# Repository Guidance for `bid-predictor`

## Purpose and Top-Level Layout
- The repo provides utilities to train and tune a CatBoost-based bid prediction pipeline.
- Top-level entry points:
  - `train.py`: fits a single model using a feature configuration YAML and logs to MLflow.
  - `tune_catboost.py`: runs Bayesian hyperparameter optimization over CatBoost options and feature-transform overrides.
  - `preprocessing/` & `containers/`: deployment/packaging helpers (rarely touched for tuning tasks).
  - `search_configs/`: canned YAML grids used by the tuner.
  - `docs/`: rendered HTML docs (e.g., Bayesian tuning overview).

## Core Python Package (`bid_predictor/`)
- `bid_predictor.py`: builds the sklearn pipeline (feature prep + CatBoost model).
- `feature_config.py`: loads YAML feature configs and normalizes metadata (ensures group-by keys, etc.).
- `preprocessor.py` & `transform.py`: feature engineering utilities used by the pipeline.
- `tracking.py`: MLflow helpers shared by training and tuning flows.
- `tuning/`: orchestration of tuning (search space construction, CV, MLflow logging, result writers).
  - `data_access.py`: resolves training parquet path and loads data with optional testing split.
  - `cross_validation.py`: wraps stratified CV evaluation and scoring aggregation.
  - `search_config.py`: parses YAML/JSON search configs into structured dictionaries.
  - `search_grid.py`: converts search configs into skopt `Dimension`s, including wrappers for non-hashable values.
  - `feature_tuning.py`: splits/merges flattened tuning combos and rebuilds feature configs.
  - `mlflow_logging.py`: context manager to set up/tear down MLflow runs, even if MLflow is missing.
  - `result_writing.py`: CSV/JSON/YAML writers for tuning summaries (normalize numpy types first).

## Dash UI Helpers
- `dash_app.py` at the repository root should focus on layout and callbacks. Move reusable logic into the `bid_predictor/ui/` package.
- Each helper module in `ui/` should stay small and purpose-driven (e.g., data access, formatting, plotting).
- Add or update unit tests under `ui_tests/` whenever changing the UI helpers.

## Typical Tuning Flow (`tune_catboost.py`)
1. Parse CLI arguments (data path, feature config, search config, CV, MLflow, artifact outputs).
2. Load base feature metadata via `feature_config.load_feature_config` (package default or user-specified YAML).
3. Read search config (YAML/JSON) using `tuning.search_config.load_search_config`; fall back to metadata-derived defaults if omitted.
4. Convert search definitions to skopt `Optimizer` dimensions (`tuning.search_grid.build_search_space`).
5. Iterate suggestions:
   - Merge CatBoost params & transform overrides (`feature_tuning.split_combination`).
   - Clone base feature metadata, apply overrides, rebuild feature config dict.
   - Build sklearn pipeline via `bid_predictor.build_pipeline` and evaluate with `cross_validate_with_eval`.
   - Track metrics/params in MLflow (if available) using `tuning.mlflow_logging.mlflow_run`.
   - Accumulate results and optionally emit CSV/JSON/YAML artifacts through `tuning.result_writing` helpers.
6. Display top configurations and export best settings when requested.

### Objective & Scoring
- Objective: maximize the specified sklearn scoring metric (defaults to ROC AUC averaged across folds).
- Cross-validation uses `StratifiedKFold` with a deterministic random state; the target column is inferred inside `cross_validate_with_eval`.
- Scores and standard deviations are logged/stored for ranking.

## Search Configuration Files
- YAML structure:
  ```yaml
  catboost:
    iterations: [200, 400]
    depth: [6, 8]
    learning_rate: [0.05, 0.1]
  transform:
    outlier:
      item_count:
        - {max: 5}
        - {max: 8}
    impute_median:
      seats_available: [true, false]
  ```
- `catboost` keys map directly to CatBoost parameters.
- `transform` sections (`impute_value`, `impute_median`, `outlier`, `bins`) override feature metadata. Non-hashable values are wrapped during search so they can live inside skopt categorical dimensions.
- Files live in `search_configs/`; the tuner accepts either YAML or JSON.

## MLflow Integration
- Use `--mlflow-experiment` to set or override the experiment name.
- `mlflow_logging.mlflow_run` handles optional MLflow availability: if MLflow is missing, the context becomes a no-op (safe for tests).
- Parameters logged to MLflow are normalized via `normalize_search_value`/`stringify_param_value` to avoid numpy serialization issues.

## Result Artifacts
- `--output-csv`: writes per-iteration metrics/params via `write_results_csv`.
- `--results-json`: persists the best run summary (`write_best_result_json`).
- `--best-config-out`: dumps the reconstructed feature configuration for the best score (`write_best_feature_config`).
- Writers sanitize values (NumPy scalars, nested dicts) before serialization.

## Testing Strategy
- Unit tests live under `tests/unit/`, integration tests under `tests/integration/`.
- Run the full suite with `pytest -q` from the repo root.
- The `--testing` CLI flag mirrors the integration suite's small evaluation split; use it when running scripts in CI-like contexts.

## Contribution Tips
- Prefer modifying helper modules inside `bid_predictor/tuning/` when adding tuning-related functionality; keep `tune_catboost.py` focused on orchestration.
- Maintain deterministic behavior: keep random seeds plumbed from CLI into CatBoost and CV splitters.
- Normalize any user-facing or serialized outputs (NumPy -> Python types) to prevent downstream errors.
- Update or add unit tests when touching tuning utilities; leverage existing fixtures in `tests/unit/test_tuning_modules.py`.

