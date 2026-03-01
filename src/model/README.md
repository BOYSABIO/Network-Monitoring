# Model Module

This directory uses a single training architecture:

- `registry.py` defines **what** can be trained.
- `trainer.py` defines **how** training runs end-to-end.

Legacy standalone scripts (`train_logreg.py`, `train_rf.py`) were removed to
avoid duplicate training paths and inconsistent behavior.

## Responsibilities

### `registry.py`
- Registers models by name (e.g., `logistic_regression`, `random_forest`).
- Returns `(estimator, param_grid)` for each model.
- Supports mode-specific search spaces (`dev` for fast iteration, `prod` for fuller search).

### `trainer.py`
- Splits data into train/test.
- Scales numeric features (fit on train only).
- Runs `GridSearchCV` with the model/grid from `registry.py`.
- Saves:
  - model artifact (`models/<model>.joblib`)
  - grid search results (`reports/<model>/grid_search_results.csv`)
  - overfitting diagnostics (`reports/<model>/overfitting_diagnostics.csv`)

Overfitting diagnostics are generated during `train`, after GridSearchCV selects
the best model and the trainer computes held-out test ROC-AUC.

## How to add a new model

1. Open `registry.py`.
2. Add a builder function that accepts `mode` and returns:
   - an unfitted sklearn estimator
   - a `param_grid` dictionary
3. Decorate it with `@register("your_model_name")`.
4. Train through the CLI:
   - `python -m src.main train --model your_model_name --mode dev`
   - `python -m src.main train --model your_model_name --mode prod`

## Recommended usage

- Always train via `src.main` so logging, preprocessing, and outputs stay consistent.
- Use `--mode dev` while iterating and `--mode prod` for final model selection.
