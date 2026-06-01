# Experiment: Classifier benchmark (June 2026)

| Field | Value |
|-------|--------|
| **Experiment ID** | `2026-06-classifier-benchmark` |
| **Date** | 2026-06-01 |
| **Dataset** | UNSW-NB15-style flow CSV (`data/01_Raw/rawdata.csv`) |
| **Task** | Binary classification — benign (0) vs malicious (1) |
| **Split** | 80/20 stratified, `random_state=42` |
| **W&B project** | [models-ie-university / network-monitoring](https://wandb.ai/models-ie-university/network-monitoring) |
| **Artifacts** | `reports/<model>/metrics.json`, `models/<model>.joblib` |

## Objective

Compare three registered classifiers against the historical ~80% accuracy RF baseline and check whether a simple model swap or tuning could reach **85%+** held-out performance without new features.

## Models evaluated

| Registry key | Description |
|--------------|-------------|
| `random_forest_baseline` | `RandomForestClassifier(n_estimators=100)`, defaults otherwise |
| `xgboost` | `XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=200)` |
| `random_forest_tuned` | `n_estimators=300`, `max_depth=15`, `min_samples_split=5` |

All models share the same pipeline: validate → preprocess (one-hot `proto` / `service` / `state`) → scale numerics → train → evaluate on the shared test split in `data/03_Enriched/`.

## Results (held-out test set)

Metrics below come from `reports/<model>/metrics.json` after `evaluate` (source of truth for this write-up). ROC-AUC uses `predict_proba` on the same 51,535-row test split for every model.

| Model | Accuracy | ROC-AUC | Precision (malicious) | Recall (malicious) | F1 (malicious) | Train time (s) |
|-------|----------|---------|------------------------|--------------------|----------------|----------------|
| **RF baseline** | **0.9511** | **0.9922** | 0.9630 | 0.9603 | **0.9617** | ~8.3 |
| XGBoost | 0.9468 | 0.9915 | 0.9650 | 0.9514 | 0.9581 | **~5.6** |
| RF tuned | 0.9422 | 0.9900 | 0.9521 | 0.9578 | 0.9550 | ~24.5 |

**Benign class (F1):** baseline 0.9324 · XGBoost 0.9273 · tuned 0.9195.

**Confusion matrix (baseline)** — TN/FP/FN/TP:

```
[[17386, 1214],
 [1307, 31628]]
```

### Metric naming (train vs evaluate)

| Logged name | When | Meaning |
|-------------|------|---------|
| `test_roc_auc` | Train | ROC-AUC on the held-out test split immediately after fitting |
| `roc_auc` | Evaluate | Same split, same metric — should match `test_roc_auc` closely |

On this run, **baseline has the highest `test_roc_auc` and `roc_auc`** (~0.9922), ahead of XGBoost (~0.9915) and tuned RF (~0.9900). Differences are under 0.3 percentage points on accuracy — models are effectively tied on ranking quality.

### Training time

The only large practical gap is **wall-clock training**:

- **XGBoost** — fastest (~5.6 s)
- **RF baseline** — middle (~8.3 s)
- **RF tuned** — slowest (~24.5 s), more trees and capped depth without accuracy gain

## Conclusion

1. **85% target:** All three models land near **~95% accuracy** and **~0.99 ROC-AUC**, well above the 85% goal.
2. **Winner on test metrics:** **`random_forest_baseline`** — best accuracy and ROC-AUC on this split.
3. **Runner-up for speed:** **XGBoost** — nearly the same scores, ~35% faster training than baseline.
4. **Tuned RF:** Did not beat baseline here; extra trees/depth added cost without payoff.

**Production choice:** `random_forest_baseline` is set as `model.active` in `src/config/config.yaml` so infer/live and default evaluate use `models/random_forest_baseline.joblib` unless you pass `--model` on the CLI.

## Weights & Biases

Runs are logged under entity `models-ie-university`, project `network-monitoring`. Each model typically produces one run (train metrics, then evaluate resumes the same run with `accuracy` / `roc_auc`).

Use the project workspace for bar-chart comparison of `roc_auc` and `accuracy` across runs. Ignore stale auto-panels named `train/*` from earlier logging shapes — custom charts for `train_time_sec` and `test_roc_auc` are clearer.

## Reproduce

```bash
pip install -r requirements.txt
wandb login   # optional; config has wandb.enabled: true

python -m src.main train --model random_forest_baseline
python -m src.main evaluate --model random_forest_baseline
```

Inference with the production model (no `--model` flag needed):

```bash
python -m src.main infer --input path/to/flows.csv
```
