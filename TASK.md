# Weekly Task — 2026-05-30

**Project:** Network Monitoring & Traffic Analysis Lab
**Type:** Code task
**Tool:** Cursor

## Task

Integrate Weights & Biases experiment tracking into the Network Monitoring classifier pipeline and run a benchmark sweep comparing the current Random Forest baseline against two alternative classifiers (e.g. Gradient Boosting / XGBoost and a tuned Random Forest with engineered features). The goal is to close the gap from the ~80% baseline toward 85%+ and produce documented evidence of the improvement attempt.

## Deliverable

A `FINDINGS.md` in the repo root with a before/after metrics table (accuracy, precision, recall, F1 per class), a W&B run link or exported artifact, and a one-paragraph interpretation of which model won and why.

## Context for Cursor

The pipeline lives in `src/` and produces clean ndjson output for the SOC Triage Agent. The current classifier is scikit-learn Random Forest with ~80% accuracy on benign vs. malicious classification from network flow features. Add W&B `wandb.init()` / `wandb.log()` calls around the training loop, then define a sweep config with 2–3 alternative model configs. Run the sweep, capture results, and write FINDINGS.md. The W&B run should be the source of truth for metrics; FINDINGS.md summarizes it for the portfolio README.

---

## Implementation Guide — How to Do This Like a Real Engineer

### Step 1 — Open a GitHub Issue

Before touching any code, open an issue on the repo. This creates a paper trail that makes the project look like it's under active, professional development.

**Title:** `feat: add W&B experiment tracking and classifier benchmark`

**Body template:**
```
## Problem
Current classifier (~80% accuracy) has no experiment tracking. There's no documented
baseline and no systematic way to compare models.

## Proposed change
- Integrate W&B for experiment tracking (replace or supplement MLflow)
- Run a sweep: RF baseline vs. XGBoost vs. tuned RF with additional features
- Document results in FINDINGS.md

## Success criteria
- W&B run logged with all three models
- FINDINGS.md written with before/after metrics table
- README updated with link to findings
```

Note the issue number (e.g. `#7`) — you'll reference it in the branch name and PR.

---

### Step 2 — Create a Feature Branch

From `main`, create a branch named after the issue:

```bash
git checkout -b feat/wandb-classifier-benchmark
```

Keep all work on this branch. Never commit the W&B API key — use a `.env` file and make sure `.env` is in `.gitignore`.

---

### Step 3 — Implement in Cursor

Open the branch in Cursor. The implementation has three parts:

**Part A — W&B integration**
- `pip install wandb` and add to `requirements.txt`
- At the top of the training script, add:
  ```python
  import wandb
  wandb.init(project="network-monitoring", name="rf-baseline", config={...})
  ```
- Log metrics after evaluation:
  ```python
  wandb.log({"accuracy": acc, "f1": f1, "precision": prec, "recall": rec})
  ```
- Log the trained model as a W&B artifact
- End with `wandb.finish()`

**Part B — Sweep config**
Create `sweep_config.yaml` (or define it inline):
```yaml
method: grid
metric:
  name: f1
  goal: maximize
parameters:
  model_type:
    values: ["random_forest_baseline", "xgboost", "random_forest_tuned"]
```

Run each model as a separate W&B run (not a hyperparameter sweep — just three named runs for clarity). This is cleaner to read and easier to present.

**Part C — FINDINGS.md**
Write this file after the runs complete. See format below.

---

### Step 4 — Run the Experiments

With Cursor doing the code work, run the three training scripts and confirm all three appear in your W&B project dashboard at `wandb.ai`. Each run should show:
- Config (model type, hyperparameters)
- Metrics (accuracy, F1, precision, recall — per class if you have multi-class)
- The trained model artifact

Screenshot or export the comparison table from the W&B UI — you'll embed it or link it in FINDINGS.md.

---

### Step 5 — Write FINDINGS.md

Place the file at the **repo root** (same level as README.md). This is standard for ML projects — it's where a hiring manager or recruiter will find it.

**Format:**

```markdown
# Classifier Benchmark — FINDINGS

_Experiment date: YYYY-MM-DD_
_W&B project: [link to your W&B run]_

## Objective

Evaluate three classifier configurations on binary network flow classification
(benign vs. malicious) and determine whether the ~80% RF baseline can be
improved to 85%+.

## Models Evaluated

| Model | Config notes |
|-------|-------------|
| Random Forest (baseline) | n_estimators=100, default params |
| XGBoost | learning_rate=0.1, max_depth=6, n_estimators=200 |
| Random Forest (tuned) | n_estimators=300, max_depth=15, min_samples_split=5 |

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| RF Baseline | 0.XX | 0.XX | 0.XX | 0.XX |
| XGBoost | 0.XX | 0.XX | 0.XX | 0.XX |
| RF Tuned | 0.XX | 0.XX | 0.XX | 0.XX |

_Full run details: [W&B link]_

## Interpretation

[2–3 sentences. Which model won, by how much, why you think it did, and
whether the 85% target was hit. If it wasn't, what the next logical step is.]

## Next Steps

- [ ] Wire winning model into the SOC Triage Agent pipeline
- [ ] Add Zeek + Kafka live inference (see README roadmap)
```

---

### Step 6 — Update the README

In the repo's README.md, add a one-line reference under the relevant section:

```markdown
**Benchmark results:** See [FINDINGS.md](./FINDINGS.md) — W&B experiment tracking,
three classifiers evaluated, [X]% best F1.
```

This makes the work discoverable without burying the README in metrics.

---

### Step 7 — Commit, Push, and Open a Pull Request

Commit in logical chunks (not one giant commit):
```
git add requirements.txt src/train.py
git commit -m "feat: add W&B integration to training pipeline"

git add sweep_config.yaml
git commit -m "feat: add three-model sweep configuration"

git add FINDINGS.md README.md
git commit -m "docs: add classifier benchmark findings, update README"
```

Push the branch:
```bash
git push origin feat/wandb-classifier-benchmark
```

Open a PR on GitHub. **Title:** `feat: W&B experiment tracking + classifier benchmark (#7)` — referencing the issue number closes it automatically on merge.

**PR description:**
```
Closes #7

## What this does
- Adds W&B experiment tracking to the classifier training loop
- Benchmarks three classifiers (RF baseline, XGBoost, tuned RF)
- Documents results in FINDINGS.md

## Results summary
[paste the one-line result, e.g. "XGBoost reached 87% F1, up from 80% RF baseline"]

## W&B run
[link]
```

---

### Step 8 — Review and Merge

Even though you're the only contributor, review the PR yourself before merging — scroll through the diff, check that no API keys or `.env` files snuck in, confirm FINDINGS.md looks clean. Then merge into `main` and delete the branch.

The closed issue + merged PR on your public GitHub repo is what makes this look like real engineering practice, not just a folder of scripts.
