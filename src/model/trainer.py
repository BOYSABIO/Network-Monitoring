# Generic Model Trainer
# ---------------------
# Model-agnostic training pipeline that works with any model registered
# in the registry. Handles: train/test split, scaling, GridSearchCV
# hyperparameter tuning, and saving the full artifact for later inference.
#
# The key idea: the trainer doesn't know or care whether it's training
# logistic regression or random forest. It gets the model and param_grid
# from the registry, and everything else is the same.

import logging
import os
from contextlib import redirect_stdout
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.model.registry import get_model
from src.config.loader import get_config


class _LogWriter:
    """File-like stream that forwards sklearn verbose output to logger."""

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ''

    def write(self, message):
        if not message:
            return
        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        line = self._buffer.strip()
        if line:
            self.logger.log(self.level, line)
        self._buffer = ''


def train(df, model_name=None, mode='prod'):
    """
    Train a model end-to-end: split, scale, tune, save.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with features and 'label' column.
    model_name : str, optional
        Registry key (e.g. 'logistic_regression'). If None, reads
        from config.yaml model.active.
    mode : str, optional
        Training profile: 'dev' (faster search) or 'prod' (full search).

    Returns
    -------
    dict
        The saved artifact dictionary containing the model, scaler,
        and feature metadata needed for inference.
    """
    config = get_config()
    model_name = model_name or config['model']['active']
    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    if mode not in {'dev', 'prod'}:
        raise ValueError(f"Unsupported training mode '{mode}'. Use 'dev' or 'prod'.")
    cv_folds = 2 if mode == 'dev' else config['model']['cv_folds']
    scoring = config['model']['scoring']
    numeric_features = config['features']['numeric']

    logging.info(f"Starting training pipeline for model: {model_name} (mode={mode})")

    # ------------------------------------------------------------------
    # 1. Split into train and test sets
    # ------------------------------------------------------------------
    # Stratify on the label so both sets have the same malicious/benign ratio.
    # This matters in security data where class imbalance is common.
    target = config['features']['target']
    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserve class balance in both splits
    )
    logging.info(
        f"Train/test split: {X_train.shape[0]} train, "
        f"{X_test.shape[0]} test, {X_train.shape[1]} features"
    )

    # Save split data for reproducibility
    enriched_dir = config['paths']['enriched']
    os.makedirs(enriched_dir, exist_ok=True)
    X_train.to_csv(os.path.join(enriched_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(enriched_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(enriched_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(enriched_dir, 'y_test.csv'), index=False)
    logging.info(f"Split data saved to {enriched_dir}")

    # ------------------------------------------------------------------
    # 2. Scale numeric features
    # ------------------------------------------------------------------
    # Identify which columns in the encoded DataFrame are numeric
    # (some original numeric features remain, categorical ones are now dummies)
    numeric_cols = [c for c in X_train.columns if c in numeric_features]
    categorical_cols = [c for c in X_train.columns if c not in numeric_features]

    # fit ONLY on train to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled_num = scaler.transform(X_test[numeric_cols])

    # Recombine: scaled numerics + untouched dummy columns
    X_train_scaled = np.hstack([X_train_scaled_num, X_train[categorical_cols].values])
    X_test_scaled = np.hstack([X_test_scaled_num, X_test[categorical_cols].values])

    logging.info(f"Scaled {len(numeric_cols)} numeric features, kept {len(categorical_cols)} dummy features")

    # ------------------------------------------------------------------
    # 3. Get model and hyperparameter grid from registry
    # ------------------------------------------------------------------
    model, param_grid = get_model(model_name, mode=mode)
    total_combos = 1
    for vals in param_grid.values():
        total_combos *= len(vals)
    total_fits = total_combos * cv_folds

    logging.info(
        f"GridSearchCV: {total_combos} combos x {cv_folds} folds = "
        f"{total_fits} fits, scoring={scoring}"
    )
    logging.info(
        f"Training on {X_train_scaled.shape[0]} samples with "
        f"{X_train_scaled.shape[1]} features — this may take several minutes"
    )
    logging.info("Preprocessing complete. Starting hyperparameter search next.")

    # ------------------------------------------------------------------
    # 4. Hyperparameter tuning via cross-validation
    # ------------------------------------------------------------------
    # GridSearchCV tries every combination in param_grid and picks the
    # one with the best cross-validated score on the TRAINING set only.
    #
    # verbose=3 prints a line per fit so you can track progress.
    # n_jobs=1 (sequential) ensures verbose output actually prints to terminal.
    # Parallel mode (n_jobs=-1) is faster but swallows verbose output due to
    # buffering in child processes, making it look like the training hangs.
    import time
    start_time = time.time()
    start_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"GridSearchCV started at {start_ts}")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_folds,
        verbose=3,
        n_jobs=1,
        return_train_score=True
    )

    stdout_logger = _LogWriter(logging.getLogger(), logging.INFO)
    with redirect_stdout(stdout_logger):
        grid_search.fit(X_train_scaled, y_train)
    stdout_logger.flush()

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    end_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_model = grid_search.best_estimator_
    best_idx = grid_search.best_index_
    best_train_cv = float(grid_search.cv_results_['mean_train_score'][best_idx])
    best_val_cv = float(grid_search.cv_results_['mean_test_score'][best_idx])
    logging.info(f"GridSearchCV completed in {minutes}m {seconds}s")
    logging.info(f"GridSearchCV finished at {end_ts}")
    logging.info(f"Best CV {scoring}: {grid_search.best_score_:.4f}")
    logging.info(f"Best params: {grid_search.best_params_}")
    logging.info(f"Best CV train score (training-fold mean): {best_train_cv:.4f}")
    logging.info(f"Best CV validation score (held-out fold mean): {best_val_cv:.4f}")

    # Save grid search results for analysis
    reports_dir = os.path.join(config['paths']['reports'], model_name)
    os.makedirs(reports_dir, exist_ok=True)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(os.path.join(reports_dir, 'grid_search_results.csv'), index=False)

    # ------------------------------------------------------------------
    # 5. Overfitting diagnostics (CSV)
    # ------------------------------------------------------------------
    # Compare train-vs-validation-vs-test to detect overfitting patterns.
    y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    test_roc_auc = float(roc_auc_score(y_test, y_test_prob))
    train_minus_val = best_train_cv - best_val_cv
    val_minus_test = best_val_cv - test_roc_auc
    train_minus_test = best_train_cv - test_roc_auc

    logging.info("Overfitting diagnostics context:")
    logging.info("- CV train/validation scores come from GridSearchCV during training")
    logging.info("- Test ROC-AUC below is computed on held-out test split during training")
    logging.info(f"Held-out test ROC-AUC: {test_roc_auc:.4f}")
    logging.info(f"Generalization gap (train_cv - val_cv): {train_minus_val:.4f}")
    logging.info(f"Generalization gap (val_cv - test): {val_minus_test:.4f}")
    logging.info(f"Generalization gap (train_cv - test): {train_minus_test:.4f}")

    diagnostics_df = pd.DataFrame([{
        'model_name': model_name,
        'mode': mode,
        'scoring': scoring,
        'cv_folds': cv_folds,
        'best_train_cv_score': best_train_cv,
        'best_validation_cv_score': best_val_cv,
        'test_roc_auc': test_roc_auc,
        'train_minus_validation': train_minus_val,
        'validation_minus_test': val_minus_test,
        'train_minus_test': train_minus_test,
        'best_params': str(grid_search.best_params_),
    }])
    diagnostics_path = os.path.join(reports_dir, 'overfitting_diagnostics.csv')
    diagnostics_df.to_csv(diagnostics_path, index=False)
    logging.info(f"Overfitting diagnostics saved to {diagnostics_path}")

    # ------------------------------------------------------------------
    # 6. Save the full artifact
    # ------------------------------------------------------------------
    # Everything needed to reproduce predictions on new data:
    # the trained model, the fitted scaler, and the exact feature order.
    artifact = {
        'model': best_model,
        'scaler': scaler,
        'model_name': model_name,
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols,
        'feature_order': list(X_train.columns),
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'mode': mode,
    }

    models_dir = config['paths']['models']
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, f'{model_name}.joblib')
    joblib.dump(artifact, artifact_path)
    logging.info(f"Artifact saved to {artifact_path}")

    return artifact
