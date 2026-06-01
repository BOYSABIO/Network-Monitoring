"""
# Generic Model Trainer
# ---------------------
# Model-agnostic training pipeline that works with any model registered
# in the registry. Handles: train/test split, scaling, GridSearchCV
# hyperparameter tuning, and saving the full artifact for later inference.
#
# The key idea: the trainer doesn't know or care whether it's training
# logistic regression or random forest. It gets the model and param_grid
# from the registry, and everything else is the same.
"""

import time
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.model.registry import get_model
from src.config.loader import get_config
from src.utils.wandb_tracker import log_training_run


def train(df, model_name=None):
    """
    Train a model end-to-end: split, scale, tune, save.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with features and 'label' column.
    model_name : str, optional
        Registry key (e.g. 'logistic_regression'). If None, reads
        from config.yaml model.active.

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
    cv_folds = config['model']['cv_folds']
    scoring = config['model']['scoring']
    numeric_features = config['features']['numeric']

    logging.info("Starting training pipeline for model: %s", model_name)

    # ------------------------------------------------------------------
    # 1. Split into train and test sets
    # ------------------------------------------------------------------
    # Stratify on the label so both sets have the same malicious/benign ratio.
    # This matters in security data where class imbalance is common.
    target = config['features']['target']
    x = df.drop(columns=target)
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserve class balance in both splits
    )
    logging.info(
        "Train/test split: %d train, %d test, %d features",
        x_train.shape[0],
        x_test.shape[0],
        x_train.shape[1],
    )

    # Save split data for reproducibility
    enriched_dir = config['paths']['enriched']
    os.makedirs(enriched_dir, exist_ok=True)
    x_train.to_csv(os.path.join(enriched_dir, 'X_train.csv'), index=False)
    x_test.to_csv(os.path.join(enriched_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(enriched_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(enriched_dir, 'y_test.csv'), index=False)
    logging.info("Split data saved to %s", enriched_dir)

    # ------------------------------------------------------------------
    # 2. Scale numeric features
    # ------------------------------------------------------------------
    # Identify which columns in the encoded DataFrame are numeric
    # (some original numeric features remain, categorical ones are now dummies)
    numeric_cols = [c for c in x_train.columns if c in numeric_features]
    categorical_cols = [
        c for c in x_train.columns if c not in numeric_features
        ]

    # fit ONLY on train to prevent data leakage
    scaler = StandardScaler()
    x_train_scaled_num = scaler.fit_transform(x_train[numeric_cols])
    x_test_scaled_num = scaler.transform(x_test[numeric_cols])

    # Recombine: scaled numerics + untouched dummy columns
    x_train_scaled = np.hstack([x_train_scaled_num, x_train[categorical_cols].values])
    x_test_scaled = np.hstack([x_test_scaled_num, x_test[categorical_cols].values])

    logging.info(
        "Scaled %d numeric features, kept %d dummy features",
        len(numeric_cols),
        len(categorical_cols),
    )

    # ------------------------------------------------------------------
    # 3. Get model and hyperparameter grid from registry
    # ------------------------------------------------------------------
    model, param_grid = get_model(model_name)
    reports_dir = os.path.join(config['paths']['reports'], model_name)
    os.makedirs(reports_dir, exist_ok=True)

    start_time = time.time()

    if param_grid:
        total_combos = 1
        for vals in param_grid.values():
            total_combos *= len(vals)
        total_fits = total_combos * cv_folds

        logging.info(
            "GridSearchCV: %d combos x %d folds = %d fits, scoring=%s",
            total_combos,
            cv_folds,
            total_fits,
            scoring,
        )
        logging.info(
            "Training on %d samples with %d features — this may take several minutes",
            x_train_scaled.shape[0],
            x_train_scaled.shape[1],
        )

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
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_folds,
            verbose=3,
            n_jobs=1
        )

        grid_search.fit(x_train_scaled, y_train)

        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)

        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        used_grid_search = True

        # Save grid search results for analysis
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(
            os.path.join(reports_dir, 'grid_search_results.csv'),
            index=False
        )
    else:
        # Fast path: train directly with fixed hyperparameters.
        logging.info("Training model directly without grid search")
        best_model = model.fit(x_train_scaled, y_train)
        best_score = None
        best_params = model.get_params()
        used_grid_search = False

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    y_test_pred_prob = best_model.predict_proba(x_test_scaled)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    logging.info("Model trained in %dm %ds", minutes, seconds)
    if used_grid_search:
        logging.info("Best CV %s: %.4f", scoring, best_score)
    logging.info("Test ROC-AUC: %.4f", test_roc_auc)
    logging.info("Best params: %s", best_params)

    # ------------------------------------------------------------------
    # 5. Save the full artifact
    # ------------------------------------------------------------------
    # Everything needed to reproduce predictions on new data:
    # the trained model, the fitted scaler, and the exact feature order.
    artifact = {
        'model': best_model,
        'scaler': scaler,
        'model_name': model_name,
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols,
        'feature_order': list(x_train.columns),
        'best_params': best_params,
        'best_score': best_score,
    }

    models_dir = config['paths']['models']
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, f'{model_name}.joblib')
    joblib.dump(artifact, artifact_path)
    logging.info("Artifact saved to %s", artifact_path)

    try:
        wandb_info = log_training_run(
            model_name=model_name,
            metrics={
                "best_cv_score": best_score,
                "test_roc_auc": test_roc_auc,
                "train_time_sec": elapsed,
                "train_samples": x_train.shape[0],
                "test_samples": x_test.shape[0],
                "n_features": x_train.shape[1],
            },
            artifact_path=artifact_path,
            best_params=best_params,
            reports_dir=reports_dir,
            used_grid_search=used_grid_search,
        )
        if wandb_info:
            artifact['wandb_run_id'] = wandb_info['run_id']
            joblib.dump(artifact, artifact_path)
    except ImportError:
        pass

    return artifact
