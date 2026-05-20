"""
Model Evaluator
# ---------------
# Model-agnostic evaluation: takes a saved artifact and test data,
# produces metrics and reports. Works with any sklearn-compatible model.
#
# Key metrics for network security classification:
#   - ROC-AUC: overall ranking quality (best single metric for imbalanced data)
#   - Precision: of everything flagged malicious, how much actually was?
#   - Recall: of all actual attacks, how many did we catch?
#   - Confusion matrix: FP/FN breakdown (FN = missed attack, FP = false alarm)
"""
import logging
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from src.config.loader import get_config
from src.utils.wandb_tracker import log_evaluation_run


def evaluate(artifact_path, x_test, y_test):
    """
    Evaluate a trained model on held-out test data.

    Parameters
    ----------
    artifact_path : str
        Path to the .joblib artifact saved by trainer.py.
    x_test : pd.DataFrame
        Test features (not yet scaled — the artifact's scaler is applied here).
    y_test : pd.Series
        True test labels.

    Returns
    -------
    dict
        Dictionary of all computed metrics.
    """
    config = get_config()

    # Load the artifact (model + scaler + feature metadata)
    artifact = joblib.load(artifact_path)
    model = artifact['model']
    scaler = artifact['scaler']
    numeric_features = artifact['numeric_features']
    categorical_features = artifact['categorical_features']
    model_name = artifact['model_name']

    logging.info("Evaluating model: %s", model_name)

    # Scale test data using the TRAINING scaler (no leakage)
    # The scaler was fit on training data during trainer.py.
    # We only call .transform() here — never .fit_transform().
    x_test_scaled_num = scaler.transform(x_test[numeric_features])
    x_test_scaled = np.hstack([
        x_test_scaled_num,
        x_test[categorical_features].values
    ])

    # Generate predictions
    y_pred = model.predict(x_test_scaled)

    # predict_proba gives the model's confidence for each class.
    # [:, 1] = probability of class 1 (malicious).
    # We need probabilities (not hard 0/1 predictions) for ROC-AUC,
    # because AUC measures ranking quality across all thresholds.
    y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)

    # ROC-AUC: use probabilities, NOT hard predictions.
    # Using y_pred instead of y_pred_prob is a common mistake that
    # underestimates model quality.
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    # Classification report: precision, recall, f1 per class
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # FN (bottom-left) = missed attacks, FP (top-right) = false alarms
    cm = confusion_matrix(y_test, y_pred)

    # ROC curve coordinates for plotting
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Log results
    logging.info("Accuracy: %.4f", accuracy)
    logging.info("ROC-AUC: %.4f", roc_auc)
    logging.info("Confusion Matrix:\n%s", cm)
    logging.info("Classification Report:\n%s", report_text)

    # 6. Save reports to disk
    reports_dir = os.path.join(config['paths']['reports'], model_name)
    os.makedirs(reports_dir, exist_ok=True)

    # Structured metrics as JSON for programmatic access
    best_score = artifact.get('best_score')
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'best_params': artifact.get('best_params', {}),
        'best_cv_score': (
            float(best_score) if best_score is not None else None
        ),
    }

    metrics_path = os.path.join(reports_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to %s", metrics_path)

    # Classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(reports_dir, 'classification_report.csv'))

    # ROC curve data for plotting later
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
    roc_df.to_csv(os.path.join(reports_dir, 'roc_curve.csv'), index=False)

    logging.info("All reports saved to %s", reports_dir)

    try:
        log_evaluation_run(
            model_name=model_name,
            metrics={
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "best_cv_score": metrics.get("best_cv_score"),
            },
            artifact_path=artifact_path,
        )
    except ImportError:
        pass

    return metrics
