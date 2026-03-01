# Inference / Predictor Module
# ----------------------------
# Loads a trained model artifact and classifies new network connections
# as malicious (1) or benign (0).
#
# This module bridges ingestion and output: it takes a raw feature DataFrame
# (from Zeek/PCAP ingestion or CSV), applies the same preprocessing and
# scaling used during training, and returns predictions with confidence scores.
#
# The key principle: inference must exactly replicate the training pipeline's
# preprocessing. If the training pipeline scaled, encoded, and ordered
# features a certain way, inference must do the same — otherwise the model
# sees garbage inputs and produces meaningless predictions.

import logging
import os
import joblib
import numpy as np
import pandas as pd

from src.config.loader import get_config
from src.preprocess.preprocess import preprocess


def load_artifact(artifact_path=None):
    """
    Load a saved model artifact from disk.

    Parameters
    ----------
    artifact_path : str, optional
        Path to .joblib file. If None, loads the active model from config.

    Returns
    -------
    dict
        Artifact containing model, scaler, and feature metadata.
    """
    if artifact_path is None:
        config = get_config()
        model_name = config['model']['active']
        artifact_path = os.path.join(config['paths']['models'], f'{model_name}.joblib')

    logging.info(f"Loading artifact from {artifact_path}")
    artifact = joblib.load(artifact_path)
    logging.info(f"Loaded model: {artifact['model_name']}")
    return artifact


def predict(df, artifact=None, artifact_path=None):
    """
    Run inference on a DataFrame of network connections.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame — can be raw (will be preprocessed) or already
        preprocessed. Must contain the features the model expects.
    artifact : dict, optional
        Pre-loaded artifact. If None, loads from artifact_path.
    artifact_path : str, optional
        Path to .joblib artifact. If both artifact and artifact_path are
        None, loads the active model from config.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - 'prediction': 0 (benign) or 1 (malicious)
        - 'confidence': model's probability estimate for the predicted class
    """
    # Load artifact if not provided
    if artifact is None:
        artifact = load_artifact(artifact_path)

    model = artifact['model']
    scaler = artifact['scaler']
    numeric_features = artifact['numeric_features']
    categorical_features = artifact['categorical_features']
    feature_order = artifact['feature_order']

    logging.info(f"Running inference on {len(df)} connections")

    # ------------------------------------------------------------------
    # 1. Preprocess: apply the same encoding used during training
    # ------------------------------------------------------------------
    # The model was trained on dummy-encoded data. New data must be
    # encoded the same way, with the same columns in the same order.
    config = get_config()
    cat_columns = config['features']['categorical']
    drop_dummies = config['features']['drop_dummies']
    drop_cols = config['features']['drop_columns']

    # Drop columns that exist and aren't features
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Replace '-' with 'unknown' (same as training preprocessing)
    for col in df.columns:
        df[col] = df[col].replace('-', 'unknown')

    # One-hot encode categorical features
    cat_present = [c for c in cat_columns if c in df.columns]
    if cat_present:
        df = pd.get_dummies(df, columns=cat_present)

    # Drop reference dummies to avoid multicollinearity (same as training)
    dummies_to_drop = [c for c in drop_dummies if c in df.columns]
    if dummies_to_drop:
        df = df.drop(columns=dummies_to_drop)

    # Remove 'label' if present (inference data shouldn't have labels)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # ------------------------------------------------------------------
    # 2. Align columns to match training feature order
    # ------------------------------------------------------------------
    # Add any missing columns as 0 (e.g., dummy categories not seen in
    # this batch but present during training)
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model expects, in the right order
    df_aligned = df[feature_order]

    # ------------------------------------------------------------------
    # 3. Scale numeric features using the TRAINING scaler
    # ------------------------------------------------------------------
    # Critical: use .transform(), NOT .fit_transform()
    # The scaler must use the same mean/std it learned during training.
    numeric_cols = [c for c in numeric_features if c in df_aligned.columns]
    cat_cols = [c for c in df_aligned.columns if c not in numeric_features]

    X_scaled_num = scaler.transform(df_aligned[numeric_cols])
    X_scaled = np.hstack([X_scaled_num, df_aligned[cat_cols].values])

    # ------------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------------
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Confidence = probability of the predicted class
    confidence = np.max(probabilities, axis=1)

    # ------------------------------------------------------------------
    # 5. Attach results to the original data
    # ------------------------------------------------------------------
    result = df_aligned.copy()
    result['prediction'] = predictions
    result['confidence'] = confidence
    result['prediction_label'] = np.where(predictions == 1, 'malicious', 'benign')

    # Summary logging
    n_malicious = (predictions == 1).sum()
    n_benign = (predictions == 0).sum()
    logging.info(
        f"Predictions: {n_malicious} malicious, {n_benign} benign "
        f"(avg confidence: {confidence.mean():.3f})"
    )

    if n_malicious > 0:
        logging.warning(f"ALERT: {n_malicious} potentially malicious connections detected")

    return result
