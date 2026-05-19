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


def decode_one_hot_categories(df, categorical_columns, drop_dummies):
    """
    Convert one-hot columns like proto_udp/service_dns/state_CON 
    back into single categorical columns like proto/service/state.
    """
    out = df.copy()

    # Build baseline map from dropped reference dummies
    # e.g. proto_tcp -> proto: tcp
    baseline = {}
    for d in drop_dummies:
        if "_" in d:
            k, v = d.split("_", 1)
            baseline[k] = v
    
    for cat in categorical_columns:
        prefix = f"{cat}_"
        one_hot_cols = [c for c in out.columns if c.startswith(prefix)]
        if not one_hot_cols:
            continue 
        
        # Which one-hot won?
        winners = out[one_hot_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)

        # Handle all-zero rows (means all dummies were 0, so use baseline)
        has_any = out[one_hot_cols].sum(axis=1) > 0
        out[cat] = np.where(has_any, winners, baseline.get(cat, "unknown"))

        # Remove sparse one-hot columns from export view
        out = out.drop(columns=one_hot_cols)

    return out

def enrich_soc_event_v1(df, model_name, source_type, input_ref, pipeline_version="v1"):
    out = df.copy()

    # Contract metadata
    out["event_version"] = "1.0"
    out["pipeline_version"] = pipeline_version
    out["source_type"] = source_type
    out["input_ref"] = input_ref
    out["model_name"] = model_name 

    if "timestamp" not in out.columns:
        out["timestamp"] = pd.Timestamp.utcnow().isoformat()

    # Risk + Severity
    out["confidence_percentage"] = (out["confidence"] * 100).round().astype(int)
    out["severity"] = np.where(
        out["prediction"] == 0,
        "low",
        np.where(out["confidence"] >= 0.90, "high", "medium")
    )

    # Derive Ports from Services
    service_to_port = {
        "dns": 53, "http": 80, "https": 443, "ssl": 443,
        "ssh": 22, "ftp": 21, "ftp-data": 20, "smtp": 25,
        "pop3": 110, "dhcp": 67, "snmp": 161, "radius": 1812, "irc": 6667
    }

    # Ensure identity fields exist (nullable for downstream typing)
    for c in ["src_ip", "dst_ip", "src_port", "dst_port"]:
        if c not in out.columns:
            out[c] = None

    out["likely_dst_port"] = out["service"].map(service_to_port).astype("Int64")
    out["dst_port_inferred"] = out["dst_port"].isna() & out["likely_dst_port"].notna()

    return out


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


    # Capture Flow ID / ID to be added back after
    df_input = df.copy()

    if 'flow_id' in df_input.columns:
        flow_id_series = df_input['flow_id'].astype(str)
    elif 'id' in df_input.columns:
        flow_id_series = 'flow_' + df_input['id'].astype(str)
    else:
        flow_id_series = pd.Series(
            [f"flow_{i:06d}" for i in range(len(df_input))],
            index=df_input.index
        )


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
    result.insert(0, 'flow_id', flow_id_series.values)
    result['prediction'] = predictions
    result['confidence'] = confidence
    result['prediction_label'] = np.where(predictions == 1, 'malicious', 'benign')

    export_result = decode_one_hot_categories(
        result,
        categorical_columns=cat_columns,
        drop_dummies=drop_dummies
    )

    # Summary logging
    n_malicious = (predictions == 1).sum()
    n_benign = (predictions == 0).sum()
    logging.info(
        f"Predictions: {n_malicious} malicious, {n_benign} benign "
        f"(avg confidence: {confidence.mean():.3f})"
    )

    if n_malicious > 0:
        logging.warning(f"ALERT: {n_malicious} potentially malicious connections detected")

    return {
        "model_result": result,
        "export_result": export_result
    }
