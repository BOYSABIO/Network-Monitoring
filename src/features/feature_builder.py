# Feature Engineering (if needed)
# Same or more data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from src.data_load.data_loader import load_data
from src.preprocess.preprocess import preprocess

def feature_engineering(df):
    '''
    Feature Engineering
    Same or more data
    '''

    numeric_features = [
        'dur',
        'spkts',
        'dpkts',
        'sbytes',
        'dbytes',
        'rate',
        'sttl',
        'dttl',
        'sload',
        'dload',
        'sloss',
        'dloss',
        'sinpkt',
        'dinpkt',
        'sjit',
        'djit',
        'swin',
        'stcpb',
        'dtcpb',
        'dwin',
        'tcprtt',
        'synack',
        'ackdat',
        'smean',
        'dmean',
        'trans_depth',
        'response_body_len',
        'ct_srv_src',
        'ct_state_ttl',
        'ct_dst_ltm',
        'ct_src_dport_ltm',
        'ct_dst_sport_ltm',
        'ct_dst_src_ltm'
    ]

    categorical_features = [col for col in df.drop(columns=['label']).columns if col not in numeric_features] 

    X = df.drop(columns='label')
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train.to_csv('./data/03_Enriched/train.csv')
    X_test.to_csv('./data/03_Enriched/test.csv')
    logging.info(f"Train Data Saved \nTraining samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    logging.info(f"Test Data Saved \nTesting samples: {X_test.shape[0]}| Features: {X_test.shape[1]}")

    # Scaling: fit ONLY on training data to prevent data leakage.
    # If we fit on test data too, the model "sees" test distributions during training,
    # which inflates metrics and gives a false sense of accuracy.
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[numeric_features])  # learn mean/std from train
    X_test_scaled_num = scaler.transform(X_test[numeric_features])        # apply train's mean/std to test

    # Combine back together
    X_train_scaled = np.hstack([X_train_scaled_num, X_train[categorical_features].values])
    X_test_scaled = np.hstack([X_test_scaled_num, X_test[categorical_features].values])

    # OLS feature analysis is now in src/utils/ols_analysis.py
    # Run it separately: from src.utils.ols_analysis import run_ols_analysis

    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    df = load_data()
    processed = preprocess(df)
    feature_engineering(processed)

if __name__ == "__main__":
    main()
    logging.info("Feature Engineering Completed Successfully")