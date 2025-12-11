# Feature Engineering (if needed)
# Same or more data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar
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

    categorical_features = [col for col in X_train.columns if col not in numeric_features] 

    X = df.drop(columns='label')
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train.to_csv('../data/03_Enriched/train.csv')
    X_test.to_csv('../data/03_Enriched/test.csv')
    logging.info(f"Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    logging.info(f"Testing samples: {X_test.shape[0]}| Features: {X_test.shape[1]}")

    scalar = StandardScalar()
    X_train_scaled_num = scalar.fit_transform(X_train[numeric_features])
    X_test_scaled_num = scalar.fit_transform(X_test[numeric_features])

    # Combine back together
    X_train_scaled = np.hstach([X_train_scaled_num, X_train[categorical_features].values])
    X_test_scaled = np.hstack([X_test_scaled_num, X_test[categorical_features].values])


def main():
    df = load_data()
    processed = preprocess(df)
    feature_engineering(processed)

if __name__ == "__main__":
    main()
    logging.info("Feature Engineering Completed Successfully")