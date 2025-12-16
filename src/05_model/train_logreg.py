# Training Script for Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from src.data_load.data_loader import load_data
from src.preprocess.preprocess import preprocess
from statsmodels.api import OLS, add_constant
from src.data_validation.validator import data_validator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

def train_logreg(df):
    '''
    Split for training
    OLS Features / EDA for documentation
    Train Model
    Save
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

    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled_num = scaler.fit_transform(X_test[numeric_features])

    # Combine back together
    X_train_scaled = np.hstack([X_train_scaled_num, X_train[categorical_features].values])
    X_test_scaled = np.hstack([X_test_scaled_num, X_test[categorical_features].values])

    # Feature Analysis
    logging.info('OLS Feature Sigificance...')
    ols_X = X_train.astype(float)
    ols_X_const = add_constant(ols_X)
    ols_model = OLS(y_train, ols_X_const).fit()

    ols_summary_df = (
        pd.DataFrame(ols_model.params, columns = ['coef'])
        .join(ols_model.pvalues.rename('p_value'))
        .join(ols_model.bse.rename('std_err'))
    )

    ols_summary_df_sorted = ols_summary_df.sort_values('p_value')
    significant_mask = ols_summary_df_sorted['p_value'] < 0.05
    logging.info(f'Total features with p < 0.05: {significant_mask.sum()} out of {ols_summary_df_sorted.shape[0]}')

    ols_summary_df_sorted.to_csv('./reports/logreg/ols.csv')
    logging.info("Full OLS report logged in reports")

    elastic_net = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=1000,
        random_state=42
    )

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    total_candidates = len(param_grid['C']) * len(param_grid['l1_ratio'])
    total_fits = total_candidates * 3 #cv = 3
    logging.info(f"Starting Elastic Net Grid Search over {total_candidates} combos ({total_fits} fits)...")

    grid_search = GridSearchCV(
        estimator=elastic_net,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=3
    )

    grid_search.fit(X_train_scaled, y_train)
    logging.info("Grid Search Complete. Selecting Best Model...")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    logging.info(f"Best CV ROC-AUC: {grid_search.best_score_:3f}")
    logging.info(f"Best Params: {grid_search.best_params_}")

    joblib.dump(best_model, '../models/logreg.joblib')
    logging.info("Model Saved")

def main():
    df = load_data()
    data_validator(df)
    processed = preprocess(df)
    #feature_engineering(processed)
    train_logreg(processed)

if __name__ == '__main__':
    main()
    logging.info("LogReg Training Completed.")