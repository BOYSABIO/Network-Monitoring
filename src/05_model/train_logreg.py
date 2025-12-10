# Training Script for Logistic Regression
from src.data_load.data_loader import load_data
from src.data_validation.validator import data_validator
from src.preprocess.preprocess import preprocess
import pandas as pd
import logging

def train_logreg(df):
    '''
    Split for training
    OLS Features / EDA for documentation
    Train Model
    Save
    '''
    return

def main():
    df = load_data()
    data_validator(df)
    processed = preprocess(df)
    train_logreg(df)

if __name__ == '__main__':
    main()
    logging.info("LogReg Training Completed.")