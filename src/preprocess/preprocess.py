"""
# Cleaning, encoding, scaling
# Same or less data
"""


import logging
import pandas as pd
from src.config.loader import get_expected_schema
from src.data_load.data_loader import load_data
from src.data_validation.validator import check_schema

def preprocess(df):
    '''
    Drops extra columns
    Removes NANs
    '''

    expected_schema = get_expected_schema()

    try:
        issues = check_schema(df, expected_schema)

        if "missing_columns" in issues or "wrong_types" in issues:
            raise ValueError(f"Critical schema violations: {issues}")

        extra_cols = issues.get("extra_columns", [])
        if extra_cols:
            logging.warning("Dropping extra columns: %s", extra_cols)
            df = df.drop(columns=extra_cols)

        # Change - to unknown
        for column in list(df.columns):
            df[column] = df[column].replace('-', 'unknown')

        df = df.dropna()

        # Create Dummy Varables
        logging.info("Number of columns before encoding: %d", len(list(df.columns)))
        df_encoded = pd.get_dummies(df, columns=['proto', 'service', 'state'])
        df_encoded.drop(columns=['proto_tcp', 'service_unknown', 'state_INT'], inplace=True)
        logging.info("Encoded dummy variables and dropped NANs")

        # Drop extra unnecessary variables
        df_clean = df_encoded.drop(columns=['id', 'attack_cat'])
        logging.info("Number of columns after encoding: %d", len(list(df_clean.columns)))

    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise e

    return df_clean

def main():
    '''
    Main function
    '''
    df = load_data()
    processed = preprocess(df)
    processed.to_csv("./data/02_Processed/preprocessed.csv")

if __name__ == "__main__":
    main()
    logging.info("Preprocessing Completed")
