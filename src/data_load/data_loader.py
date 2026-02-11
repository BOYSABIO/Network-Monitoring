# Data Loader
# -----------
# Loads the raw CSV dataset into a pandas DataFrame.
# This is the entry point of the training pipeline — raw data in, DataFrame out.

import logging
import pandas as pd


def load_data(raw_data_path='data/01_Raw/rawdata.csv'):
    """
    Load the raw CSV file into a DataFrame.

    Parameters
    ----------
    raw_data_path : str
        Path to the CSV file (relative to project root or absolute).

    Returns
    -------
    pd.DataFrame
        The raw dataset, unmodified.
    """
    try:
        logging.info(f"Loading data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def main():
    """Standalone entry point for testing the loader."""
    from src.utils.logger import setup_logger
    setup_logger()
    load_data()


if __name__ == "__main__":
    main()
