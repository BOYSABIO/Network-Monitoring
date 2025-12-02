# Load Raw CSV/PCAP -> DataFrame

import os
import logging 
import pandas as pd
import numpy as np

logging.basicConfig(filename='logs/data_loader.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(raw_data_path = 'data/01_Raw/rawdata.csv'):
    '''
    Load data from raw data path
    '''
    try:
        logging.info(f"Loading data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        logging.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e

def main():
    load_data()
    logging.info("Data loaded successfully")

if __name__ == "__main__":
    main()
    logging.info("Program ended successfully")
