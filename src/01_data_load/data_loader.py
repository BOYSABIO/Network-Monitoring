# Load Raw CSV/PCAP -> DataFrame

import os
import logging 
import pandas as pd

# TRY TO MOVE THE LOGGER TO SEPERATE CONFIG
# Set up the logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler('logs/data_loader.log')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_format)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


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


