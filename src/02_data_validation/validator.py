# Check Schema, data types, ranges, anomolies
# CHECK ONLY PHASE
# Example: NAN -> Checks amount of NANs and if exceeds a certain amount, then flag

import logging
import os
import pandas as pd
import numpy as np
from src.data_load.data_loader import load_data

def data_validator():
    '''
    Valdate data after loading
    '''
    try:
        logging.info("Validating data after loading")
        df = load_data()
        logging.info("Data validated successfully")
    except Exception as e:
        logging.error(f"Error validating data: {e}")
        raise e

def main():
    data_validator()

if __name__ == "__main__":
    main()
    logging.info("Program ended successfully")