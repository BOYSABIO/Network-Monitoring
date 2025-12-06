# Cleaning, encoding, scaling
# Same or less data

import logging
import pandas as pd
from src.data_load.data_loader import load_data
from src.data_validation.validator import check_schema



def preprocess(df):
    '''
    Drops extra columns
    Removes NANs
    '''

    EXPECTED_SCHEMA = {
        "id": "int64",
        "dur": "float64",
        "proto": "object",
        "service": "object",
        "state": "object",
        "spkts": "int64",
        "dpkts": "int64",
        "sbytes": "int64",
        "dbytes": "int64",
        "rate": "float64",
        "sttl": "int64",
        "dttl": "int64",
        "sload": "float64",
        "dload": "float64",
        "sloss": "int64",
        "dloss": "int64",
        "sinpkt": "float64",
        "dinpkt": "float64",
        "sjit": "float64",
        "djit": "float64",
        "swin": "int64",
        "stcpb": "int64",
        "dtcpb": "int64",
        "dwin": "int64",
        "tcprtt": "float64",
        "synack": "float64",
        "ackdat": "float64",
        "smean": "int64",
        "dmean": "int64",
        "trans_depth": "int64",
        "response_body_len": "int64",
        "ct_srv_src": "int64",
        "ct_state_ttl": "int64",
        "ct_dst_ltm": "int64",
        "ct_src_dport_ltm": "int64",
        "ct_dst_sport_ltm": "int64",
        "ct_dst_src_ltm": "int64",
        "is_ftp_login": "int64",
        "ct_ftp_cmd": "int64",
        "ct_flw_http_mthd": "int64",
        "ct_src_ltm": "int64",
        "ct_srv_dst": "int64",
        "is_sm_ips_ports": "int64",
        "attack_cat": "object",
        "label": "int64"
    }

    try:
        issues = check_schema(df, EXPECTED_SCHEMA)

        if "missing_columns" in issues or "wrong_types" in issues:
            raise ValueError(f"Critical schema violations: {issues}")
        
        extra_cols = issues.get("extra_columns", [])
        if extra_cols:
            logging.warning(f"Dropping extra columns: {extra_cols}")
            df = df.drop(columns=extra_cols)
        
        df = df.dropna()

        logging.info("Preprocessing Successfully Completed")

    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise e

    return df

def main():
    df = load_data()
    processed = preprocess(df)
    processed.to_csv("./data/02_Processed/preprocessed.csv")

if __name__ == "__main__":
    main()
    logging.info("Preprocessing Completed")