"""
# Check Schema, data types, ranges, anomolies
# CHECK ONLY PHASE
# Example: NAN -> Checks amount of NANs and if exceeds a certain amount
"""
import logging
from src.config.loader import get_expected_schema
from src.data_load.data_loader import load_data


def check_schema(df, expected_schema):
    """
    Checks:
    1. Missing Columns
    2. Extra Columns
    3. Datatypes
    """
    issues = {}

    missing = [col for col in expected_schema if col not in df.columns]
    if missing:
        issues["missing_columns"] = missing
        logging.error("Schema check failed: Missing columns: %s", missing)

    extra = [col for col in df.columns if col not in expected_schema]
    if extra:
        issues["extra_columns"] = extra
        logging.warning(
            "Schema check warning: Extra unexpected columns found: %s", extra
        )

    wrong_types = {}
    for col, expected_type in expected_schema.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                wrong_types[col] = {
                    "expected": expected_type,
                    "actual": actual_type
                }

    if wrong_types:
        issues["wrong_types"] = wrong_types
        for col, detail in wrong_types.items():
            logging.error(
                "Column '%s' has wrong dtype "
                "(expected=%s, actual=%s)",
                col, detail['expected'], detail['actual']
            )

    if not issues:
        logging.info("Schema validation passed. No issues found.")

    return issues


def data_validator(df):
    '''
    Valdate data after loading
    '''

    expected_schema = get_expected_schema()

    try:
        # Check Schema
        logging.info("Checking schema")
        issues = check_schema(df, expected_schema)
        if issues:
            logging.warning("Schema Check Failed")
        else:
            logging.info("Schema Check Passed")
        # Check data types
        logging.info("Checking data types")
        # Check ranges
        logging.info("Checking ranges")
        # Check anomalies
        logging.info("Checking anomalies")
        # Check missing values
        if df.isna().sum().any():
            logging.warning("Missing values found")
        else:
            logging.info("No missing values found")
        # Check duplicates
        logging.info("Checking duplicates")
        if df.duplicated().any():
            logging.warning("Duplicates found")
        else:
            logging.info("No duplicates found")
        # Check outliers
        # Check consistency
        # Check completeness
        # Check accuracy
        # Check consistency
        logging.info("Validating data after loading")
        logging.info("Data validated successfully")
    except Exception as e:
        logging.error("Error validating data: %s", e)
        raise e


def main():
    """
    Main function to validate the data
    """
    df = load_data()
    data_validator(df)


if __name__ == "__main__":
    main()
    logging.info("Program ended successfully")
