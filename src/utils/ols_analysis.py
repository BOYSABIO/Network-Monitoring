"""
# OLS Feature Significance Analysis
# -----------------------------------
# OLS (Ordinary Least Squares) is used here as a quick diagnostic tool
# to estimate which features have a statistically significant linear
# relationship with the target variable.
#
# IMPORTANT: OLS assumes a continuous target. For a binary label (0/1)
# the p-values are approximate, not exact. For rigorous feature selection,
# consider logistic regression coefficients, chi-squared tests, or mutual
# information instead.
#
# This module is intentionally separate from training and feature engineering
# because OLS analysis is a diagnostic — it helps you understand your data,
# but it is NOT part of the model pipeline itself.
"""

import logging
import os
import pandas as pd
from statsmodels.api import OLS, add_constant


def run_ols_analysis(x_train, y_train, report_dir='reports/ols'):
    """
    Run OLS regression on training data to assess feature significance.

    Why OLS here?
    - Quick way to get p-values for every feature at once
    - Helps identify which features the model will likely rely on
    - Useful for documentation and understanding, not for prediction

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features (already split — never use test data here).
    y_train : pd.Series
        Training labels.
    report_dir : str
        Directory to save the OLS report CSV.

    Returns
    -------
    pd.DataFrame
        Sorted OLS summary with coefficients, p-values, and std errors.
    """
    logging.info("Running OLS feature significance analysis...")

    # add_constant adds an intercept column — OLS needs this explicitly
    ols_x = x_train.astype(float)
    ols_x_const = add_constant(ols_x)
    ols_model = OLS(y_train, ols_x_const).fit()

    # Build a summary table: coefficient, p-value, standard error
    ols_summary = (
        pd.DataFrame(ols_model.params, columns=['coef'])
        .join(ols_model.pvalues.rename('p_value'))
        .join(ols_model.bse.rename('std_err'))
    )

    ols_summary_sorted = ols_summary.sort_values('p_value')

    # Count features that pass the standard significance threshold
    significant = ols_summary_sorted['p_value'] < 0.05
    logging.info(
        "OLS: %d of %d features have p < 0.05 (statistically significant)",
        significant.sum(),
        len(ols_summary_sorted),
    )

    # Save the full report for later review
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'ols_feature_significance.csv')
    ols_summary_sorted.to_csv(report_path)
    logging.info("OLS report saved to %s", report_path)

    return ols_summary_sorted
