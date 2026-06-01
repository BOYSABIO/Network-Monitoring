"""
CLI Orchestrator
----------------
Single entry point for the entire pipeline. Uses argparse subcommands:

  python -m src.main train       --model logistic_regression
  python -m src.main evaluate    --model logistic_regression
  python -m src.main infer       --input capture.pcap [--model NAME]
  python -m src.main live        --interface eth0 [--model NAME]
  python -m src.main ols         (run OLS feature analysis)

Each subcommand wires together the appropriate modules.
"""

import argparse
import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.config.loader import get_config
from src.inference.predictor import enrich_soc_event_v1
from src.data_load.data_loader import load_data
from src.data_validation.validator import data_validator
from src.preprocess.preprocess import preprocess
from src.model.trainer import train
from src.evaluation.evaluator import evaluate
from src.inference.predictor import predict, load_artifact
from src.ingestion.pcap_to_features import ingest_live
from src.ingestion.pcap_to_features import ingest_pcap
from src.utils.ols_analysis import run_ols_analysis


def cmd_train(args):
    """
    Train a model using the full pipeline: load -> validate -> preprocess -> train.
    """
    config = get_config()
    model_name = args.model or config['model']['active']

    logging.info("=== TRAINING PIPELINE: %s ===", model_name)

    # Step 1: Load raw data
    raw_path = args.data or config['paths']['raw_data']
    df = load_data(raw_path)

    # Step 2: Validate data quality
    data_validator(df)

    # Step 3: Preprocess (clean, encode, drop columns)
    df_processed = preprocess(df)

    # Persist the fully preprocessed dataset for inspection/reuse
    processed_dir = config['paths']['processed']
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, 'preprocessed.csv')
    df_processed.to_csv(processed_path, index=False)
    logging.info("Preprocessed dataset saved to %s", processed_path)

    # Step 4: Train model with GridSearchCV
    artifact = train(df_processed, model_name=model_name)

    best_score = artifact.get('best_score')
    score_text = f"{best_score:.4f}" if best_score is not None else "N/A (grid skipped)"
    logging.info(
        "=== TRAINING COMPLETE: %s — Best CV score: %s ===",
        model_name, score_text
    )


def cmd_evaluate(args):
    """Evaluate a trained model on test data."""

    config = get_config()
    model_name = args.model or config['model']['active']
    artifact_path = os.path.join(config['paths']['models'], f"{model_name}.joblib")

    if not os.path.exists(artifact_path):
        logging.error("No artifact found at %s. Train the model first.", artifact_path)
        sys.exit(1)

    logging.info("=== EVALUATION: %s ===", model_name)

    # Load test data from the saved split
    enriched = config['paths']['enriched']
    x_test_path = os.path.join(enriched, 'X_test.csv')
    y_test_path = os.path.join(enriched, 'y_test.csv')

    if not os.path.exists(x_test_path):
        logging.error("Test data not found at %s. Train the model first.", x_test_path)
        sys.exit(1)

    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    metrics = evaluate(artifact_path, x_test, y_test)

    logging.info("=== EVALUATION COMPLETE: ROC-AUC=%.4f ===", metrics['roc_auc'])


def cmd_infer(args):
    """Run inference on a PCAP file or CSV file."""

    config = get_config()
    input_path = args.input

    if not os.path.exists(input_path):
        logging.error("Input file not found: %s", input_path)
        sys.exit(1)

    model_name = args.model or config['model']['active']
    artifact_path = os.path.join(config['paths']['models'], f'{model_name}.joblib')
    if not os.path.exists(artifact_path):
        logging.error(
            "No artifact at %s. Train with --model %s first.",
            artifact_path, model_name
        )
        sys.exit(1)
    logging.info("=== INFERENCE: %s (model=%s) ===", input_path, model_name)

    # Determine input type by extension
    if input_path.endswith('.pcap') or input_path.endswith('.pcapng'):
        # PCAP: run through Zeek first to extract features
        df = ingest_pcap(input_path)
    elif input_path.endswith('.csv'):
        # CSV: load directly (assumed to have the right columns)
        df = pd.read_csv(input_path)
    else:
        logging.error("Unsupported file type: %s. Use .pcap or .csv", input_path)
        sys.exit(1)

    # Run predictions
    pred_out = predict(df, artifact_path=artifact_path)
    results_model = pred_out['model_result']
    results_export = pred_out['export_result']

    source_type = "csv" if input_path.endswith('.csv') else "pcap"
    soc_events = enrich_soc_event_v1(
        results_export,
        model_name=model_name,
        source_type=source_type,
        input_ref=input_path,
        pipeline_version="v1"
    )

    # Save results
    output_path = args.output or 'reports/inference_results.csv'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    results_model.to_csv(output_path, index=False)
    logging.info("Results saved to %s", output_path)

    # Save results to a NDJSON file
    output_ndjson = 'reports/inference_results.ndjson'
    soc_events.to_json(output_ndjson, orient="records", lines=True)
    logging.info("Json file saved to %s", output_ndjson)

    # Print summary
    n_malicious = (results_model['prediction'] == 1).sum()
    n_total = len(results_model)
    logging.info("=== INFERENCE COMPLETE: %d/%d flagged malicious ===", n_malicious, n_total)


def cmd_live(args):
    """Live monitoring on a network interface."""

    config = get_config()

    model_name = args.model or config['model']['active']
    artifact_path = os.path.join(config['paths']['models'], f'{model_name}.joblib')
    if not os.path.exists(artifact_path):
        logging.error(
            "No artifact at %s. Train with --model %s first.",
            artifact_path, model_name
        )
        sys.exit(1)

    logging.info(
        "=== LIVE MONITORING: %s (model=%s) ===",
        args.interface, model_name
    )

    # Pre-load the model artifact once (avoid reloading per batch)
    artifact = load_artifact(artifact_path)

    def on_new_connections(df):
        """Callback: classify each batch of new connections."""
        pred_out = predict(df, artifact=artifact)
        results_model = pred_out['model_result']
        malicious = results_model[results_model['prediction'] == 1]

        if not malicious.empty:
            logging.warning(
                "ALERT: %s malicious connections in latest batch",
                len(malicious)
            )
            # Could extend this to: send email, push to SIEM, write to DB, etc.

    ingest_live(
        interface=args.interface,
        callback=on_new_connections,
        duration=args.duration
    )

    logging.info("=== LIVE MONITORING STOPPED ===")


def cmd_ols(args):
    """Run OLS feature significance analysis (diagnostic, not training)."""

    config = get_config()

    logging.info("=== OLS FEATURE ANALYSIS ===")

    # Load and preprocess
    raw_path = args.data or config['paths']['raw_data']
    df = load_data(raw_path)
    df_processed = preprocess(df)

    # Split (only use training data for OLS — no leakage)
    target = config['features']['target']
    x = df_processed.drop(columns=target)
    y = df_processed[target]
    x_train, _, y_train, _ = train_test_split(
        x, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )

    report_dir = args.output or os.path.join(config['paths']['reports'], 'ols')
    run_ols_analysis(x_train, y_train, report_dir=report_dir)

    logging.info("=== OLS ANALYSIS COMPLETE ===")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Network Monitoring MLOps Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main train --model logistic_regression
  python -m src.main evaluate
  python -m src.main infer --input capture.pcap --model logistic_regression_fast
  python -m src.main live --interface eth0 --duration 300 --model logistic_regression_fast
  python -m src.main ols
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # --- train ---
    p_train = subparsers.add_parser('train', help='Train a model')
    p_train.add_argument('--model', type=str, default=None,
                         help='Model name from registry (default: config active)')
    p_train.add_argument('--data', type=str, default=None,
                         help='Path to raw CSV data (default: config path)')

    # --- evaluate ---
    p_eval = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    p_eval.add_argument('--model', type=str, default=None,
                        help='Model name (default: config active)')

    # --- infer ---
    p_infer = subparsers.add_parser('infer', help='Run inference on PCAP or CSV')
    p_infer.add_argument('--input', type=str, required=True,
                         help='Path to .pcap or .csv file')
    p_infer.add_argument('--output', type=str, default=None,
                         help='Path for results CSV (default: reports/inference_results.csv)')
    p_infer.add_argument('--model', type=str, default=None,
                         help='Model name (default: config model.active)')

    # --- live ---
    p_live = subparsers.add_parser('live', help='Live monitoring on a network interface')
    p_live.add_argument('--interface', type=str, required=True,
                        help='Network interface (e.g., eth0, wlan0)')
    p_live.add_argument('--duration', type=int, default=None,
                        help='Stop after N seconds (default: run until Ctrl+C)')
    p_live.add_argument('--model', type=str, default=None,
                        help='Model name (default: config model.active)')

    # --- ols ---
    p_ols = subparsers.add_parser('ols', help='Run OLS feature significance analysis')
    p_ols.add_argument('--data', type=str, default=None,
                       help='Path to raw CSV data (default: config path)')
    p_ols.add_argument('--output', type=str, default=None,
                       help='Report output directory')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize logging before anything else
    setup_logger()

    # Dispatch to the right subcommand
    commands = {
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'infer': cmd_infer,
        'live': cmd_live,
        'ols': cmd_ols,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
