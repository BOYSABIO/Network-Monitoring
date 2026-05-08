# CLI Orchestrator
# ----------------
# Single entry point for the entire pipeline. Uses argparse subcommands:
#
#   python -m src.main train       --model logistic_regression
#   python -m src.main evaluate    --model logistic_regression
#   python -m src.main infer       --input capture.pcap [--model NAME]
#   python -m src.main live        --interface eth0 [--model NAME]
#   python -m src.main ols         (run OLS feature analysis)
#
# Each subcommand wires together the appropriate modules.

import argparse
import logging
import os
import sys
import pandas as pd

from src.utils.logger import setup_logger
from src.config.loader import get_config


def cmd_train(args):
    """Train a model using the full pipeline: load -> validate -> preprocess -> train."""
    from src.data_load.data_loader import load_data
    from src.data_validation.validator import data_validator
    from src.preprocess.preprocess import preprocess
    from src.model.trainer import train

    config = get_config()
    model_name = args.model or config['model']['active']

    logging.info(f"=== TRAINING PIPELINE: {model_name} ===")

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
    logging.info(f"Preprocessed dataset saved to {processed_path}")

    # Step 4: Train model with GridSearchCV
    artifact = train(df_processed, model_name=model_name)

    best_score = artifact.get('best_score')
    score_text = f"{best_score:.4f}" if best_score is not None else "N/A (grid skipped)"
    logging.info(
        f"=== TRAINING COMPLETE: {model_name} — "
        f"Best CV score: {score_text} ==="
    )


def cmd_evaluate(args):
    """Evaluate a trained model on test data."""
    from src.evaluation.evaluator import evaluate

    config = get_config()
    model_name = args.model or config['model']['active']
    artifact_path = os.path.join(config['paths']['models'], f'{model_name}.joblib')

    if not os.path.exists(artifact_path):
        logging.error(f"No artifact found at {artifact_path}. Train the model first.")
        sys.exit(1)

    logging.info(f"=== EVALUATION: {model_name} ===")

    # Load test data from the saved split
    enriched = config['paths']['enriched']
    X_test_path = os.path.join(enriched, 'X_test.csv')
    y_test_path = os.path.join(enriched, 'y_test.csv')

    if not os.path.exists(X_test_path):
        logging.error(f"Test data not found at {X_test_path}. Train the model first.")
        sys.exit(1)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    metrics = evaluate(artifact_path, X_test, y_test)

    logging.info(f"=== EVALUATION COMPLETE: ROC-AUC={metrics['roc_auc']:.4f} ===")


def cmd_infer(args):
    """Run inference on a PCAP file or CSV file."""
    from src.inference.predictor import predict

    config = get_config()
    input_path = args.input

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    model_name = args.model or config['model']['active']
    artifact_path = os.path.join(config['paths']['models'], f'{model_name}.joblib')
    if not os.path.exists(artifact_path):
        logging.error(
            f"No artifact at {artifact_path}. Train with --model {model_name} first."
        )
        sys.exit(1)
    logging.info(f"=== INFERENCE: {input_path} (model={model_name}) ===")

    # Determine input type by extension
    if input_path.endswith('.pcap') or input_path.endswith('.pcapng'):
        # PCAP: run through Zeek first to extract features
        from src.ingestion.pcap_to_features import ingest_pcap
        df = ingest_pcap(input_path)
    elif input_path.endswith('.csv'):
        # CSV: load directly (assumed to have the right columns)
        df = pd.read_csv(input_path)
    else:
        logging.error(f"Unsupported file type: {input_path}. Use .pcap or .csv")
        sys.exit(1)

    # Run predictions
    results = predict(df, artifact_path=artifact_path)

    # Save results
    output_path = args.output or 'reports/inference_results.csv'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    results.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")

    # Save results to a NDJSON file
    results.to_json("reports/inference_results.ndjson", orient="records", lines=True)
    logging.info(f'Json file saved to reports/inference_results.ndjson')

    # Print summary
    n_malicious = (results['prediction'] == 1).sum()
    n_total = len(results)
    logging.info(f"=== INFERENCE COMPLETE: {n_malicious}/{n_total} flagged malicious ===")


def cmd_live(args):
    """Live monitoring on a network interface."""
    from src.inference.predictor import predict, load_artifact
    from src.ingestion.pcap_to_features import ingest_live

    config = get_config()

    model_name = args.model or config['model']['active']
    artifact_path = os.path.join(config['paths']['models'], f'{model_name}.joblib')
    if not os.path.exists(artifact_path):
        logging.error(
            f"No artifact at {artifact_path}. Train with --model {model_name} first."
        )
        sys.exit(1)

    logging.info(
        f"=== LIVE MONITORING: {args.interface} (model={model_name}) ==="
    )

    # Pre-load the model artifact once (avoid reloading per batch)
    artifact = load_artifact(artifact_path)

    def on_new_connections(df):
        """Callback: classify each batch of new connections."""
        results = predict(df, artifact=artifact)
        malicious = results[results['prediction'] == 1]

        if not malicious.empty:
            logging.warning(
                f"ALERT: {len(malicious)} malicious connections in latest batch"
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
    from src.data_load.data_loader import load_data
    from src.preprocess.preprocess import preprocess
    from src.utils.ols_analysis import run_ols_analysis
    from sklearn.model_selection import train_test_split

    config = get_config()

    logging.info("=== OLS FEATURE ANALYSIS ===")

    # Load and preprocess
    raw_path = args.data or config['paths']['raw_data']
    df = load_data(raw_path)
    df_processed = preprocess(df)

    # Split (only use training data for OLS — no leakage)
    target = config['features']['target']
    X = df_processed.drop(columns=target)
    y = df_processed[target]
    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )

    report_dir = args.output or os.path.join(config['paths']['reports'], 'ols')
    run_ols_analysis(X_train, y_train, report_dir=report_dir)

    logging.info("=== OLS ANALYSIS COMPLETE ===")


def main():
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
                         help='Trained model name (artifact models/<name>.joblib; default: config active)')

    # --- live ---
    p_live = subparsers.add_parser('live', help='Live monitoring on a network interface')
    p_live.add_argument('--interface', type=str, required=True,
                        help='Network interface (e.g., eth0, wlan0)')
    p_live.add_argument('--duration', type=int, default=None,
                        help='Stop after N seconds (default: run until Ctrl+C)')
    p_live.add_argument('--model', type=str, default=None,
                        help='Trained model name (artifact models/<name>.joblib; default: config active)')

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
