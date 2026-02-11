# PCAP-to-Features Ingestion Module
# -----------------------------------
# Converts raw PCAP files (or live network traffic) into a DataFrame
# matching the UNSW-NB15 feature schema so the trained model can classify
# each connection as malicious or benign.
#
# Uses Zeek (free, open source) for network traffic analysis:
#   - Batch mode:  zeek -r capture.pcap  -> conn.log -> DataFrame
#   - Live mode:   zeek -i eth0          -> conn.log (streaming) -> DataFrame
#
# Zeek's conn.log provides per-connection summaries. We map its fields
# to the UNSW-NB15 feature names, and compute derived features
# (like ct_srv_src, ct_dst_ltm) from a rolling window of recent connections.
#
# Not all 49 UNSW-NB15 features can be perfectly reconstructed from
# Zeek's conn.log alone. Features that require packet-level inspection
# (e.g., swin, dwin, stcpb, dtcpb) are set to 0 by default. The model
# still works because it learned which features matter most during training.

import logging
import os
import subprocess
import tempfile
import time
from collections import deque

import pandas as pd
import numpy as np

from src.config.loader import get_config


# -------------------------------------------------------
# Zeek conn.log field -> UNSW-NB15 feature mapping
# -------------------------------------------------------
# Zeek conn.log columns (tab-separated, # header lines):
#   ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p,
#   proto, service, duration, orig_bytes, resp_bytes,
#   conn_state, local_orig, local_resp, missed_bytes,
#   history, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes,
#   tunnel_parents

# Mapping from Zeek field name -> UNSW-NB15 feature name
ZEEK_TO_UNSW = {
    'duration':    'dur',
    'proto':       'proto',
    'service':     'service',
    'conn_state':  'state',
    'orig_pkts':   'spkts',
    'resp_pkts':   'dpkts',
    'orig_bytes':  'sbytes',
    'resp_bytes':  'dbytes',
}


def parse_conn_log(log_path):
    """
    Parse a Zeek conn.log file into a list of dictionaries.

    Zeek logs use a tab-separated format with comment header lines
    starting with '#'. The '#fields' line tells us the column names.

    Parameters
    ----------
    log_path : str
        Path to a Zeek conn.log file.

    Returns
    -------
    list[dict]
        One dict per connection with Zeek field names as keys.
    """
    fields = None
    rows = []

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#fields'):
                # Extract column names from the header
                fields = line.split('\t')[1:]
            elif line.startswith('#'):
                continue  # skip other comment lines
            elif fields:
                values = line.split('\t')
                if len(values) == len(fields):
                    rows.append(dict(zip(fields, values)))

    logging.info(f"Parsed {len(rows)} connections from {log_path}")
    return rows


def _safe_float(val, default=0.0):
    """Convert a Zeek value to float, handling '-' (unset) and errors."""
    if val == '-' or val == '(empty)':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Convert a Zeek value to int, handling '-' (unset) and errors."""
    return int(_safe_float(val, default))


def compute_derived_features(connections, current_idx, window_size=100):
    """
    Compute UNSW-NB15 connection-window features from recent history.

    These features count how many recent connections share certain
    attributes with the current connection. They capture patterns like:
    - ct_srv_src: same (service, src_ip) pairs -> possible scanning
    - ct_dst_ltm: connections to the same dest in recent window
    - ct_src_dport_ltm: same (src_ip, dest_port) pairs
    - ct_dst_sport_ltm: same (dest_ip, src_port) pairs
    - ct_dst_src_ltm: same (dest_ip, src_ip) pairs

    Parameters
    ----------
    connections : list[dict]
        All parsed Zeek connection records.
    current_idx : int
        Index of the current connection being processed.
    window_size : int
        How many recent connections to look back.

    Returns
    -------
    dict
        Derived feature values.
    """
    current = connections[current_idx]
    src_ip = current.get('id.orig_h', '')
    dst_ip = current.get('id.resp_h', '')
    src_port = current.get('id.orig_p', '')
    dst_port = current.get('id.resp_p', '')
    service = current.get('service', '-')
    state = current.get('conn_state', '')

    # Look at the last `window_size` connections before this one
    start = max(0, current_idx - window_size)
    window = connections[start:current_idx]

    ct_srv_src = sum(
        1 for c in window
        if c.get('service', '-') == service and c.get('id.orig_h', '') == src_ip
    )
    ct_state_ttl = sum(
        1 for c in window
        if c.get('conn_state', '') == state
    )
    ct_dst_ltm = sum(
        1 for c in window
        if c.get('id.resp_h', '') == dst_ip
    )
    ct_src_dport_ltm = sum(
        1 for c in window
        if c.get('id.orig_h', '') == src_ip and c.get('id.resp_p', '') == dst_port
    )
    ct_dst_sport_ltm = sum(
        1 for c in window
        if c.get('id.resp_h', '') == dst_ip and c.get('id.orig_p', '') == src_port
    )
    ct_dst_src_ltm = sum(
        1 for c in window
        if c.get('id.resp_h', '') == dst_ip and c.get('id.orig_h', '') == src_ip
    )
    ct_src_ltm = sum(
        1 for c in window
        if c.get('id.orig_h', '') == src_ip
    )
    ct_srv_dst = sum(
        1 for c in window
        if c.get('service', '-') == service and c.get('id.resp_h', '') == dst_ip
    )

    return {
        'ct_srv_src': ct_srv_src,
        'ct_state_ttl': ct_state_ttl,
        'ct_dst_ltm': ct_dst_ltm,
        'ct_src_dport_ltm': ct_src_dport_ltm,
        'ct_dst_sport_ltm': ct_dst_sport_ltm,
        'ct_dst_src_ltm': ct_dst_src_ltm,
        'ct_src_ltm': ct_src_ltm,
        'ct_srv_dst': ct_srv_dst,
    }


def zeek_connections_to_dataframe(connections):
    """
    Convert parsed Zeek connection records into a DataFrame matching
    the UNSW-NB15 feature schema.

    Parameters
    ----------
    connections : list[dict]
        Parsed Zeek conn.log records.

    Returns
    -------
    pd.DataFrame
        One row per connection with UNSW-NB15 column names.
    """
    rows = []

    for i, conn in enumerate(connections):
        # --- Direct mappings from Zeek conn.log ---
        row = {
            'dur':       _safe_float(conn.get('duration', '0')),
            'proto':     conn.get('proto', 'unknown'),
            'service':   conn.get('service', '-'),
            'state':     conn.get('conn_state', 'unknown'),
            'spkts':     _safe_int(conn.get('orig_pkts', '0')),
            'dpkts':     _safe_int(conn.get('resp_pkts', '0')),
            'sbytes':    _safe_int(conn.get('orig_bytes', '0')),
            'dbytes':    _safe_int(conn.get('resp_bytes', '0')),
        }

        # --- Computed from available data ---
        dur = row['dur'] if row['dur'] > 0 else 1e-6
        total_pkts = row['spkts'] + row['dpkts']
        row['rate'] = total_pkts / dur if dur > 0 else 0.0

        # sload/dload: bits per second in each direction
        row['sload'] = (row['sbytes'] * 8) / dur
        row['dload'] = (row['dbytes'] * 8) / dur

        # Packet loss: estimate from byte counts (approximate)
        row['sloss'] = max(0, row['spkts'] - 1)
        row['dloss'] = max(0, row['dpkts'] - 1)

        # Mean packet size
        row['smean'] = row['sbytes'] // max(row['spkts'], 1)
        row['dmean'] = row['dbytes'] // max(row['dpkts'], 1)

        # Inter-packet arrival time (approximate)
        row['sinpkt'] = dur / max(row['spkts'], 1) * 1000  # milliseconds
        row['dinpkt'] = dur / max(row['dpkts'], 1) * 1000

        # --- Features not available from conn.log (set to 0) ---
        # These require packet-level inspection that Zeek's conn.log
        # doesn't provide. The model can still make predictions without them.
        row['sttl'] = 0
        row['dttl'] = 0
        row['sjit'] = 0.0
        row['djit'] = 0.0
        row['swin'] = 0
        row['dwin'] = 0
        row['stcpb'] = 0
        row['dtcpb'] = 0
        row['tcprtt'] = 0.0
        row['synack'] = 0.0
        row['ackdat'] = 0.0
        row['trans_depth'] = 0
        row['response_body_len'] = 0
        row['is_ftp_login'] = 0
        row['ct_ftp_cmd'] = 0
        row['ct_flw_http_mthd'] = 0
        row['is_sm_ips_ports'] = (
            1 if (conn.get('id.orig_h') == conn.get('id.resp_h') and
                  conn.get('id.orig_p') == conn.get('id.resp_p'))
            else 0
        )

        # --- Window-based derived features ---
        derived = compute_derived_features(connections, i)
        row.update(derived)

        rows.append(row)

    df = pd.DataFrame(rows)
    logging.info(f"Built feature DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def run_zeek_on_pcap(pcap_path, output_dir=None):
    """
    Run Zeek on a PCAP file to generate conn.log.

    Parameters
    ----------
    pcap_path : str
        Path to the .pcap file.
    output_dir : str, optional
        Directory for Zeek output. Defaults to a temp directory.

    Returns
    -------
    str
        Path to the generated conn.log file.
    """
    config = get_config()
    zeek_binary = config['zeek']['binary']

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='zeek_')

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Running Zeek on {pcap_path}...")

    try:
        subprocess.run(
            [zeek_binary, '-r', pcap_path],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        logging.error(
            f"Zeek binary not found at '{zeek_binary}'. "
            "Install Zeek: https://zeek.org/get-zeek/"
        )
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Zeek failed: {e.stderr}")
        raise

    conn_log = os.path.join(output_dir, 'conn.log')
    if not os.path.exists(conn_log):
        raise FileNotFoundError(f"Zeek did not produce conn.log in {output_dir}")

    logging.info(f"Zeek conn.log generated at {conn_log}")
    return conn_log


def ingest_pcap(pcap_path):
    """
    Full batch pipeline: PCAP file -> Zeek -> feature DataFrame.

    Parameters
    ----------
    pcap_path : str
        Path to a .pcap file.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame ready for preprocessing and inference.
    """
    config = get_config()
    output_dir = config['paths']['zeek_logs']

    conn_log = run_zeek_on_pcap(pcap_path, output_dir)
    connections = parse_conn_log(conn_log)
    df = zeek_connections_to_dataframe(connections)
    return df


def ingest_live(interface, callback, duration=None):
    """
    Live monitoring: run Zeek on a network interface and process
    new connections as they appear.

    Parameters
    ----------
    interface : str
        Network interface to monitor (e.g., 'eth0', 'wlan0').
    callback : callable
        Function called with a DataFrame of new connections each cycle.
        Signature: callback(df: pd.DataFrame) -> None
    duration : int, optional
        Stop after this many seconds. None = run until interrupted.
    """
    config = get_config()
    zeek_binary = config['zeek']['binary']
    watch_interval = config['zeek']['watch_interval']
    output_dir = config['paths']['zeek_logs']
    os.makedirs(output_dir, exist_ok=True)

    conn_log = os.path.join(output_dir, 'conn.log')

    logging.info(f"Starting live Zeek capture on interface '{interface}'...")

    # Start Zeek in background monitoring mode
    try:
        zeek_proc = subprocess.Popen(
            [zeek_binary, '-i', interface],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        logging.error(
            f"Zeek binary not found at '{zeek_binary}'. "
            "Install Zeek: https://zeek.org/get-zeek/"
        )
        raise

    logging.info(f"Zeek PID: {zeek_proc.pid} — monitoring {interface}")

    # Track how many lines we've already processed
    lines_processed = 0
    all_connections = []
    start_time = time.time()

    try:
        while True:
            # Check if Zeek is still running
            if zeek_proc.poll() is not None:
                logging.warning("Zeek process exited unexpectedly")
                break

            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                logging.info(f"Duration limit reached ({duration}s), stopping")
                break

            # Read new connections from conn.log
            if os.path.exists(conn_log):
                connections = parse_conn_log(conn_log)

                if len(connections) > lines_processed:
                    new_connections = connections[lines_processed:]
                    all_connections.extend(new_connections)
                    lines_processed = len(connections)

                    # Build features for new connections
                    df = zeek_connections_to_dataframe(new_connections)
                    if not df.empty:
                        logging.info(f"New batch: {len(df)} connections")
                        callback(df)

            time.sleep(watch_interval)

    except KeyboardInterrupt:
        logging.info("Live capture interrupted by user")
    finally:
        # Clean up Zeek process
        zeek_proc.terminate()
        try:
            zeek_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            zeek_proc.kill()
        logging.info("Zeek process stopped")
