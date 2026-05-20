"""
# Centralized Logger
# ------------------
# One place to configure logging for the entire pipeline.
# Every module should call setup_logger() once at startup (the CLI
# orchestrator handles this), then use standard logging.info(), etc.
"""

import logging
import os


def setup_logger(log_dir='logs', log_file='pipeline.log', level=logging.INFO):
    """
    Configure the root logger with both console and file output.

    Call this ONCE at application startup (in main.py).
    After that, any module using `import logging; logging.info(...)` will
    automatically use this configuration.

    Parameters
    ----------
    log_dir : str
        Directory for log files.
    log_file : str
        Name of the log file.
    level : int
        Logging level (default: INFO).
    """
    os.makedirs(log_dir, exist_ok=True)

    # Clear any existing handlers to avoid duplicate log lines
    # (happens when setup_logger is called more than once, e.g. in tests)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # Consistent format: timestamp — level — message
    fmt = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')

    # Console handler — see output in terminal
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — persistent log for later review
    filepath = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logging.info("Logger initialized — writing to %s", filepath)
