# Configuration Loader
# --------------------
# Reads config.yaml and exposes it as a simple dictionary.
# Every module imports config from here instead of hardcoding values.

import os
import logging
import yaml

# Path to config.yaml — resolved relative to this file's location
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, 'config.yaml')

_config = None  # cached after first load


def load_config(config_path=None):
    """
    Load the YAML config file and return it as a dictionary.

    Uses a module-level cache so the file is only read once,
    no matter how many modules import it.

    Parameters
    ----------
    config_path : str, optional
        Override path to a different YAML file (useful for testing).
    """
    global _config

    if _config is not None and config_path is None:
        return _config

    path = config_path or _DEFAULT_CONFIG_PATH

    with open(path, 'r') as f:
        _config = yaml.safe_load(f)

    logging.info(f"Config loaded from {path}")
    return _config


def get_config(config_path=None):
    """Convenience alias — returns the cached config dict."""
    return load_config(config_path)
