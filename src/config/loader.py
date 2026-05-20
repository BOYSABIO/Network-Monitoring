"""
# Configuration Loader
# --------------------
# Reads config.yaml and exposes it as a simple dictionary.
# Every module imports config from here instead of hardcoding values.
"""


import os
import logging
from typing import Any, cast

import yaml

# Path to config.yaml — resolved relative to this file's location
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, 'config.yaml')

_CACHE = {"config": None}  # cached after first load


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load the YAML config file and return it as a dictionary.

    Uses a module-level cache so the file is only read once,
    no matter how many modules import it.

    Parameters
    ----------
    config_path : str, optional
        Override path to a different YAML file (useful for testing).
    """
    if _CACHE["config"] is not None and config_path is None:
        return _CACHE["config"]

    path = config_path or _DEFAULT_CONFIG_PATH

    with open(path, 'r', encoding='utf-8') as f:
        _CACHE["config"] = cast(dict[str, Any], yaml.safe_load(f))

    logging.info("Config loaded from %s", path)
    return _CACHE["config"]


def get_config(config_path: str | None = None) -> dict[str, Any]:
    """Convenience alias — returns the cached config dict."""
    return load_config(config_path)


def get_expected_schema(config_path=None):
    """
    Return the raw-data column schema from config.yaml.

    Maps column name -> pandas dtype string (e.g. "int64", "float64").
    """
    config = load_config(config_path)
    schema = config.get("schema")
    if not schema:
        raise KeyError("config.yaml must define 'schema'")
    return {column: str(dtype) for column, dtype in schema.items()}
