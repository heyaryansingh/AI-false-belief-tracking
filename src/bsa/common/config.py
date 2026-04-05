"""Configuration management for BSA experiments.

This module provides utilities for loading, saving, and validating YAML-based
configuration files used throughout the belief state attribution framework.

Example:
    >>> from pathlib import Path
    >>> from bsa.common.config import load_config, save_config
    >>> config = load_config(Path("experiments/config.yaml"))
    >>> config["experiment"]["name"] = "new_experiment"
    >>> save_config(config, Path("experiments/config_modified.yaml"))
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ConfigError(Exception):
    """Exception raised for configuration-related errors.

    Attributes:
        message: Human-readable description of the error.
    """

    pass


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        ConfigError: If the configuration file does not exist.

    Example:
        >>> config = load_config(Path("config.yaml"))
        >>> print(config["experiment"]["seed"])
        42
    """
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to a YAML file.

    Creates parent directories if they do not exist.

    Args:
        config: Configuration dictionary to save.
        config_path: Path where the YAML file should be written.

    Example:
        >>> save_config({"seed": 42, "env": "gridhouse"}, Path("output/config.yaml"))
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class BaseConfig(BaseModel):
    """Base configuration model with strict validation.

    All configuration models should inherit from this class to ensure
    consistent validation behavior across the codebase.

    Attributes:
        Config.extra: Set to "forbid" to reject unknown fields.
        Config.frozen: Set to True to make instances immutable.
    """

    class Config:
        extra = "forbid"
        frozen = True
