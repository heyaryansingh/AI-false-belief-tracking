"""Logging configuration for BSA experiments.

This module provides a centralized logging setup that supports both console
and file output with configurable log levels and formatting.

Example:
    >>> from bsa.common.logging import setup_logging
    >>> logger = setup_logging(level="DEBUG", log_file=Path("logs/experiment.log"))
    >>> logger.info("Starting experiment")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration with console and optional file output.

    Args:
        level: Logging level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
        log_file: Optional path to a log file. Parent directories will be created
            if they do not exist. If None, logs only to console.
        format_string: Custom log format string. Defaults to a standard format
            with timestamp, logger name, level, and message.

    Returns:
        A configured logger instance for the current module.

    Example:
        >>> logger = setup_logging(level="DEBUG")
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )

    return logging.getLogger(__name__)
