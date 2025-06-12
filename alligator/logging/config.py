"""
Simple logging configuration for the Alligator entity linking system.

This module provides a centralized logging setup with consistent formatting
and appropriate log levels for different components.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO", format_string: Optional[str] = None, include_timestamp: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the Alligator system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages

    Returns:
        Configured logger instance
    """

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format string with [Time - module - level] format
    if format_string is None:
        if include_timestamp:
            format_string = "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        else:
            format_string = "[%(name)s - %(levelname)s] %(message)s"

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Return the root logger for Alligator
    return logging.getLogger("alligator")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Name of the component (e.g., 'coordinator', 'data_manager')

    Returns:
        Logger instance for the component
    """
    return logging.getLogger(f"alligator.{name}")


# Initialize default logging setup
logger = setup_logging()
