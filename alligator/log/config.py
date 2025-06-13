"""
Simple logging configuration for the Alligator entity linking system.

This module provides a centralized logging setup with consistent formatting
and appropriate log levels for different components.
"""

import logging
import os
import sys
from typing import Optional


class SilentLogger(logging.Logger):
    """A completely silent logger that does nothing."""

    def __init__(self, name: str = "silent"):
        super().__init__(name)
        self.setLevel(logging.CRITICAL + 1)

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def __getstate__(self):
        """Support for pickling."""
        return {"name": self.name}

    def __setstate__(self, state):
        """Support for unpickling."""
        self.__init__(state["name"])

    def __reduce__(self):
        """Support for multiprocessing pickling."""
        return (SilentLogger, (self.name,))


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    disable_logging: bool = False,
) -> logging.Logger:
    """
    Set up logging configuration for the Alligator system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log messages
        disable_logging: Completely disable all logging

    Returns:
        Configured logger instance
    """

    # Set environment variable to communicate disabled state to worker processes
    if disable_logging:
        os.environ["ALLIGATOR_DISABLE_LOGGING"] = "1"
        logging.disable(logging.CRITICAL)  # Disable all logging
        return SilentLogger("alligator")
    else:
        os.environ.pop("ALLIGATOR_DISABLE_LOGGING", None)
        logging.disable(logging.NOTSET)  # Re-enable logging

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
    # Check if logging is disabled via environment variable (for worker processes)
    if os.environ.get("ALLIGATOR_DISABLE_LOGGING") == "1":
        # Ensure logging is disabled in this process too
        logging.disable(logging.CRITICAL)
        return SilentLogger(f"alligator.{name}")

    return logging.getLogger(f"alligator.{name}")


def disable_logging():
    """Completely disable all logging."""
    os.environ["ALLIGATOR_DISABLE_LOGGING"] = "1"
    logging.disable(logging.CRITICAL)


def enable_logging():
    """Re-enable logging."""
    os.environ.pop("ALLIGATOR_DISABLE_LOGGING", None)
    logging.disable(logging.NOTSET)


# Initialize default logging setup
# Check if logging should be disabled at import time (for worker processes)
if os.environ.get("ALLIGATOR_DISABLE_LOGGING") == "1":
    logging.disable(logging.CRITICAL)
    logger = SilentLogger("alligator")
else:
    logger = setup_logging()
