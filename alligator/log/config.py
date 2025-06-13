"""
Simple logging configuration for the Alligator entity linking system.

This module provides a centralized logging setup with consistent formatting
and appropriate log levels for different components.
"""

import logging
import os
import sys
from enum import IntEnum
from typing import Optional


class LogLevel(IntEnum):
    """Log levels that can be set by users via environment variables."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ConditionalLogger(logging.Logger):
    """A logger subclass that respects user-configurable minimum log level."""

    def __init__(self, name: str):
        super().__init__(name)
        self._user_min_level = self._get_user_min_level()

        # Inherit configuration from root logger if it has handlers
        root_logger = logging.getLogger()
        if root_logger.handlers and not self.handlers:
            # Copy handlers from root logger
            for handler in root_logger.handlers:
                self.addHandler(handler)
            # Set the logger level to match the root logger's level
            if root_logger.level != logging.NOTSET:
                self.setLevel(root_logger.level)

    def _get_user_min_level(self) -> int:
        """Get the minimum log level from environment variable."""
        level_str = os.environ.get("ALLIGATOR_MIN_LOG_LEVEL", "INFO").upper()
        try:
            return getattr(LogLevel, level_str).value
        except AttributeError:
            return LogLevel.INFO.value

    def _log(self, level, msg, args, **kwargs):
        """Override the internal _log method to implement level filtering."""
        if level >= self._user_min_level:
            super()._log(level, msg, args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """Log with explicit level (supports both int and LogLevel enum)."""
        if isinstance(level, LogLevel):
            level = level.value
        if level >= self._user_min_level:
            super().log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log a debug message."""
        if LogLevel.DEBUG >= self._user_min_level:
            super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log an info message."""
        if LogLevel.INFO >= self._user_min_level:
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message."""
        if LogLevel.WARNING >= self._user_min_level:
            super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message."""
        if LogLevel.ERROR >= self._user_min_level:
            super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log a critical message."""
        if LogLevel.CRITICAL >= self._user_min_level:
            super().critical(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """Log an exception message."""
        if LogLevel.ERROR >= self._user_min_level:
            super().exception(msg, *args, exc_info=exc_info, **kwargs)

    def __getstate__(self):
        """Support for pickling."""
        return {"name": self.name}

    def __setstate__(self, state):
        """Support for unpickling."""
        self.__init__(state["name"])

    def __reduce__(self):
        """Support for multiprocessing pickling."""
        return (ConditionalLogger, (self.name,))


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
    return get_logger("alligator")


def get_logger(name: str):
    """
    Get a logger for a specific component.

    Args:
        name: Name of the component (e.g., 'coordinator', 'data_manager')

    Returns:
        ConditionalLogger instance for the component (or SilentLogger if disabled)
    """
    # Check if logging is disabled via environment variable (for worker processes)
    if os.environ.get("ALLIGATOR_DISABLE_LOGGING") == "1":
        # Ensure logging is disabled in this process too
        logging.disable(logging.CRITICAL)
        return SilentLogger(f"alligator.{name}")

    # Return a ConditionalLogger instance
    return ConditionalLogger(f"alligator.{name}")


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
