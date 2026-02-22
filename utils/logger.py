"""Logging utilities for the MATCH-A framework.

This module provides a configurable logging setup with support for
different log levels, formats, and output destinations.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "matcha",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with the specified configuration.
    
    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file.
        format_string: Custom format string for log messages.
        
    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name.
    
    Args:
        name: Logger name (should include module path).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(f"matcha.{name}")


class LoggingMixin:
    """Mixin class that provides logging functionality.
    
    This mixin adds a `logger` property that returns a logger
    for the class it is mixed into.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this class.
        
        Returns:
            Logger instance for the class.
        """
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)
