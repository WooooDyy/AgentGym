"""
Centralized logging configuration for agentenv-mcp.

Provides consistent logging across all modules with configurable levels and formats.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the agentenv-mcp package.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        log_file: Path to log file (optional, logs to stderr by default)

    Returns:
        Root logger for agentenv_mcp
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get the root logger for our package
    logger = logging.getLogger("agentenv_mcp")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with agentenv_mcp)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"agentenv_mcp.{name}")


# Initialize default logging
_root_logger = setup_logging()
