# -*- coding: utf-8 -*-
"""Logging configuration for the project."""

import logging
import sys
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    output_stream: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for a module.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        output_stream: Output stream - 'stderr' or 'stdout' (default: stderr for MCP servers, stdout for others)

    Returns:
        Configured logger instance
    """
    # Determine the output stream
    stream = sys.stderr if output_stream == "stderr" else sys.stdout

    # Create logger with specific name to avoid global configuration
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create handler for this specific logger only
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # Don't propagate to root logger

    return logger
