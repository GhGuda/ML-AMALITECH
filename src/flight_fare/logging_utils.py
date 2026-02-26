"""Logging utilities for production-grade file and console logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .settings import Stage1Settings


def _log_file_path(settings: Stage1Settings) -> Path:
    """Return the fully resolved path for the stage log file."""
    return settings.logs_dir / settings.log_file_name


def configure_logging(settings: Stage1Settings, logger_name: str = "flight_fare") -> logging.Logger:
    """Configure and return a logger with rotating file and console handlers."""
    settings.ensure_directories()
    logger = logging.getLogger(logger_name)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    # Reset handlers to avoid duplicate log lines in repeated runs.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=str(_log_file_path(settings)),
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(settings.log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(settings.log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
