"""Resilient dataset loading and schema validation utilities."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

ReadCsvFunction = Callable[..., pd.DataFrame]

REQUIRED_COLUMNS = {
    "Airline",
    "Source",
    "Destination",
    "Departure Date & Time",
    "Total Fare (BDT)",
}


class DataLoadError(RuntimeError):
    """Raised when dataset loading fails."""


class DataLoadTimeoutError(DataLoadError):
    """Raised when dataset loading exceeds the configured timeout."""


class SchemaValidationError(ValueError):
    """Raised when required dataset columns are missing."""


def read_csv_with_timeout(
    csv_path: Path,
    timeout_seconds: float,
    read_csv_func: ReadCsvFunction,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """Read a CSV file with timeout enforcement using a worker thread."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read_csv_func, str(csv_path), **read_csv_kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            message = f"Timed out while reading dataset after {timeout_seconds} seconds."
            raise DataLoadTimeoutError(message) from exc
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to read dataset at '{csv_path}'."
            raise DataLoadError(message) from exc


def validate_dataset_schema(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """Validate that the input dataframe contains all required columns."""
    columns_to_check = set(required_columns or REQUIRED_COLUMNS)
    missing_columns = columns_to_check.difference(dataframe.columns)
    if missing_columns:
        missing_sorted = ", ".join(sorted(missing_columns))
        raise SchemaValidationError(f"Dataset is missing required columns: {missing_sorted}")


def load_flight_dataset(
    csv_path: Path,
    logger: logging.Logger,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 1.5,
    read_csv_func: ReadCsvFunction = pd.read_csv,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """Load and validate flight fare data with retries and timeout protection."""
    if retries < 1:
        raise ValueError("Parameter 'retries' must be at least 1.")
    if timeout_seconds <= 0:
        raise ValueError("Parameter 'timeout_seconds' must be greater than 0.")

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.info("Loading dataset from %s (attempt %s/%s).", csv_path, attempt, retries)
            dataframe = read_csv_with_timeout(
                csv_path=csv_path,
                timeout_seconds=timeout_seconds,
                read_csv_func=read_csv_func,
                **read_csv_kwargs,
            )
            validate_dataset_schema(dataframe)
            logger.info("Dataset loaded successfully with %s rows.", len(dataframe))
            return dataframe
        except (DataLoadError, SchemaValidationError) as exc:
            last_error = exc
            should_retry = attempt < retries
            logger.warning("Dataset load attempt %s failed: %s", attempt, exc)
            if should_retry:
                # Delay briefly to allow transient I/O issues to recover.
                time.sleep(retry_delay_seconds)

    raise DataLoadError(f"Dataset loading failed after {retries} attempts.") from last_error
