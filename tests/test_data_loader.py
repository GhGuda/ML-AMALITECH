"""Unit tests for resilient dataset loading and schema validation."""

from __future__ import annotations

import logging
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from src.flight_fare.data_loader import (
    DataLoadError,
    DataLoadTimeoutError,
    SchemaValidationError,
    load_flight_dataset,
    validate_dataset_schema,
)


def _build_logger() -> logging.Logger:
    """Create a lightweight test logger."""
    logger = logging.getLogger("tests.data_loader")
    logger.handlers.clear()
    logger.setLevel("INFO")
    logger.addHandler(logging.NullHandler())
    return logger


def _sample_dataframe() -> pd.DataFrame:
    """Create a valid dataframe that satisfies required schema expectations."""
    return pd.DataFrame(
        {
            "Airline": ["Airline A"],
            "Source": ["Dhaka"],
            "Destination": ["Chittagong"],
            "Departure Date & Time": ["2024-01-01 10:00"],
            "Base Fare (BDT)": [1000.0],
            "Tax & Surcharge (BDT)": [200.0],
            "Total Fare (BDT)": [1200.0],
        }
    )


class DataLoaderTests(unittest.TestCase):
    """Test suite for data loader robustness guarantees."""

    def setUp(self) -> None:
        """Create temporary workspace and baseline test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.logger = _build_logger()

    def tearDown(self) -> None:
        """Clean up temporary resources after each test."""
        self.temp_dir.cleanup()

    def test_load_flight_dataset_success(self) -> None:
        """Load succeeds for a valid local CSV file."""
        csv_path = self.base_path / "valid.csv"
        _sample_dataframe().to_csv(csv_path, index=False)

        loaded = load_flight_dataset(csv_path=csv_path, logger=self.logger, retries=1, timeout_seconds=5)
        self.assertEqual(len(loaded), 1)
        self.assertIn("Total Fare (BDT)", loaded.columns)

    def test_validate_dataset_schema_raises_on_missing_columns(self) -> None:
        """Schema validation raises a clear error when required columns are missing."""
        dataframe = _sample_dataframe().drop(columns=["Destination"])
        with self.assertRaises(SchemaValidationError):
            validate_dataset_schema(dataframe)

    def test_validate_dataset_schema_allows_missing_fare_components(self) -> None:
        """Schema validation should allow datasets that omit base/tax fare components."""
        dataframe = _sample_dataframe().drop(columns=["Base Fare (BDT)", "Tax & Surcharge (BDT)"])
        validate_dataset_schema(dataframe)

    def test_load_flight_dataset_retries_then_succeeds(self) -> None:
        """Loader retries on transient failures and then returns data."""
        call_counter = {"count": 0}

        def flaky_reader(_: str) -> pd.DataFrame:
            """Fail once before returning valid data."""
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                raise OSError("Transient read failure")
            return _sample_dataframe()

        loaded = load_flight_dataset(
            csv_path=self.base_path / "unused.csv",
            logger=self.logger,
            retries=2,
            timeout_seconds=5,
            retry_delay_seconds=0.01,
            read_csv_func=flaky_reader,
        )
        self.assertEqual(call_counter["count"], 2)
        self.assertEqual(loaded.iloc[0]["Airline"], "Airline A")

    def test_load_flight_dataset_timeout(self) -> None:
        """Loader raises timeout-specific error when read exceeds timeout budget."""

        def slow_reader(_: str) -> pd.DataFrame:
            """Simulate a slow CSV backend."""
            time.sleep(0.2)
            return _sample_dataframe()

        with self.assertRaises(DataLoadError) as context:
            load_flight_dataset(
                csv_path=self.base_path / "unused.csv",
                logger=self.logger,
                retries=1,
                timeout_seconds=0.01,
                read_csv_func=slow_reader,
            )
        self.assertIsInstance(context.exception.__cause__, DataLoadTimeoutError)


if __name__ == "__main__":
    unittest.main()
