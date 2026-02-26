"""Integration-style tests for Stage 1 pipeline execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.flight_fare.settings import Stage1Settings
from src.flight_fare.stage1_pipeline import run_stage_1


def _valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid dataset for pipeline testing."""
    return pd.DataFrame(
        {
            "Airline": ["Airline A", "Airline B"],
            "Source": ["Dhaka", "Dhaka"],
            "Destination": ["Chittagong", "Sylhet"],
            "Departure Date & Time": ["2024-01-01 10:00", "2024-01-02 18:45"],
            "Base Fare (BDT)": [1000.0, 1500.0],
            "Tax & Surcharge (BDT)": [200.0, 350.0],
            "Total Fare (BDT)": [1200.0, 1850.0],
        }
    )


class Stage1PipelineTests(unittest.TestCase):
    """Validate end-to-end artifact creation for Stage 1."""

    def test_run_stage_1_generates_artifacts(self) -> None:
        """Stage 1 run should write profile and problem definition artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "flight_data.csv"
            _valid_dataframe().to_csv(dataset_path, index=False)

            settings = Stage1Settings(
                project_root=root,
                dataset_path=dataset_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
            )

            outputs = run_stage_1(settings=settings, retries=1, timeout_seconds=5)

            self.assertTrue(outputs.profile_path.exists())
            self.assertTrue(outputs.problem_definition_path.exists())

            profile_data = json.loads(outputs.profile_path.read_text(encoding="utf-8"))
            self.assertEqual(profile_data["shape"]["rows"], 2)
            self.assertIn("outlier_flags_iqr", profile_data)

            log_file = settings.logs_dir / settings.log_file_name
            self.assertTrue(log_file.exists())


if __name__ == "__main__":
    unittest.main()
