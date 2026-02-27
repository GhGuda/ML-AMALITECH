"""Integration tests for Stage 2 pipeline execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.flight_fare.settings import Stage2Settings
from src.flight_fare.stage2_pipeline import run_stage_2


def build_stage2_dataset(rows: int = 20) -> pd.DataFrame:
    """Create a synthetic but schema-valid dataset for Stage 2 tests."""
    records: list[dict] = []
    for index in range(rows):
        records.append(
            {
                "Airline": f"Airline {index % 3}",
                "Source": "DAC" if index % 2 == 0 else "CGP",
                "Source Name": "Dhaka" if index % 2 == 0 else "Chittagong",
                "Destination": "CXB" if index % 2 == 0 else "DAC",
                "Destination Name": "Coxs Bazar" if index % 2 == 0 else "Dacca",
                "Departure Date & Time": f"2025-05-{(index % 28) + 1:02d} 10:15:00",
                "Arrival Date & Time": f"2025-05-{(index % 28) + 1:02d} 11:25:00",
                "Duration (hrs)": 1.1 + (index % 3) * 0.2,
                "Stopovers": str(index % 2),
                "Aircraft Type": "A320" if index % 2 == 0 else "B737",
                "Class": "Economy",
                "Booking Source": "Web" if index % 2 == 0 else "App",
                "Base Fare (BDT)": 1000 + (index * 15),
                "Tax & Surcharge (BDT)": 200 + (index * 4),
                "Total Fare (BDT)": 1200 + (index * 19),
                "Seasonality": "Summer",
                "Days Before Departure": 5 + (index % 10),
            }
        )
    return pd.DataFrame(records)


class Stage2PipelineTests(unittest.TestCase):
    """Validate end-to-end Stage 2 outputs and report integrity."""

    def test_run_stage_2_generates_expected_artifacts(self) -> None:
        """Stage 2 run should produce processed train/test data and report artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "flight_data.csv"
            build_stage2_dataset(rows=24).to_csv(dataset_path, index=False)

            settings = Stage2Settings(
                project_root=root,
                dataset_path=dataset_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
                test_size=0.25,
                random_state=42,
            )

            outputs = run_stage_2(
                settings=settings,
                retries=1,
                timeout_seconds=10.0,
                io_retries=1,
                io_timeout_seconds=10.0,
            )

            self.assertTrue(outputs.cleaned_dataset_path.exists())
            self.assertTrue(outputs.x_train_processed_path.exists())
            self.assertTrue(outputs.x_test_processed_path.exists())
            self.assertTrue(outputs.y_train_path.exists())
            self.assertTrue(outputs.y_test_path.exists())
            self.assertTrue(outputs.preprocessor_path.exists())
            self.assertTrue(outputs.report_path.exists())

            report = json.loads(outputs.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["stage"], 2)
            self.assertEqual(report["split"]["train_rows"] + report["split"]["test_rows"], 24)
            self.assertGreater(report["preprocessing"]["processed_feature_count"], 0)
            self.assertIn("Base Fare (BDT)", report["cleaning_report"]["dropped_leaky_columns"])
            self.assertIn("Tax & Surcharge (BDT)", report["cleaning_report"]["dropped_leaky_columns"])

            cleaned_df = pd.read_csv(outputs.cleaned_dataset_path)
            self.assertNotIn("Base Fare (BDT)", cleaned_df.columns)
            self.assertNotIn("Tax & Surcharge (BDT)", cleaned_df.columns)


if __name__ == "__main__":
    unittest.main()
