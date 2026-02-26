"""Integration and unit tests for Stage 3 EDA pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.flight_fare.settings import Stage3Settings
from src.flight_fare.stage3_eda import compute_kpi_tables, prepare_eda_dataframe, run_stage_3


def build_stage3_dataset(rows: int = 40) -> pd.DataFrame:
    """Create a schema-valid synthetic dataset suitable for Stage 3 execution."""
    records: list[dict] = []
    for index in range(rows):
        source = "DAC" if index % 3 != 0 else "CGP"
        destination = "CXB" if index % 2 == 0 else "DAC"
        records.append(
            {
                "Airline": f"Airline {index % 4}",
                "Source": source,
                "Source Name": "Dhaka" if source == "DAC" else "Chittagong",
                "Destination": destination,
                "Destination Name": "Coxs Bazar" if destination == "CXB" else "Dhaka",
                "Departure Date & Time": f"2025-08-{(index % 28) + 1:02d} 08:15:00",
                "Arrival Date & Time": f"2025-08-{(index % 28) + 1:02d} 10:20:00",
                "Duration (hrs)": 1.1 + (index % 3) * 0.2,
                "Stopovers": str(index % 2),
                "Aircraft Type": "A320" if index % 2 == 0 else "B737",
                "Class": "Economy",
                "Booking Source": "Web" if index % 2 == 0 else "App",
                "Base Fare (BDT)": 1000 + (index * 10),
                "Tax & Surcharge (BDT)": 200 + (index * 5),
                "Total Fare (BDT)": 1200 + (index * 15),
                "Seasonality": "Monsoon",
                "Days Before Departure": 5 + (index % 12),
            }
        )
    return pd.DataFrame(records)


class Stage3EdaTests(unittest.TestCase):
    """Validate KPI logic and end-to-end Stage 3 artifact generation."""

    def test_compute_kpi_tables(self) -> None:
        """KPI tables should include non-empty route frequency and top route output."""
        dataframe = prepare_eda_dataframe(build_stage3_dataset(rows=30))
        kpis = compute_kpi_tables(dataframe, top_n_routes=5)

        self.assertIn("route_frequency", kpis)
        self.assertIn("top_expensive_routes", kpis)
        self.assertGreater(len(kpis["route_frequency"]), 0)
        self.assertGreater(len(kpis["top_expensive_routes"]), 0)

    def test_run_stage_3_generates_expected_artifacts(self) -> None:
        """Stage 3 run should produce summary, KPI tables, and plot artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "stage3_data.csv"
            build_stage3_dataset(rows=45).to_csv(dataset_path, index=False)

            settings = Stage3Settings(
                project_root=root,
                dataset_path=dataset_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
                top_n_routes=5,
                max_airlines_boxplot=10,
            )

            outputs = run_stage_3(
                settings=settings,
                retries=1,
                timeout_seconds=10.0,
                io_retries=1,
                io_timeout_seconds=20.0,
            )

            self.assertTrue(outputs.summary_report_path.exists())
            self.assertTrue(outputs.descriptive_stats_path.exists())
            self.assertTrue(outputs.correlation_matrix_path.exists())
            self.assertTrue(outputs.average_fare_by_airline_path.exists())
            self.assertTrue(outputs.route_frequency_path.exists())
            self.assertTrue(outputs.seasonal_fare_path.exists())
            self.assertTrue(outputs.top_expensive_routes_path.exists())
            self.assertTrue(outputs.plots_directory.exists())

            summary = json.loads(outputs.summary_report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["stage"], 3)
            self.assertEqual(summary["dataset_rows"], 45)
            self.assertIn("most_popular_route", summary["kpis"])


if __name__ == "__main__":
    unittest.main()
