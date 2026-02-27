"""Integration tests for Stage 6 interpretation and insight pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.flight_fare.settings import Stage6Settings
from src.flight_fare.stage6_interpretation import run_stage_6


def _build_cleaned_dataset(rows: int = 120) -> pd.DataFrame:
    """Create a synthetic cleaned dataset compatible with Stage 6 insights."""
    records: list[dict] = []
    seasons = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
    for index in range(rows):
        source = "DAC" if index % 2 == 0 else "CGP"
        destination = "CXB" if index % 3 == 0 else "DAC"
        season = seasons[index % len(seasons)]
        records.append(
            {
                "Airline": f"Airline {index % 4}",
                "Source": source,
                "Destination": destination,
                "Departure Season": season,
                "Departure Month": (index % 12) + 1,
                "Total Fare (BDT)": 1800 + (index % 7) * 250 + (index % 4) * 120,
            }
        )
    return pd.DataFrame(records)


def _build_feature_impact() -> pd.DataFrame:
    """Create synthetic feature-impact table similar to Stage 5 output schema."""
    return pd.DataFrame(
        {
            "feature_name": [
                "num__Days Before Departure",
                "cat__Class_Business",
                "num__Duration (hrs)",
                "cat__Stopovers_1 Stop",
            ],
            "importance": [0.52, 0.23, 0.15, 0.10],
            "importance_type": ["feature_importance"] * 4,
            "abs_importance": [0.52, 0.23, 0.15, 0.10],
        }
    )


def _build_stage5_summary() -> dict:
    """Create synthetic Stage 5 summary payload used by Stage 6 report context."""
    return {
        "stage": 5,
        "best_model": {
            "model_name": "RandomForestRegressor",
            "test_metrics": {"rmse": 950.4},
            "train_metrics": {"rmse": 500.2},
            "best_params": {"n_estimators": 80},
        },
    }


class Stage6InterpretationTests(unittest.TestCase):
    """Validate Stage 6 artifact generation and summary integrity."""

    def test_run_stage_6_generates_expected_artifacts(self) -> None:
        """Stage 6 should generate interpretation tables, plots, and markdown report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cleaned_dataset_path = root / "cleaned_dataset.csv"
            feature_impact_path = root / "best_model_feature_impact.csv"
            stage5_summary_path = root / "stage5_summary_report.json"

            _build_cleaned_dataset(rows=160).to_csv(cleaned_dataset_path, index=False)
            _build_feature_impact().to_csv(feature_impact_path, index=False)
            stage5_summary_path.write_text(json.dumps(_build_stage5_summary()), encoding="utf-8")

            settings = Stage6Settings(
                project_root=root,
                cleaned_dataset_path=cleaned_dataset_path,
                stage5_summary_path=stage5_summary_path,
                feature_impact_path=feature_impact_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
                top_feature_count=3,
                top_route_season_count=5,
                min_route_season_flights=5,
            )

            outputs = run_stage_6(
                settings=settings,
                retries=1,
                timeout_seconds=10.0,
                io_retries=1,
                io_timeout_seconds=10.0,
            )

            self.assertTrue(outputs.summary_report_path.exists())
            self.assertTrue(outputs.stakeholder_report_path.exists())
            self.assertTrue(outputs.top_feature_table_path.exists())
            self.assertTrue(outputs.airline_fare_summary_path.exists())
            self.assertTrue(outputs.seasonal_pricing_path.exists())
            self.assertTrue(outputs.route_season_hotspots_path.exists())
            self.assertTrue(outputs.plots_directory.exists())

            summary = json.loads(outputs.summary_report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["stage"], 6)
            self.assertEqual(summary["best_model_name"], "RandomForestRegressor")
            self.assertIsNotNone(summary["top_driver"])

            top_features = pd.read_csv(outputs.top_feature_table_path)
            self.assertEqual(len(top_features), 3)
            self.assertIn("importance_pct", top_features.columns)
            self.assertNotIn("Base Fare (BDT)", top_features["readable_feature"].tolist())
            self.assertNotIn("Tax & Surcharge (BDT)", top_features["readable_feature"].tolist())


if __name__ == "__main__":
    unittest.main()
