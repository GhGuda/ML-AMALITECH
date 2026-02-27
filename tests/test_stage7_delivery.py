"""Integration tests for Stage 7 packaging and delivery pipeline."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.flight_fare.preprocessing import create_preprocessor
from src.flight_fare.settings import Stage7Settings
from src.flight_fare.stage7_delivery import predict_from_package, run_stage_7


def _build_cleaned_dataset(rows: int = 120) -> pd.DataFrame:
    """Create synthetic cleaned dataset that matches Stage 2 expected schema."""
    records: list[dict] = []
    for index in range(rows):
        records.append(
            {
                "Airline": f"Airline {index % 3}",
                "Source": "DAC" if index % 2 == 0 else "CGP",
                "Source Name": "Dhaka" if index % 2 == 0 else "Chattogram",
                "Destination": "CXB" if index % 2 == 0 else "DAC",
                "Destination Name": "Coxs Bazar" if index % 2 == 0 else "Dhaka",
                "Departure Date & Time": f"2025-07-{(index % 28) + 1:02d} 09:10:00",
                "Arrival Date & Time": f"2025-07-{(index % 28) + 1:02d} 10:30:00",
                "Duration (hrs)": 1.2 + (index % 3) * 0.2,
                "Stopovers": "Direct" if index % 2 == 0 else "1 Stop",
                "Aircraft Type": "A320" if index % 2 == 0 else "B737",
                "Class": "Economy",
                "Booking Source": "Web" if index % 2 == 0 else "App",
                "Base Fare (BDT)": 1000 + (index * 12),
                "Tax & Surcharge (BDT)": 200 + (index * 4),
                "Total Fare (BDT)": 1200 + (index * 16),
                "Seasonality": "Peak" if index % 4 == 0 else "Regular",
                "Days Before Departure": 5 + (index % 10),
                "Departure Month": ((index % 12) + 1),
                "Departure Day": ((index % 28) + 1),
                "Departure Weekday": (index % 7),
                "Departure Season": "monsoon",
            }
        )
    return pd.DataFrame(records)


def _prepare_model_and_preprocessor(cleaned_df: pd.DataFrame) -> tuple[object, object, list[str], list[str]]:
    """Train synthetic preprocessor+model pair for delivery package testing."""
    numeric_cols = [
        "Duration (hrs)",
        "Base Fare (BDT)",
        "Tax & Surcharge (BDT)",
        "Days Before Departure",
        "Departure Month",
        "Departure Day",
        "Departure Weekday",
    ]
    categorical_cols = [
        "Airline",
        "Source",
        "Source Name",
        "Destination",
        "Destination Name",
        "Stopovers",
        "Aircraft Type",
        "Class",
        "Booking Source",
        "Seasonality",
        "Departure Season",
    ]

    preprocessor = create_preprocessor(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        numeric_impute_strategy="median",
        categorical_impute_strategy="most_frequent",
        scaler_strategy="standard",
    )
    x_train = cleaned_df[numeric_cols + categorical_cols]
    y_train = cleaned_df["Total Fare (BDT)"]
    transformed = preprocessor.fit_transform(x_train)

    model = RandomForestRegressor(n_estimators=40, random_state=42)
    model.fit(transformed, y_train)
    return model, preprocessor, numeric_cols, categorical_cols


class Stage7DeliveryTests(unittest.TestCase):
    """Validate Stage 7 package assembly and packaged inference execution."""

    def test_run_stage_7_and_predict_from_package(self) -> None:
        """Stage 7 should produce package artifacts and support packaged predictions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cleaned_df = _build_cleaned_dataset(rows=150)
            model, preprocessor, numeric_cols, categorical_cols = _prepare_model_and_preprocessor(cleaned_df)

            stage2_dir = root / "artifacts" / "stage_2"
            stage5_dir = root / "artifacts" / "stage_5"
            stage2_dir.mkdir(parents=True, exist_ok=True)
            stage5_dir.mkdir(parents=True, exist_ok=True)

            cleaned_path = stage2_dir / "cleaned_dataset.csv"
            preprocessor_path = stage2_dir / "preprocessor.pkl"
            preprocessing_report_path = stage2_dir / "preprocessing_report.json"
            best_model_path = stage5_dir / "best_model.pkl"
            stage5_summary_path = stage5_dir / "stage5_summary_report.json"

            cleaned_df.to_csv(cleaned_path, index=False)
            preprocessor_path.write_bytes(pickle.dumps(preprocessor))
            best_model_path.write_bytes(pickle.dumps(model))

            preprocessing_report_path.write_text(
                json.dumps(
                    {
                        "stage": 2,
                        "preprocessing": {
                            "numeric_feature_columns": numeric_cols,
                            "categorical_feature_columns": categorical_cols,
                        },
                    }
                ),
                encoding="utf-8",
            )
            stage5_summary_path.write_text(
                json.dumps(
                    {
                        "stage": 5,
                        "best_model": {
                            "model_name": "RandomForestRegressor",
                            "test_metrics": {"rmse": 1200.0},
                        },
                    }
                ),
                encoding="utf-8",
            )

            settings = Stage7Settings(
                project_root=root,
                best_model_path=best_model_path,
                preprocessor_path=preprocessor_path,
                stage5_summary_path=stage5_summary_path,
                preprocessing_report_path=preprocessing_report_path,
                cleaned_dataset_path=cleaned_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
                package_version="v_test",
                smoke_test_rows=10,
            )

            outputs = run_stage_7(
                settings=settings,
                retries=1,
                timeout_seconds=10.0,
                io_retries=1,
                io_timeout_seconds=10.0,
            )

            self.assertTrue(outputs.package_dir.exists())
            self.assertTrue(outputs.manifest_path.exists())
            self.assertTrue(outputs.schema_path.exists())
            self.assertTrue(outputs.smoke_test_predictions_path.exists())
            self.assertTrue(outputs.package_readme_path.exists())
            self.assertTrue(outputs.summary_report_path.exists())

            predict_input_df = cleaned_df.head(8).copy()
            predictions_df = predict_from_package(
                package_dir=outputs.package_dir,
                input_df=predict_input_df,
                retries=1,
                timeout_seconds=10.0,
            )
            self.assertEqual(len(predictions_df), 8)
            self.assertIn("predicted_total_fare_bdt", predictions_df.columns)
            self.assertIn("package_version", predictions_df.columns)


if __name__ == "__main__":
    unittest.main()
