"""Integration tests for local REST API server."""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from src.flight_fare.api_server import create_app
from src.flight_fare.logging_utils import configure_logging, shutdown_logger_handlers
from src.flight_fare.preprocessing import create_preprocessor
from src.flight_fare.settings import ApiSettings


def _build_training_frame(rows: int = 80) -> pd.DataFrame:
    """Create synthetic dataframe aligned with Stage 2 feature expectations."""
    records: list[dict] = []
    for index in range(rows):
        records.append(
            {
                "Airline": f"Airline {index % 3}",
                "Source": "DAC" if index % 2 == 0 else "CGP",
                "Source Name": "Dhaka" if index % 2 == 0 else "Chattogram",
                "Destination": "CXB" if index % 2 == 0 else "DAC",
                "Destination Name": "Coxs Bazar" if index % 2 == 0 else "Dhaka",
                "Departure Date & Time": f"2025-08-{(index % 28) + 1:02d} 07:35:00",
                "Arrival Date & Time": f"2025-08-{(index % 28) + 1:02d} 09:10:00",
                "Duration (hrs)": 1.1 + (index % 3) * 0.2,
                "Stopovers": "Direct" if index % 2 == 0 else "1 Stop",
                "Aircraft Type": "A320" if index % 2 == 0 else "B737",
                "Class": "Economy",
                "Booking Source": "Web" if index % 2 == 0 else "App",
                "Total Fare (BDT)": 1200 + (index * 14),
                "Seasonality": "Regular",
                "Days Before Departure": 4 + (index % 12),
                "Departure Month": ((index % 12) + 1),
                "Departure Day": ((index % 28) + 1),
                "Departure Weekday": (index % 7),
                "Departure Season": "monsoon",
            }
        )
    return pd.DataFrame(records)


def _build_test_package(package_dir: Path) -> tuple[list[str], pd.DataFrame]:
    """Create a minimal packaged model directory suitable for API testing."""
    package_dir.mkdir(parents=True, exist_ok=True)
    training_df = _build_training_frame(rows=100)

    numeric_cols = [
        "Duration (hrs)",
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
    all_cols = numeric_cols + categorical_cols

    preprocessor = create_preprocessor(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        numeric_impute_strategy="median",
        categorical_impute_strategy="most_frequent",
        scaler_strategy="standard",
    )
    transformed = preprocessor.fit_transform(training_df[all_cols])
    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(transformed, training_df["Total Fare (BDT)"])

    (package_dir / "best_model.pkl").write_bytes(pickle.dumps(model))
    (package_dir / "preprocessor.pkl").write_bytes(pickle.dumps(preprocessor))
    (package_dir / "model_manifest.json").write_text(
        json.dumps({"package_version": "v_test", "model": {"model_name": "RandomForestRegressor"}}),
        encoding="utf-8",
    )
    (package_dir / "input_schema.json").write_text(
        json.dumps({"required_fields": [{"name": col, "type": "numeric" if col in numeric_cols else "categorical"} for col in all_cols]}),
        encoding="utf-8",
    )
    return all_cols, training_df


class ApiServerTests(unittest.TestCase):
    """Validate API health and prediction endpoints."""

    def test_health_and_predict_endpoints(self) -> None:
        """API should return health status and fare predictions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_dir = root / "package"
            _, sample_df = _build_test_package(package_dir)

            settings = ApiSettings(
                project_root=root,
                package_dir=package_dir,
                logs_dir=root / "logs",
                host="127.0.0.1",
                port=0,
                request_timeout_seconds=10.0,
                predict_retries=1,
                predict_timeout_seconds=10.0,
                predict_retry_delay_seconds=0.1,
            )
            logger = configure_logging(settings, logger_name="tests.api_server")
            try:
                app = create_app(settings, logger=logger)
                client = TestClient(app)

                health_response = client.get("/health")
                self.assertEqual(health_response.status_code, 200)
                payload = health_response.json()
                self.assertEqual(payload["status"], "ok")

                schema_response = client.get("/schema")
                self.assertEqual(schema_response.status_code, 200)
                schema_payload = schema_response.json()
                self.assertIn("required_fields", schema_payload)

                records = sample_df.head(5).to_dict(orient="records")
                predict_response = client.post("/predict", json={"records": records})
                self.assertEqual(predict_response.status_code, 200)
                predict_payload = predict_response.json()
                self.assertEqual(predict_payload["count"], 5)
                self.assertEqual(len(predict_payload["predictions"]), 5)
                self.assertIn("predicted_total_fare_bdt", predict_payload["predictions"][0])

                single_response = client.post("/predict", json={"record": records[0]})
                self.assertEqual(single_response.status_code, 200)
                single_payload = single_response.json()
                self.assertEqual(single_payload["count"], 1)
                self.assertEqual(len(single_payload["predictions"]), 1)

                bad_response = client.post("/predict", json={"records": "not-a-list"})
                self.assertEqual(bad_response.status_code, 400)

                both_keys_response = client.post("/predict", json={"record": records[0], "records": records})
                self.assertEqual(both_keys_response.status_code, 400)
            finally:
                shutdown_logger_handlers(logger)


if __name__ == "__main__":
    unittest.main()
