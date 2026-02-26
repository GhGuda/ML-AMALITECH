"""Integration tests for Stage 4 baseline linear regression pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.flight_fare.settings import Stage4Settings
from src.flight_fare.stage4_baseline import run_stage_4


def _build_stage4_splits(
    train_rows: int = 120,
    test_rows: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create synthetic linear dataset splits for deterministic baseline validation."""
    rng = np.random.default_rng(seed=42)
    total_rows = train_rows + test_rows

    feature_1 = rng.normal(loc=0.0, scale=1.0, size=total_rows)
    feature_2 = rng.normal(loc=5.0, scale=2.0, size=total_rows)
    feature_3 = rng.normal(loc=-2.0, scale=0.5, size=total_rows)
    noise = rng.normal(loc=0.0, scale=0.3, size=total_rows)

    target = 1200 + (300 * feature_1) + (90 * feature_2) - (40 * feature_3) + noise

    x = pd.DataFrame(
        {
            "num__feature_1": feature_1,
            "num__feature_2": feature_2,
            "num__feature_3": feature_3,
        }
    )
    y = pd.DataFrame({"Total Fare (BDT)": target})

    x_train = x.iloc[:train_rows].reset_index(drop=True)
    x_test = x.iloc[train_rows:].reset_index(drop=True)
    y_train = y.iloc[:train_rows].reset_index(drop=True)
    y_test = y.iloc[train_rows:].reset_index(drop=True)
    return x_train, x_test, y_train, y_test


class Stage4BaselineTests(unittest.TestCase):
    """Validate Stage 4 artifact generation and baseline metric sanity."""

    def test_run_stage_4_generates_expected_artifacts(self) -> None:
        """Stage 4 run should output model, metrics, predictions, and diagnostics plots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_train, x_test, y_train, y_test = _build_stage4_splits()

            x_train_path = root / "x_train.csv"
            x_test_path = root / "x_test.csv"
            y_train_path = root / "y_train.csv"
            y_test_path = root / "y_test.csv"
            x_train.to_csv(x_train_path, index=False)
            x_test.to_csv(x_test_path, index=False)
            y_train.to_csv(y_train_path, index=False)
            y_test.to_csv(y_test_path, index=False)

            settings = Stage4Settings(
                project_root=root,
                x_train_path=x_train_path,
                x_test_path=x_test_path,
                y_train_path=y_train_path,
                y_test_path=y_test_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
            )

            outputs = run_stage_4(
                settings=settings,
                retries=1,
                timeout_seconds=10.0,
                io_retries=1,
                io_timeout_seconds=10.0,
            )

            self.assertTrue(outputs.metrics_path.exists())
            self.assertTrue(outputs.predictions_path.exists())
            self.assertTrue(outputs.model_path.exists())
            self.assertTrue(outputs.summary_path.exists())
            self.assertTrue(outputs.coefficients_path.exists())
            self.assertTrue(outputs.plots_directory.exists())

            metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("test_metrics", metrics)
            self.assertGreater(metrics["test_metrics"]["r2"], 0.99)
            self.assertLess(metrics["test_metrics"]["mae"], 1.0)

            predictions = pd.read_csv(outputs.predictions_path)
            self.assertEqual(len(predictions), len(y_test))
            self.assertIn("residual", predictions.columns)


if __name__ == "__main__":
    unittest.main()
