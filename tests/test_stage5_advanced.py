"""Integration tests for Stage 5 advanced modeling pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.flight_fare.settings import Stage5Settings
from src.flight_fare.stage5_advanced import run_stage_5


def _build_stage5_splits(
    train_rows: int = 180,
    test_rows: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create synthetic splits with linear and non-linear signal components."""
    rng = np.random.default_rng(seed=123)
    total_rows = train_rows + test_rows

    x1 = rng.normal(0, 1, total_rows)
    x2 = rng.normal(4, 2, total_rows)
    x3 = rng.uniform(-2, 2, total_rows)
    x4 = rng.normal(1, 0.5, total_rows)
    noise = rng.normal(0, 0.4, total_rows)

    y = 950 + (220 * x1) + (75 * x2) - (40 * x3) + (18 * x4**2) + noise

    x = pd.DataFrame(
        {
            "num__x1": x1,
            "num__x2": x2,
            "num__x3": x3,
            "num__x4": x4,
        }
    )
    y_df = pd.DataFrame({"Total Fare (BDT)": y})

    x_train = x.iloc[:train_rows].reset_index(drop=True)
    x_test = x.iloc[train_rows:].reset_index(drop=True)
    y_train = y_df.iloc[:train_rows].reset_index(drop=True)
    y_test = y_df.iloc[train_rows:].reset_index(drop=True)
    return x_train, x_test, y_train, y_test


class Stage5AdvancedTests(unittest.TestCase):
    """Validate Stage 5 artifact generation and comparison outputs."""

    def test_run_stage_5_generates_expected_artifacts(self) -> None:
        """Stage 5 should generate model comparison, best model, and tuning artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            x_train, x_test, y_train, y_test = _build_stage5_splits()

            x_train_path = root / "x_train.csv"
            x_test_path = root / "x_test.csv"
            y_train_path = root / "y_train.csv"
            y_test_path = root / "y_test.csv"
            x_train.to_csv(x_train_path, index=False)
            x_test.to_csv(x_test_path, index=False)
            y_train.to_csv(y_train_path, index=False)
            y_test.to_csv(y_test_path, index=False)

            settings = Stage5Settings(
                project_root=root,
                x_train_path=x_train_path,
                x_test_path=x_test_path,
                y_train_path=y_train_path,
                y_test_path=y_test_path,
                logs_dir=root / "logs",
                artifacts_dir=root / "artifacts",
                cv_folds=2,
                tuning_sample_fraction=0.8,
                tree_search_iterations=3,
                forest_search_iterations=2,
                gb_search_iterations=2,
                n_jobs=1,
            )

            outputs = run_stage_5(
                settings=settings,
                retries=1,
                timeout_seconds=30.0,
                io_retries=1,
                io_timeout_seconds=15.0,
            )

            self.assertTrue(outputs.comparison_table_path.exists())
            self.assertTrue(outputs.tuning_results_path.exists())
            self.assertTrue(outputs.best_model_path.exists())
            self.assertTrue(outputs.best_model_predictions_path.exists())
            self.assertTrue(outputs.summary_report_path.exists())
            self.assertTrue(outputs.regularization_effect_path.exists())
            self.assertTrue(outputs.bias_variance_path.exists())
            self.assertTrue(outputs.best_model_importance_path.exists())
            self.assertTrue(outputs.plots_directory.exists())

            comparison = pd.read_csv(outputs.comparison_table_path)
            self.assertGreaterEqual(len(comparison), 5)
            self.assertIn("test_rmse", comparison.columns)

            summary = json.loads(outputs.summary_report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["stage"], 5)
            self.assertIn("best_model", summary)


if __name__ == "__main__":
    unittest.main()
