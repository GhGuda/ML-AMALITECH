"""Stage 5 pipeline for advanced model tuning, comparison, and optimization."""

from __future__ import annotations

import json
import pickle
import time

import joblib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from .logging_utils import configure_logging, shutdown_logger_handlers
from .settings import Stage5Settings

T = TypeVar("T")
SCORING = "neg_mean_squared_error"


class Stage5Error(RuntimeError):
    """Raised when Stage 5 execution fails."""


class Stage5TimeoutError(Stage5Error):
    """Raised when Stage 5 operation exceeds timeout."""


@dataclass(frozen=True)
class Stage5Outputs:
    """Artifacts produced by Stage 5 pipeline execution."""

    comparison_table_path: Path
    tuning_results_path: Path
    best_model_path: Path
    best_model_predictions_path: Path
    summary_report_path: Path
    regularization_effect_path: Path
    bias_variance_path: Path
    best_model_importance_path: Path
    plots_directory: Path


def json_default(value: Any) -> Any:
    """Convert non-JSON-native values into serializable primitives."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def run_with_retries_and_timeout(
    operation: Callable[[], T],
    operation_name: str,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> T:
    """Run an operation with retries and timeout enforcement."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0.")

    import logging

    logger = logging.getLogger(logger_name)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(operation)
                return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            last_error = Stage5TimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
            )
            logger.warning(
                "Operation '%s' timed out on attempt %s/%s.",
                operation_name,
                attempt,
                retries,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = Stage5Error(f"Operation '{operation_name}' failed: {exc}")
            logger.warning(
                "Operation '%s' failed on attempt %s/%s: %s",
                operation_name,
                attempt,
                retries,
                exc,
            )

        if attempt < retries:
            # Short delay to reduce transient I/O or compute contention.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise Stage5Error(f"Operation '{operation_name}' failed with unknown error.")
    raise last_error


def load_csv_with_retries(
    csv_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> pd.DataFrame:
    """Load CSV with retry and timeout handling."""
    return run_with_retries_and_timeout(
        operation=lambda: pd.read_csv(csv_path),
        operation_name=f"load_csv:{csv_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def select_target_column(dataframe: pd.DataFrame, target_column: str) -> pd.Series:
    """Select target column from dataframe with numeric coercion."""
    if target_column in dataframe.columns:
        return pd.to_numeric(dataframe[target_column], errors="coerce")

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric target column available.")
    return pd.to_numeric(dataframe[numeric_columns[0]], errors="coerce")


def validate_inputs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int,
    tuning_sample_fraction: float,
) -> None:
    """Validate model tuning inputs before expensive operations."""
    if x_train.empty or x_test.empty:
        raise ValueError("Feature inputs must not be empty.")
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train row counts must match.")
    if len(x_test) != len(y_test):
        raise ValueError("x_test and y_test row counts must match.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train and x_test must have matching feature counts.")
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Feature matrices contain missing values.")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Target vectors contain missing values.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if not 0 < tuning_sample_fraction <= 1:
        raise ValueError("tuning_sample_fraction must be within (0, 1].")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics for model evaluation."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mse**0.5),
    }


def save_json(payload: dict, output_path: Path) -> Path:
    """Persist dictionary payload as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return output_path


def save_dataframe_csv(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Persist dataframe as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_model_pickle(model: Any, output_path: Path) -> Path:
    """Persist trained model object as pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def sample_tuning_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample training data for efficient hyperparameter search."""
    if sample_fraction >= 1:
        return x_train, y_train
    sampled_indices = x_train.sample(frac=sample_fraction, random_state=random_state).index
    return x_train.loc[sampled_indices], y_train.loc[sampled_indices]


def tune_estimator(
    model_name: str,
    search: GridSearchCV | RandomizedSearchCV,
    x_tune: pd.DataFrame,
    y_tune: pd.Series,
    x_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    """Tune estimator, refit best parameters on full train set, and evaluate metrics."""
    fitted_search = run_with_retries_and_timeout(
        operation=lambda: search.fit(x_tune, y_tune),
        operation_name=f"tune_{model_name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )

    best_estimator = clone(fitted_search.best_estimator_)
    best_estimator = run_with_retries_and_timeout(
        operation=lambda: best_estimator.fit(x_train_full, y_train_full),
        operation_name=f"fit_best_{model_name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )

    train_pred = pd.Series(best_estimator.predict(x_train_full), index=y_train_full.index)
    test_pred = pd.Series(best_estimator.predict(x_test), index=y_test.index)
    train_metrics = compute_metrics(y_true=y_train_full, y_pred=train_pred)
    test_metrics = compute_metrics(y_true=y_test, y_pred=test_pred)

    return {
        "model_name": model_name,
        "best_estimator": best_estimator,
        "best_params": fitted_search.best_params_,
        "cv_rmse": float((-fitted_search.best_score_) ** 0.5),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def build_predictions_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Build prediction table with residual diagnostics."""
    predictions_df = pd.DataFrame(
        {
            "actual_total_fare_bdt": y_true.values,
            "predicted_total_fare_bdt": y_pred.values,
        }
    )
    predictions_df["residual"] = predictions_df["actual_total_fare_bdt"] - predictions_df["predicted_total_fare_bdt"]
    predictions_df["absolute_error"] = predictions_df["residual"].abs()
    predictions_df["squared_error"] = predictions_df["residual"] ** 2
    return predictions_df


def extract_feature_importance(best_model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature impact from coefficients or feature importances."""
    if hasattr(best_model, "coef_"):
        impacts = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": best_model.coef_,
                "importance_type": "coefficient",
            }
        )
    elif hasattr(best_model, "feature_importances_"):
        impacts = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": best_model.feature_importances_,
                "importance_type": "feature_importance",
            }
        )
    else:
        impacts = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": [0.0] * len(feature_names),
                "importance_type": "unavailable",
            }
        )

    impacts["abs_importance"] = impacts["importance"].abs()
    return impacts.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def build_regularization_effect_table(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compute Ridge and Lasso performance across alpha values."""
    rows: list[dict[str, Any]] = []
    alphas = [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha, random_state=None)
        ridge_model.fit(x_train, y_train)
        ridge_train = compute_metrics(y_train, pd.Series(ridge_model.predict(x_train), index=y_train.index))
        ridge_test = compute_metrics(y_test, pd.Series(ridge_model.predict(x_test), index=y_test.index))
        rows.append(
            {
                "model_family": "Ridge",
                "alpha": alpha,
                "train_rmse": ridge_train["rmse"],
                "test_rmse": ridge_test["rmse"],
                "train_r2": ridge_train["r2"],
                "test_r2": ridge_test["r2"],
            }
        )

        lasso_model = Lasso(alpha=alpha, max_iter=20000, tol=0.0001)
        lasso_model.fit(x_train, y_train)
        lasso_train = compute_metrics(y_train, pd.Series(lasso_model.predict(x_train), index=y_train.index))
        lasso_test = compute_metrics(y_test, pd.Series(lasso_model.predict(x_test), index=y_test.index))
        rows.append(
            {
                "model_family": "Lasso",
                "alpha": alpha,
                "train_rmse": lasso_train["rmse"],
                "test_rmse": lasso_test["rmse"],
                "train_r2": lasso_train["r2"],
                "test_r2": lasso_test["r2"],
            }
        )

    return pd.DataFrame(rows)


def build_bias_variance_table(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Compute decision-tree train/test error across depth values."""
    rows: list[dict[str, Any]] = []
    depth_values: list[int | None] = [2, 4, 6, 8, 10, 12, 16, None]

    for depth in depth_values:
        model = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        model.fit(x_train, y_train)
        train_metrics = compute_metrics(y_train, pd.Series(model.predict(x_train), index=y_train.index))
        test_metrics = compute_metrics(y_test, pd.Series(model.predict(x_test), index=y_test.index))
        rows.append(
            {
                "max_depth": depth if depth is not None else "None",
                "train_rmse": train_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
            }
        )

    return pd.DataFrame(rows)


def create_model_comparison_plot(comparison_df: pd.DataFrame, output_path: Path) -> Path:
    """Save bar chart of test RMSE by model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_df = comparison_df.sort_values("test_rmse")
    plt.figure(figsize=(10, 6))
    plt.bar(ordered_df["model_name"], ordered_df["test_rmse"], color="#2a9d8f")
    plt.title("Model Comparison by Test RMSE")
    plt.xlabel("Model")
    plt.ylabel("Test RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_regularization_plot(regularization_df: pd.DataFrame, output_path: Path) -> Path:
    """Save regularization effect plot for Ridge and Lasso."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for model_family in ["Ridge", "Lasso"]:
        family_df = regularization_df[regularization_df["model_family"] == model_family].sort_values("alpha")
        plt.plot(family_df["alpha"], family_df["test_rmse"], marker="o", label=f"{model_family} test RMSE")
        plt.plot(family_df["alpha"], family_df["train_rmse"], linestyle="--", marker="x", label=f"{model_family} train RMSE")
    plt.xscale("log")
    plt.title("Regularization Strength vs RMSE")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_bias_variance_plot(bias_variance_df: pd.DataFrame, output_path: Path) -> Path:
    """Save bias-variance tradeoff plot across tree depth values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = bias_variance_df.copy()
    labels = plot_df["max_depth"].astype(str).tolist()
    x_values = list(range(len(labels)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, plot_df["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(x_values, plot_df["test_rmse"], marker="o", label="Test RMSE")
    plt.title("Bias-Variance Tradeoff (Decision Tree Depth)")
    plt.xlabel("Max Depth")
    plt.ylabel("RMSE")
    plt.xticks(x_values, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def run_stage_5(
    settings: Stage5Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 300.0,
    retry_delay_seconds: float = 1.5,
    io_retries: int = 3,
    io_timeout_seconds: float = 60.0,
    io_retry_delay_seconds: float = 1.0,
) -> Stage5Outputs:
    """Execute Stage 5 advanced model tuning end-to-end."""
    runtime_settings = settings or Stage5Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage5")
    try:
        logger.info("Starting Stage 5 advanced modeling pipeline.")

        x_train = load_csv_with_retries(
            csv_path=runtime_settings.x_train_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        x_test = load_csv_with_retries(
            csv_path=runtime_settings.x_test_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        y_train_df = load_csv_with_retries(
            csv_path=runtime_settings.y_train_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        y_test_df = load_csv_with_retries(
            csv_path=runtime_settings.y_test_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )

        y_train = select_target_column(y_train_df, runtime_settings.target_column)
        y_test = select_target_column(y_test_df, runtime_settings.target_column)
        validate_inputs(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=runtime_settings.cv_folds,
            tuning_sample_fraction=runtime_settings.tuning_sample_fraction,
        )

        x_tune, y_tune = sample_tuning_data(
            x_train=x_train,
            y_train=y_train,
            sample_fraction=runtime_settings.tuning_sample_fraction,
            random_state=runtime_settings.random_state,
        )
        logger.info("Tuning sample rows: %s of %s", len(x_tune), len(x_train))

        model_searches: list[tuple[str, GridSearchCV | RandomizedSearchCV]] = [
            (
                "Ridge",
                GridSearchCV(
                    estimator=Ridge(),
                    param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
                    scoring=SCORING,
                    cv=runtime_settings.cv_folds,
                    n_jobs=runtime_settings.n_jobs,
                ),
            ),
            (
                "Lasso",
                GridSearchCV(
                    estimator=Lasso(max_iter=20000, tol=0.0001),
                    param_grid={"alpha": [0.0005, 0.001, 0.01, 0.1, 1.0, 5.0]},
                    scoring=SCORING,
                    cv=runtime_settings.cv_folds,
                    n_jobs=runtime_settings.n_jobs,
                ),
            ),
            (
                "DecisionTreeRegressor",
                RandomizedSearchCV(
                    estimator=DecisionTreeRegressor(random_state=runtime_settings.random_state),
                    param_distributions={
                        "max_depth": [4, 8, 12, 16, None],
                        "min_samples_split": [2, 5, 10, 20],
                        "min_samples_leaf": [1, 2, 4, 8],
                    },
                    n_iter=runtime_settings.tree_search_iterations,
                    scoring=SCORING,
                    cv=runtime_settings.cv_folds,
                    random_state=runtime_settings.random_state,
                    n_jobs=runtime_settings.n_jobs,
                ),
            ),
            (
                "RandomForestRegressor",
                RandomizedSearchCV(
                    estimator=RandomForestRegressor(random_state=runtime_settings.random_state, n_jobs=runtime_settings.n_jobs),
                    param_distributions={
                        "n_estimators": [60, 100, 140],
                        "max_depth": [8, 12, 16, None],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", 0.7],
                    },
                    n_iter=runtime_settings.forest_search_iterations,
                    scoring=SCORING,
                    cv=runtime_settings.cv_folds,
                    random_state=runtime_settings.random_state,
                    n_jobs=runtime_settings.n_jobs,
                ),
            ),
            (
                "GradientBoostingRegressor",
                RandomizedSearchCV(
                    estimator=GradientBoostingRegressor(random_state=runtime_settings.random_state),
                    param_distributions={
                        "n_estimators": [80, 120, 160],
                        "learning_rate": [0.03, 0.05, 0.1],
                        "max_depth": [2, 3, 4],
                        "subsample": [0.7, 0.85, 1.0],
                    },
                    n_iter=runtime_settings.gb_search_iterations,
                    scoring=SCORING,
                    cv=runtime_settings.cv_folds,
                    random_state=runtime_settings.random_state,
                    n_jobs=runtime_settings.n_jobs,
                ),
            ),
        ]

        tuning_results: list[dict[str, Any]] = []
        for model_name, search_obj in model_searches:
            logger.info("Tuning model: %s", model_name)
            result = tune_estimator(
                model_name=model_name,
                search=search_obj,
                x_tune=x_tune,
                y_tune=y_tune,
                x_train_full=x_train,
                y_train_full=y_train,
                x_test=x_test,
                y_test=y_test,
                logger_name=logger.name,
                retries=retries,
                timeout_seconds=timeout_seconds,
                retry_delay_seconds=retry_delay_seconds,
            )
            tuning_results.append(result)
            logger.info(
                "Completed %s: CV RMSE=%.3f, Test RMSE=%.3f",
                model_name,
                result["cv_rmse"],
                result["test_metrics"]["rmse"],
            )

        comparison_rows: list[dict[str, Any]] = []
        for result in tuning_results:
            comparison_rows.append(
                {
                    "model_name": result["model_name"],
                    "cv_rmse": result["cv_rmse"],
                    "train_r2": result["train_metrics"]["r2"],
                    "test_r2": result["test_metrics"]["r2"],
                    "train_mae": result["train_metrics"]["mae"],
                    "test_mae": result["test_metrics"]["mae"],
                    "train_rmse": result["train_metrics"]["rmse"],
                    "test_rmse": result["test_metrics"]["rmse"],
                }
            )
        comparison_df = pd.DataFrame(comparison_rows).sort_values("test_rmse").reset_index(drop=True)

        best_result = min(tuning_results, key=lambda item: item["test_metrics"]["rmse"])
        best_model = best_result["best_estimator"]
        best_test_predictions = pd.Series(best_model.predict(x_test), index=y_test.index)
        best_predictions_df = build_predictions_table(y_true=y_test, y_pred=best_test_predictions)
        best_importance_df = extract_feature_importance(best_model, feature_names=x_train.columns.tolist())

        regularization_df = build_regularization_effect_table(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        bias_variance_df = build_bias_variance_table(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            random_state=runtime_settings.random_state,
        )

        stage_output_dir = runtime_settings.artifacts_dir / "stage_5"
        plots_dir = stage_output_dir / "plots"
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        comparison_table_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(comparison_df, stage_output_dir / "model_comparison.csv"),
            operation_name="save_model_comparison",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        tuning_results_payload = [
            {
                "model_name": result["model_name"],
                "best_params": result["best_params"],
                "cv_rmse": result["cv_rmse"],
                "train_metrics": result["train_metrics"],
                "test_metrics": result["test_metrics"],
            }
            for result in tuning_results
        ]
        tuning_results_path = run_with_retries_and_timeout(
            operation=lambda: save_json({"stage": 5, "results": tuning_results_payload}, stage_output_dir / "tuning_results.json"),
            operation_name="save_tuning_results",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        best_model_path = run_with_retries_and_timeout(
            operation=lambda: save_model_pickle(best_model, stage_output_dir / "best_model.pkl"),
            operation_name="save_best_model",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        best_model_predictions_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(best_predictions_df, stage_output_dir / "best_model_predictions_test.csv"),
            operation_name="save_best_model_predictions",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        regularization_effect_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(regularization_df, stage_output_dir / "regularization_effect.csv"),
            operation_name="save_regularization_effect",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        bias_variance_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(bias_variance_df, stage_output_dir / "bias_variance_tradeoff.csv"),
            operation_name="save_bias_variance_tradeoff",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        best_model_importance_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(best_importance_df, stage_output_dir / "best_model_feature_impact.csv"),
            operation_name="save_best_model_feature_impact",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        model_comparison_plot_path = run_with_retries_and_timeout(
            operation=lambda: create_model_comparison_plot(comparison_df, plots_dir / "model_test_rmse_comparison.png"),
            operation_name="plot_model_test_rmse_comparison",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        regularization_plot_path = run_with_retries_and_timeout(
            operation=lambda: create_regularization_plot(regularization_df, plots_dir / "regularization_effect.png"),
            operation_name="plot_regularization_effect",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        bias_variance_plot_path = run_with_retries_and_timeout(
            operation=lambda: create_bias_variance_plot(bias_variance_df, plots_dir / "bias_variance_tradeoff.png"),
            operation_name="plot_bias_variance_tradeoff",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        summary_payload = {
            "stage": 5,
            "best_model": {
                "model_name": best_result["model_name"],
                "best_params": best_result["best_params"],
                "test_metrics": best_result["test_metrics"],
                "train_metrics": best_result["train_metrics"],
            },
            "comparison_rows": int(len(comparison_df)),
            "plots": {
                "model_comparison": str(model_comparison_plot_path),
                "regularization_effect": str(regularization_plot_path),
                "bias_variance_tradeoff": str(bias_variance_plot_path),
            },
        }
        summary_report_path = run_with_retries_and_timeout(
            operation=lambda: save_json(summary_payload, stage_output_dir / "stage5_summary_report.json"),
            operation_name="save_stage5_summary_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        logger.info("Stage 5 completed successfully.")
        logger.info("Best model: %s | Test RMSE: %.4f", best_result["model_name"], best_result["test_metrics"]["rmse"])

        return Stage5Outputs(
            comparison_table_path=comparison_table_path,
            tuning_results_path=tuning_results_path,
            best_model_path=best_model_path,
            best_model_predictions_path=best_model_predictions_path,
            summary_report_path=summary_report_path,
            regularization_effect_path=regularization_effect_path,
            bias_variance_path=bias_variance_path,
            best_model_importance_path=best_model_importance_path,
            plots_directory=plots_dir,
        )
    finally:
        shutdown_logger_handlers(logger)
