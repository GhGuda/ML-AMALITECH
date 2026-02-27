"""Stage 3 pipeline for traveler-centric exploratory analysis and KPI artifacts."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_loader import load_flight_dataset
from .logging_utils import configure_logging, shutdown_logger_handlers
from .preprocessing import DATE_COLUMN, TARGET_COLUMN, month_to_season
from .settings import Stage3Settings

T = TypeVar("T")

SOURCE_COLUMN = "Source"
DESTINATION_COLUMN = "Destination"
DEPARTURE_MONTH_COLUMN = "Departure Month"
DEPARTURE_SEASON_COLUMN = "Departure Season"
SEASONALITY_COLUMN = "Seasonality"
ROUTE_COLUMN = "Route"
CLASS_COLUMN = "Class"
STOPOVERS_COLUMN = "Stopovers"
DAYS_BEFORE_COLUMN = "Days Before Departure"
BOOKING_WINDOW_COLUMN = "Booking Window"


class Stage3ArtifactError(RuntimeError):
    """Raised when Stage 3 artifact generation fails after retries."""


class Stage3ArtifactTimeoutError(Stage3ArtifactError):
    """Raised when Stage 3 artifact write operation exceeds timeout."""


@dataclass(frozen=True)
class Stage3Outputs:
    """Artifacts produced by Stage 3 pipeline execution."""

    summary_report_path: Path
    descriptive_stats_path: Path
    correlation_matrix_path: Path
    average_fare_by_route_path: Path
    route_frequency_path: Path
    seasonal_fare_path: Path
    booking_window_fare_path: Path
    top_expensive_routes_path: Path
    top_affordable_routes_path: Path
    plots_directory: Path


def run_with_retries_and_timeout(
    operation: Callable[[], T],
    operation_name: str,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> T:
    """Execute an operation with retries and timeout enforcement."""
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
            last_error = Stage3ArtifactTimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
            )
            logger.warning(
                "Operation '%s' attempt %s/%s timed out.",
                operation_name,
                attempt,
                retries,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = Stage3ArtifactError(f"Operation '{operation_name}' failed: {exc}")
            logger.warning(
                "Operation '%s' attempt %s/%s failed: %s",
                operation_name,
                attempt,
                retries,
                exc,
            )

        if attempt < retries:
            # Brief delay helps absorb transient file-system contention.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise Stage3ArtifactError(f"Operation '{operation_name}' failed with unknown error.")
    raise last_error


def save_dataframe_csv(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Save dataframe to CSV and return output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_json(payload: dict, output_path: Path) -> Path:
    """Save dictionary payload as JSON and return output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_route_column(dataframe: pd.DataFrame) -> pd.Series:
    """Build a deterministic route label from source and destination codes."""
    source_series = dataframe.get(SOURCE_COLUMN, pd.Series(["Unknown"] * len(dataframe), index=dataframe.index)).astype(str)
    destination_series = dataframe.get(
        DESTINATION_COLUMN,
        pd.Series(["Unknown"] * len(dataframe), index=dataframe.index),
    ).astype(str)
    return source_series.str.strip().str.upper() + " -> " + destination_series.str.strip().str.upper()


def bucketize_booking_window(days_before_departure: float | int | None) -> str:
    """Map days-before-departure values to traveler booking windows."""
    if days_before_departure is None or pd.isna(days_before_departure):
        return "unknown"
    days = float(days_before_departure)
    if days <= 3:
        return "last_minute_0_3"
    if days <= 7:
        return "short_term_4_7"
    if days <= 14:
        return "mid_term_8_14"
    if days <= 30:
        return "planned_15_30"
    if days <= 60:
        return "early_31_60"
    return "very_early_61_plus"


def prepare_eda_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a copy of dataset with guaranteed traveler-centric helper columns."""
    prepared_df = dataframe.copy()
    prepared_df[ROUTE_COLUMN] = build_route_column(prepared_df)

    if DATE_COLUMN in prepared_df.columns:
        prepared_df[DATE_COLUMN] = pd.to_datetime(prepared_df[DATE_COLUMN], errors="coerce")

    if DEPARTURE_MONTH_COLUMN not in prepared_df.columns and DATE_COLUMN in prepared_df.columns:
        prepared_df[DEPARTURE_MONTH_COLUMN] = prepared_df[DATE_COLUMN].dt.month
    if DEPARTURE_MONTH_COLUMN not in prepared_df.columns:
        prepared_df[DEPARTURE_MONTH_COLUMN] = 1

    if DEPARTURE_SEASON_COLUMN not in prepared_df.columns:
        if DEPARTURE_MONTH_COLUMN in prepared_df.columns:
            prepared_df[DEPARTURE_SEASON_COLUMN] = prepared_df[DEPARTURE_MONTH_COLUMN].fillna(1).astype(int).apply(month_to_season)
        elif SEASONALITY_COLUMN in prepared_df.columns:
            prepared_df[DEPARTURE_SEASON_COLUMN] = prepared_df[SEASONALITY_COLUMN].astype(str).str.strip().str.lower()
        else:
            prepared_df[DEPARTURE_SEASON_COLUMN] = "unknown"

    if DAYS_BEFORE_COLUMN in prepared_df.columns:
        days_series = pd.to_numeric(prepared_df[DAYS_BEFORE_COLUMN], errors="coerce")
        prepared_df[BOOKING_WINDOW_COLUMN] = days_series.apply(bucketize_booking_window)
    else:
        prepared_df[BOOKING_WINDOW_COLUMN] = "unknown"

    return prepared_df


def summarize_fare_by_group(dataframe: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Aggregate fare statistics for a grouping column."""
    grouped = (
        dataframe.groupby(group_column, dropna=False)[TARGET_COLUMN]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    return grouped.rename(
        columns={
            "count": "flight_count",
            "mean": "avg_total_fare_bdt",
            "median": "median_total_fare_bdt",
            "min": "min_total_fare_bdt",
            "max": "max_total_fare_bdt",
            "std": "std_total_fare_bdt",
        }
    )


def summarize_if_available(dataframe: pd.DataFrame, group_column: str) -> list[dict]:
    """Return grouped summary records when the requested column exists."""
    if group_column not in dataframe.columns:
        return []
    return summarize_fare_by_group(dataframe, group_column).to_dict(orient="records")


def compute_descriptive_stats(dataframe: pd.DataFrame) -> dict:
    """Build a structured descriptive-statistics payload for travelers."""
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    numeric_summary = dataframe[numeric_columns].describe().to_dict() if numeric_columns else {}

    payload = {
        "shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
        "numeric_columns": numeric_columns,
        "target_summary": summarize_fare_by_group(dataframe, ROUTE_COLUMN).head(10).to_dict(orient="records"),
        "fare_by_route": summarize_if_available(dataframe, ROUTE_COLUMN),
        "fare_by_source": summarize_if_available(dataframe, SOURCE_COLUMN),
        "fare_by_destination": summarize_if_available(dataframe, DESTINATION_COLUMN),
        "fare_by_season": summarize_if_available(dataframe, DEPARTURE_SEASON_COLUMN),
        "fare_by_class": summarize_if_available(dataframe, CLASS_COLUMN),
        "fare_by_stopovers": summarize_if_available(dataframe, STOPOVERS_COLUMN),
        "fare_by_booking_window": summarize_if_available(dataframe, BOOKING_WINDOW_COLUMN),
        "numeric_summary": numeric_summary,
    }
    return payload


def compute_correlation_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric variables."""
    numeric_df = dataframe.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def compute_booking_window_fare(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute booking-window fare summary used for traveler planning insights."""
    if BOOKING_WINDOW_COLUMN not in dataframe.columns:
        return pd.DataFrame(columns=[BOOKING_WINDOW_COLUMN, "flight_count", "avg_total_fare_bdt", "median_total_fare_bdt"])

    booking_window_df = (
        dataframe.groupby(BOOKING_WINDOW_COLUMN, dropna=False)[TARGET_COLUMN]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "flight_count", "mean": "avg_total_fare_bdt", "median": "median_total_fare_bdt"})
        .sort_values("avg_total_fare_bdt", ascending=True)
        .reset_index(drop=True)
    )
    return booking_window_df


def compute_kpi_tables(dataframe: pd.DataFrame, top_n_routes: int) -> dict[str, pd.DataFrame]:
    """Compute Stage 3 traveler-centric KPI tables."""
    average_fare_per_route = summarize_fare_by_group(dataframe, ROUTE_COLUMN)

    route_frequency = (
        dataframe.groupby(ROUTE_COLUMN, dropna=False)
        .size()
        .reset_index(name="flight_count")
        .sort_values("flight_count", ascending=False)
    )

    seasonal_fare = (
        dataframe.groupby(DEPARTURE_SEASON_COLUMN, dropna=False)[TARGET_COLUMN]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "flight_count", "mean": "avg_total_fare_bdt", "median": "median_total_fare_bdt"})
        .sort_values("avg_total_fare_bdt", ascending=False)
    )

    booking_window_fare = compute_booking_window_fare(dataframe)

    route_fare = (
        dataframe.groupby(ROUTE_COLUMN, dropna=False)[TARGET_COLUMN]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "flight_count",
                "mean": "avg_total_fare_bdt",
                "median": "median_total_fare_bdt",
                "max": "max_total_fare_bdt",
            }
        )
    )

    top_expensive_routes = (
        route_fare[route_fare["flight_count"] >= 10]
        .sort_values("avg_total_fare_bdt", ascending=False)
        .head(top_n_routes)
    )
    if top_expensive_routes.empty:
        top_expensive_routes = route_fare.sort_values("avg_total_fare_bdt", ascending=False).head(top_n_routes)

    top_affordable_routes = (
        route_fare[route_fare["flight_count"] >= 10]
        .sort_values("avg_total_fare_bdt", ascending=True)
        .head(top_n_routes)
    )
    if top_affordable_routes.empty:
        top_affordable_routes = route_fare.sort_values("avg_total_fare_bdt", ascending=True).head(top_n_routes)

    return {
        "average_fare_per_route": average_fare_per_route,
        "route_frequency": route_frequency,
        "seasonal_fare": seasonal_fare,
        "booking_window_fare": booking_window_fare,
        "top_expensive_routes": top_expensive_routes,
        "top_affordable_routes": top_affordable_routes,
    }


def create_distribution_plot(dataframe: pd.DataFrame, column: str, title: str, output_path: Path) -> Path:
    """Create and save histogram plot for a numeric fare column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe[column].dropna(), bins=40, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_route_boxplot(dataframe: pd.DataFrame, max_routes: int, output_path: Path) -> Path:
    """Create and save total fare boxplot across top routes by frequency."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_routes = dataframe[ROUTE_COLUMN].value_counts().head(max_routes).index.tolist()
    filtered_df = dataframe[dataframe[ROUTE_COLUMN].isin(top_routes)].copy()

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=filtered_df, x=ROUTE_COLUMN, y=TARGET_COLUMN)
    plt.title("Total Fare Distribution Across Top Routes")
    plt.xlabel("Route")
    plt.ylabel("Total Fare (BDT)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_monthly_fare_plot(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Create and save average fare-by-month bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_source = dataframe.copy()
    monthly_source[DEPARTURE_MONTH_COLUMN] = pd.to_numeric(monthly_source[DEPARTURE_MONTH_COLUMN], errors="coerce").fillna(1).astype(int)

    monthly = (
        monthly_source.groupby(DEPARTURE_MONTH_COLUMN, dropna=False)[TARGET_COLUMN]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COLUMN: "avg_total_fare_bdt"})
        .sort_values(DEPARTURE_MONTH_COLUMN)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=monthly, x=DEPARTURE_MONTH_COLUMN, y="avg_total_fare_bdt", color="#2a9d8f")
    plt.title("Average Total Fare by Departure Month")
    plt.xlabel("Departure Month")
    plt.ylabel("Average Total Fare (BDT)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_booking_window_plot(booking_window_df: pd.DataFrame, output_path: Path) -> Path:
    """Create and save average fare plot across booking windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    if booking_window_df.empty:
        plt.text(0.5, 0.5, "No booking window data available", ha="center", va="center")
        plt.axis("off")
        plt.title("Average Fare by Booking Window (Unavailable)")
    else:
        sns.barplot(data=booking_window_df, x=BOOKING_WINDOW_COLUMN, y="avg_total_fare_bdt", color="#264653")
        plt.title("Average Fare by Booking Window")
        plt.xlabel("Booking Window")
        plt.ylabel("Average Total Fare (BDT)")
        plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path) -> Path:
    """Create and save correlation heatmap for numeric features."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    if correlation_matrix.empty:
        plt.text(0.5, 0.5, "No numeric features available", ha="center", va="center")
        plt.axis("off")
        plt.title("Correlation Heatmap (Unavailable)")
    else:
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu", square=True, cbar=True)
        plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def build_summary_report_payload(
    dataframe: pd.DataFrame,
    descriptive_stats: dict,
    kpi_tables: dict[str, pd.DataFrame],
    correlation_matrix: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> dict:
    """Build structured Stage 3 summary report payload."""
    most_popular_route_row = kpi_tables["route_frequency"].head(1)
    most_popular_route = None
    if not most_popular_route_row.empty:
        most_popular_route = {
            "route": most_popular_route_row.iloc[0][ROUTE_COLUMN],
            "flight_count": int(most_popular_route_row.iloc[0]["flight_count"]),
        }

    target_series = dataframe[TARGET_COLUMN].dropna()
    return {
        "stage": 3,
        "dataset_rows": int(dataframe.shape[0]),
        "dataset_columns": int(dataframe.shape[1]),
        "target_column": TARGET_COLUMN,
        "target_statistics": {
            "mean": float(target_series.mean()),
            "median": float(target_series.median()),
            "min": float(target_series.min()),
            "max": float(target_series.max()),
            "std": float(target_series.std()),
        },
        "kpis": {
            "most_popular_route": most_popular_route,
            "top_5_most_expensive_routes": kpi_tables["top_expensive_routes"].head(5).to_dict(orient="records"),
            "top_5_most_affordable_routes": kpi_tables["top_affordable_routes"].head(5).to_dict(orient="records"),
        },
        "descriptive_stats_overview": {
            "shape": descriptive_stats["shape"],
            "numeric_column_count": len(descriptive_stats["numeric_columns"]),
            "correlation_matrix_shape": list(correlation_matrix.shape),
        },
        "plot_paths": {name: str(path) for name, path in plot_paths.items()},
    }


def run_stage_3(
    settings: Stage3Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 1.5,
    io_retries: int = 3,
    io_timeout_seconds: float = 30.0,
    io_retry_delay_seconds: float = 1.0,
) -> Stage3Outputs:
    """Execute Stage 3 end-to-end and return generated artifact paths."""
    runtime_settings = settings or Stage3Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage3")
    try:
        logger.info("Starting Stage 3 pipeline execution.")
        logger.info("Input dataset path: %s", runtime_settings.dataset_path)

        dataframe = load_flight_dataset(
            csv_path=runtime_settings.dataset_path,
            logger=logger,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        eda_dataframe = prepare_eda_dataframe(dataframe)

        descriptive_stats = compute_descriptive_stats(eda_dataframe)
        correlation_matrix = compute_correlation_matrix(eda_dataframe)
        kpi_tables = compute_kpi_tables(eda_dataframe, top_n_routes=runtime_settings.top_n_routes)

        stage_output_dir = runtime_settings.artifacts_dir / "stage_3"
        plots_dir = stage_output_dir / "plots"
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        descriptive_stats_path = run_with_retries_and_timeout(
            operation=lambda: save_json(descriptive_stats, stage_output_dir / "descriptive_stats.json"),
            operation_name="save_descriptive_stats",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        correlation_matrix_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(correlation_matrix.reset_index(), stage_output_dir / "correlation_matrix.csv"),
            operation_name="save_correlation_matrix",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        average_fare_by_route_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(kpi_tables["average_fare_per_route"], stage_output_dir / "kpi_average_fare_per_route.csv"),
            operation_name="save_kpi_average_fare_per_route",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        route_frequency_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(kpi_tables["route_frequency"], stage_output_dir / "kpi_route_frequency.csv"),
            operation_name="save_kpi_route_frequency",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        seasonal_fare_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(kpi_tables["seasonal_fare"], stage_output_dir / "kpi_seasonal_fare.csv"),
            operation_name="save_kpi_seasonal_fare",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        booking_window_fare_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(kpi_tables["booking_window_fare"], stage_output_dir / "kpi_booking_window_fare.csv"),
            operation_name="save_kpi_booking_window_fare",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        top_expensive_routes_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(kpi_tables["top_expensive_routes"], stage_output_dir / "kpi_top_expensive_routes.csv"),
            operation_name="save_kpi_top_expensive_routes",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        top_affordable_routes_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(kpi_tables["top_affordable_routes"], stage_output_dir / "kpi_top_affordable_routes.csv"),
            operation_name="save_kpi_top_affordable_routes",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        plot_paths = {
            "total_fare_distribution": run_with_retries_and_timeout(
                operation=lambda: create_distribution_plot(
                    eda_dataframe,
                    TARGET_COLUMN,
                    "Total Fare Distribution",
                    plots_dir / "total_fare_distribution.png",
                ),
                operation_name="plot_total_fare_distribution",
                logger_name=logger.name,
                retries=io_retries,
                timeout_seconds=io_timeout_seconds,
                retry_delay_seconds=io_retry_delay_seconds,
            ),
            "route_boxplot": run_with_retries_and_timeout(
                operation=lambda: create_route_boxplot(
                    eda_dataframe,
                    max_routes=runtime_settings.max_airlines_boxplot,
                    output_path=plots_dir / "route_fare_boxplot.png",
                ),
                operation_name="plot_route_fare_boxplot",
                logger_name=logger.name,
                retries=io_retries,
                timeout_seconds=io_timeout_seconds,
                retry_delay_seconds=io_retry_delay_seconds,
            ),
            "monthly_average_fare": run_with_retries_and_timeout(
                operation=lambda: create_monthly_fare_plot(eda_dataframe, plots_dir / "monthly_average_fare.png"),
                operation_name="plot_monthly_average_fare",
                logger_name=logger.name,
                retries=io_retries,
                timeout_seconds=io_timeout_seconds,
                retry_delay_seconds=io_retry_delay_seconds,
            ),
            "booking_window_average_fare": run_with_retries_and_timeout(
                operation=lambda: create_booking_window_plot(
                    kpi_tables["booking_window_fare"],
                    plots_dir / "booking_window_average_fare.png",
                ),
                operation_name="plot_booking_window_average_fare",
                logger_name=logger.name,
                retries=io_retries,
                timeout_seconds=io_timeout_seconds,
                retry_delay_seconds=io_retry_delay_seconds,
            ),
            "correlation_heatmap": run_with_retries_and_timeout(
                operation=lambda: create_correlation_heatmap(correlation_matrix, plots_dir / "correlation_heatmap.png"),
                operation_name="plot_correlation_heatmap",
                logger_name=logger.name,
                retries=io_retries,
                timeout_seconds=io_timeout_seconds,
                retry_delay_seconds=io_retry_delay_seconds,
            ),
        }

        summary_payload = build_summary_report_payload(
            dataframe=eda_dataframe,
            descriptive_stats=descriptive_stats,
            kpi_tables=kpi_tables,
            correlation_matrix=correlation_matrix,
            plot_paths=plot_paths,
        )
        summary_report_path = run_with_retries_and_timeout(
            operation=lambda: save_json(summary_payload, stage_output_dir / "stage3_summary_report.json"),
            operation_name="save_stage3_summary_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        logger.info("Stage 3 completed successfully.")
        logger.info("Summary report: %s", summary_report_path)
        logger.info("Plots directory: %s", plots_dir)

        return Stage3Outputs(
            summary_report_path=summary_report_path,
            descriptive_stats_path=descriptive_stats_path,
            correlation_matrix_path=correlation_matrix_path,
            average_fare_by_route_path=average_fare_by_route_path,
            route_frequency_path=route_frequency_path,
            seasonal_fare_path=seasonal_fare_path,
            booking_window_fare_path=booking_window_fare_path,
            top_expensive_routes_path=top_expensive_routes_path,
            top_affordable_routes_path=top_affordable_routes_path,
            plots_directory=plots_dir,
        )
    finally:
        shutdown_logger_handlers(logger)
