"""Stage 6 pipeline for model interpretation and stakeholder-focused insights."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .logging_utils import configure_logging, shutdown_logger_handlers
from .preprocessing import DATE_COLUMN, month_to_season
from .settings import Stage6Settings

T = TypeVar("T")

TARGET_FALLBACK = "Total Fare (BDT)"
AIRLINE_COLUMN = "Airline"
SOURCE_COLUMN = "Source"
DESTINATION_COLUMN = "Destination"
SEASON_COLUMN = "Departure Season"
DEPARTURE_MONTH_COLUMN = "Departure Month"
ROUTE_COLUMN = "Route"
LEAKY_FEATURE_NAMES = {"Base Fare (BDT)", "Tax & Surcharge (BDT)"}


class Stage6Error(RuntimeError):
    """Raised when Stage 6 execution fails."""


class Stage6TimeoutError(Stage6Error):
    """Raised when a Stage 6 operation exceeds timeout."""


@dataclass(frozen=True)
class Stage6Outputs:
    """Artifacts produced by Stage 6 pipeline execution."""

    summary_report_path: Path
    stakeholder_report_path: Path
    top_feature_table_path: Path
    airline_fare_summary_path: Path
    seasonal_pricing_path: Path
    route_season_hotspots_path: Path
    plots_directory: Path


def run_with_retries_and_timeout(
    operation: Callable[[], T],
    operation_name: str,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> T:
    """Run any operation with retries and timeout protection."""
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
            last_error = Stage6TimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
            )
            logger.warning(
                "Operation '%s' timed out on attempt %s/%s.",
                operation_name,
                attempt,
                retries,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = Stage6Error(f"Operation '{operation_name}' failed: {exc}")
            logger.warning(
                "Operation '%s' failed on attempt %s/%s: %s",
                operation_name,
                attempt,
                retries,
                exc,
            )

        if attempt < retries:
            # Delay to mitigate transient filesystem issues.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise Stage6Error(f"Operation '{operation_name}' failed with unknown error.")
    raise last_error


def load_csv_with_retries(
    csv_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> pd.DataFrame:
    """Load CSV with retry and timeout protection."""
    return run_with_retries_and_timeout(
        operation=lambda: pd.read_csv(csv_path),
        operation_name=f"load_csv:{csv_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def load_json_with_retries(
    json_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    """Load JSON with retry and timeout protection."""
    return run_with_retries_and_timeout(
        operation=lambda: json.loads(json_path.read_text(encoding="utf-8")),
        operation_name=f"load_json:{json_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def save_dataframe_csv(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Persist dataframe as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_json(payload: dict[str, Any], output_path: Path) -> Path:
    """Persist dictionary payload as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_text(payload: str, output_path: Path) -> Path:
    """Persist text content to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def normalize_feature_name(feature_name: str) -> str:
    """Convert encoded feature names into readable labels for reporting."""
    if feature_name.startswith("num__"):
        return feature_name.removeprefix("num__")
    if feature_name.startswith("cat__"):
        raw_name = feature_name.removeprefix("cat__")
        if "_" in raw_name:
            field_name, category_name = raw_name.split("_", 1)
            return f"{field_name} = {category_name.replace('_', ' ')}"
        return raw_name
    return feature_name.replace("_", " ")


def prepare_feature_impact_table(feature_impact_df: pd.DataFrame, top_feature_count: int) -> pd.DataFrame:
    """Prepare top feature driver table with readable labels and percentages."""
    required_columns = {"feature_name", "importance"}
    missing_columns = required_columns.difference(feature_impact_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing feature impact columns: {missing_text}")

    working_df = feature_impact_df.copy()
    # Enforce leakage safety even when stale upstream artifacts contain leaked fare components.
    normalized_names = working_df["feature_name"].astype(str).apply(normalize_feature_name)
    leaky_mask = normalized_names.isin(LEAKY_FEATURE_NAMES)
    working_df = working_df.loc[~leaky_mask].copy()

    if "abs_importance" not in working_df.columns:
        working_df["abs_importance"] = working_df["importance"].abs()
    working_df["readable_feature"] = working_df["feature_name"].apply(normalize_feature_name)

    total_importance = float(working_df["abs_importance"].sum())
    if total_importance > 0:
        working_df["importance_pct"] = (working_df["abs_importance"] / total_importance) * 100
    else:
        working_df["importance_pct"] = 0.0

    top_df = (
        working_df.sort_values("abs_importance", ascending=False)
        .head(top_feature_count)
        .reset_index(drop=True)
    )
    top_df["rank"] = top_df.index + 1
    return top_df[
        [
            "rank",
            "feature_name",
            "readable_feature",
            "importance",
            "abs_importance",
            "importance_pct",
            "importance_type",
        ]
    ]


def ensure_season_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataset contains a season column for seasonal insights."""
    working_df = dataframe.copy()
    if SEASON_COLUMN in working_df.columns:
        working_df[SEASON_COLUMN] = working_df[SEASON_COLUMN].astype(str).str.strip().str.lower()
        return working_df

    if DEPARTURE_MONTH_COLUMN in working_df.columns:
        month_series = pd.to_numeric(working_df[DEPARTURE_MONTH_COLUMN], errors="coerce").fillna(1).astype(int)
        working_df[SEASON_COLUMN] = month_series.apply(month_to_season)
        return working_df

    if DATE_COLUMN in working_df.columns:
        date_series = pd.to_datetime(working_df[DATE_COLUMN], errors="coerce")
        month_series = date_series.dt.month.fillna(1).astype(int)
        working_df[SEASON_COLUMN] = month_series.apply(month_to_season)
        return working_df

    working_df[SEASON_COLUMN] = "unknown"
    return working_df


def ensure_route_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataset contains a canonical route label column."""
    working_df = dataframe.copy()
    source_series = working_df.get(SOURCE_COLUMN, pd.Series(["Unknown"] * len(working_df))).astype(str)
    destination_series = working_df.get(DESTINATION_COLUMN, pd.Series(["Unknown"] * len(working_df))).astype(str)
    working_df[ROUTE_COLUMN] = source_series.str.strip().str.upper() + " -> " + destination_series.str.strip().str.upper()
    return working_df


def build_airline_fare_summary(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Compute airline-level fare summary to support traveler cost comparison."""
    global_avg = float(dataframe[target_column].mean())
    summary_df = (
        dataframe.groupby(AIRLINE_COLUMN, dropna=False)[target_column]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "flight_count",
                "mean": "avg_total_fare_bdt",
                "median": "median_total_fare_bdt",
                "std": "std_total_fare_bdt",
                "min": "min_total_fare_bdt",
                "max": "max_total_fare_bdt",
            }
        )
    )
    if global_avg == 0:
        summary_df["avg_fare_delta_vs_global_pct"] = 0.0
    else:
        summary_df["avg_fare_delta_vs_global_pct"] = ((summary_df["avg_total_fare_bdt"] / global_avg) - 1) * 100
    return summary_df.sort_values("avg_total_fare_bdt", ascending=False).reset_index(drop=True)


def build_seasonal_pricing_summary(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Compute season-level fare summary statistics."""
    seasonal_df = (
        dataframe.groupby(SEASON_COLUMN, dropna=False)[target_column]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "flight_count",
                "mean": "avg_total_fare_bdt",
                "median": "median_total_fare_bdt",
                "std": "std_total_fare_bdt",
                "min": "min_total_fare_bdt",
                "max": "max_total_fare_bdt",
            }
        )
        .sort_values("avg_total_fare_bdt", ascending=False)
        .reset_index(drop=True)
    )
    return seasonal_df


def build_route_season_hotspots(
    dataframe: pd.DataFrame,
    target_column: str,
    min_flights: int,
    top_count: int,
) -> pd.DataFrame:
    """Compute most expensive route-season combinations with adequate support."""
    hotspot_df = (
        dataframe.groupby([SEASON_COLUMN, ROUTE_COLUMN], dropna=False)[target_column]
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
    hotspot_df = hotspot_df[hotspot_df["flight_count"] >= min_flights]
    hotspot_df = hotspot_df.sort_values("avg_total_fare_bdt", ascending=False).head(top_count).reset_index(drop=True)
    return hotspot_df


def create_feature_driver_plot(top_feature_df: pd.DataFrame, output_path: Path) -> Path:
    """Save horizontal bar chart for top feature drivers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = top_feature_df.sort_values("importance_pct", ascending=True)
    plt.figure(figsize=(11, 8))
    plt.barh(plot_df["readable_feature"], plot_df["importance_pct"], color="#2a9d8f")
    plt.title("Top Feature Drivers of Fare Predictions")
    plt.xlabel("Relative Importance (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_airline_fare_delta_plot(airline_summary_df: pd.DataFrame, output_path: Path) -> Path:
    """Save airline fare-delta plot for traveler comparison across major airlines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = airline_summary_df.head(15).sort_values("avg_fare_delta_vs_global_pct", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df[AIRLINE_COLUMN], plot_df["avg_fare_delta_vs_global_pct"], color="#e76f51")
    plt.axvline(0, color="#264653", linewidth=1.2)
    plt.title("Airline Average Fare Delta vs Global Average")
    plt.xlabel("Fare Delta vs Global Avg (%)")
    plt.ylabel("Airline")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def create_seasonal_fare_plot(seasonal_df: pd.DataFrame, output_path: Path) -> Path:
    """Save average seasonal fare plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.bar(seasonal_df[SEASON_COLUMN], seasonal_df["avg_total_fare_bdt"], color="#457b9d")
    plt.title("Average Fare by Season")
    plt.xlabel("Season")
    plt.ylabel("Average Total Fare (BDT)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def build_stakeholder_report(
    best_model_name: str,
    test_rmse: float,
    top_feature_df: pd.DataFrame,
    airline_summary_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    route_hotspots_df: pd.DataFrame,
) -> str:
    """Build markdown report focused on non-technical traveler-facing insights."""
    top_features_lines = [
        f"- `{row['readable_feature']}`: {row['importance_pct']:.2f}% relative importance"
        for _, row in top_feature_df.head(5).iterrows()
    ]
    if not top_features_lines:
        top_features_lines = ["- No non-leaky feature drivers were available in the provided feature-impact table."]
    if airline_summary_df.empty:
        highest_airline = {AIRLINE_COLUMN: "unknown", "avg_total_fare_bdt": 0.0}
        lowest_airline = {AIRLINE_COLUMN: "unknown", "avg_total_fare_bdt": 0.0}
    else:
        highest_airline = airline_summary_df.iloc[0]
        lowest_airline = airline_summary_df.sort_values("avg_total_fare_bdt", ascending=True).iloc[0]

    if seasonal_df.empty:
        highest_season = {SEASON_COLUMN: "unknown", "avg_total_fare_bdt": 0.0}
    else:
        highest_season = seasonal_df.iloc[0]
    top_hotspot = route_hotspots_df.iloc[0] if not route_hotspots_df.empty else None

    hotspot_line = (
        f"- Highest route-season hotspot: `{top_hotspot[ROUTE_COLUMN]}` in `{top_hotspot[SEASON_COLUMN]}` "
        f"(avg fare {top_hotspot['avg_total_fare_bdt']:.2f} BDT, {int(top_hotspot['flight_count'])} flights)"
        if top_hotspot is not None
        else "- No route-season hotspot met the minimum flight threshold."
    )

    return f"""# Stage 6 Stakeholder Interpretation Report

## Model Context
- Best model selected in Stage 5: `{best_model_name}`
- Test RMSE: `{test_rmse:.2f}` BDT

## Key Fare Drivers
{chr(10).join(top_features_lines)}

## Traveler Cost Insights
- Highest average fare airline: `{highest_airline[AIRLINE_COLUMN]}` with average fare `{highest_airline['avg_total_fare_bdt']:.2f}` BDT
- Most affordable airline on average: `{lowest_airline[AIRLINE_COLUMN]}` with average fare `{lowest_airline['avg_total_fare_bdt']:.2f}` BDT
- Highest-fare season: `{highest_season[SEASON_COLUMN]}` with average fare `{highest_season['avg_total_fare_bdt']:.2f}` BDT
{hotspot_line}

## Recommendations
- Book in lower-fare seasons whenever travel dates are flexible.
- Compare carriers on the same route because average fare differences are material.
- Avoid identified route-season hotspots or book earlier when those routes are unavoidable.
"""


def run_stage_6(
    settings: Stage6Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 60.0,
    retry_delay_seconds: float = 1.5,
    io_retries: int = 3,
    io_timeout_seconds: float = 30.0,
    io_retry_delay_seconds: float = 1.0,
) -> Stage6Outputs:
    """Execute Stage 6 interpretation and insight generation end-to-end."""
    runtime_settings = settings or Stage6Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage6")
    try:
        logger.info("Starting Stage 6 interpretation pipeline.")

        cleaned_df = load_csv_with_retries(
            csv_path=runtime_settings.cleaned_dataset_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        feature_impact_df = load_csv_with_retries(
            csv_path=runtime_settings.feature_impact_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        stage5_summary = load_json_with_retries(
            json_path=runtime_settings.stage5_summary_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )

        target_column = runtime_settings.target_column or TARGET_FALLBACK
        if target_column not in cleaned_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in cleaned dataset.")

        enriched_df = ensure_route_column(ensure_season_column(cleaned_df))
        top_feature_df = prepare_feature_impact_table(feature_impact_df, top_feature_count=runtime_settings.top_feature_count)
        airline_summary_df = build_airline_fare_summary(enriched_df, target_column=target_column)
        seasonal_summary_df = build_seasonal_pricing_summary(enriched_df, target_column=target_column)
        route_hotspots_df = build_route_season_hotspots(
            enriched_df,
            target_column=target_column,
            min_flights=runtime_settings.min_route_season_flights,
            top_count=runtime_settings.top_route_season_count,
        )

        best_model = stage5_summary.get("best_model", {})
        stakeholder_report = build_stakeholder_report(
            best_model_name=str(best_model.get("model_name", "Unknown")),
            test_rmse=float(best_model.get("test_metrics", {}).get("rmse", 0.0)),
            top_feature_df=top_feature_df,
            airline_summary_df=airline_summary_df,
            seasonal_df=seasonal_summary_df,
            route_hotspots_df=route_hotspots_df,
        )

        stage_output_dir = runtime_settings.artifacts_dir / "stage_6"
        plots_dir = stage_output_dir / "plots"
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        top_feature_table_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(top_feature_df, stage_output_dir / "influence_fare_prices.csv"),
            operation_name="save_influence_fare_prices",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        airline_fare_summary_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(airline_summary_df, stage_output_dir / "airline_fare_summary.csv"),
            operation_name="save_airline_fare_summary",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        seasonal_pricing_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(seasonal_summary_df, stage_output_dir / "seasonal_pricing_summary.csv"),
            operation_name="save_seasonal_pricing_summary",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        route_season_hotspots_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(route_hotspots_df, stage_output_dir / "route_season_hotspots.csv"),
            operation_name="save_route_season_hotspots",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        stakeholder_report_path = run_with_retries_and_timeout(
            operation=lambda: save_text(stakeholder_report, stage_output_dir / "stage6_stakeholder_report.md"),
            operation_name="save_stage6_stakeholder_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        feature_plot_path = run_with_retries_and_timeout(
            operation=lambda: create_feature_driver_plot(top_feature_df, plots_dir / "influence_fare_prices.png"),
            operation_name="plot_influence_fare_prices",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        airline_plot_path = run_with_retries_and_timeout(
            operation=lambda: create_airline_fare_delta_plot(airline_summary_df, plots_dir / "airline_fare_delta.png"),
            operation_name="plot_airline_fare_delta",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        seasonal_plot_path = run_with_retries_and_timeout(
            operation=lambda: create_seasonal_fare_plot(seasonal_summary_df, plots_dir / "seasonal_average_fare.png"),
            operation_name="plot_seasonal_average_fare",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        summary_payload = {
            "stage": 6,
            "best_model_name": str(best_model.get("model_name", "Unknown")),
            "best_model_test_rmse": float(best_model.get("test_metrics", {}).get("rmse", 0.0)),
            "top_driver": top_feature_df.iloc[0]["readable_feature"] if not top_feature_df.empty else None,
            "highest_fare_airline": airline_summary_df.iloc[0][AIRLINE_COLUMN] if not airline_summary_df.empty else None,
            "most_affordable_airline": (
                airline_summary_df.sort_values("avg_total_fare_bdt", ascending=True).iloc[0][AIRLINE_COLUMN]
                if not airline_summary_df.empty
                else None
            ),
            "top_season": seasonal_summary_df.iloc[0][SEASON_COLUMN] if not seasonal_summary_df.empty else None,
            "artifacts": {
                "top_feature_table_path": str(top_feature_table_path),
                "airline_fare_summary_path": str(airline_fare_summary_path),
                "seasonal_pricing_path": str(seasonal_pricing_path),
                "route_season_hotspots_path": str(route_season_hotspots_path),
                "stakeholder_report_path": str(stakeholder_report_path),
                "feature_plot_path": str(feature_plot_path),
                "airline_plot_path": str(airline_plot_path),
                "seasonal_plot_path": str(seasonal_plot_path),
            },
        }
        summary_report_path = run_with_retries_and_timeout(
            operation=lambda: save_json(summary_payload, stage_output_dir / "stage6_summary_report.json"),
            operation_name="save_stage6_summary_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        logger.info("Stage 6 completed successfully.")
        logger.info("Top feature driver: %s", summary_payload["top_driver"])
        logger.info("Highest fare airline: %s", summary_payload["highest_fare_airline"])

        return Stage6Outputs(
            summary_report_path=summary_report_path,
            stakeholder_report_path=stakeholder_report_path,
            top_feature_table_path=top_feature_table_path,
            airline_fare_summary_path=airline_fare_summary_path,
            seasonal_pricing_path=seasonal_pricing_path,
            route_season_hotspots_path=route_season_hotspots_path,
            plots_directory=plots_dir,
        )
    finally:
        shutdown_logger_handlers(logger)
