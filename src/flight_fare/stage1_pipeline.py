"""Stage 1 pipeline for problem framing and dataset understanding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data_loader import load_flight_dataset
from .logging_utils import configure_logging
from .profiling import generate_initial_profile, save_profile
from .settings import Stage1Settings


@dataclass(frozen=True)
class Stage1Outputs:
    """Artifacts produced by Stage 1 pipeline execution."""

    profile_path: Path
    problem_definition_path: Path


def write_problem_definition(output_path: Path) -> Path:
    """Write a concise, structured problem framing document for Stage 1."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    document = f"""# Stage 1 - Problem Definition and Data Understanding

Generated at: {now_utc}

## Business Objective
Estimate flight ticket prices so travel platforms and airline teams can improve pricing strategy and customer recommendations.

## Machine Learning Task
- Task type: Supervised regression
- Target variable: `Total Fare (BDT)`
- Core input features:
  - `Airline`
  - `Source`
  - `Destination`
  - `Departure Date & Time`
  - `Base Fare (BDT)`
  - `Tax & Surcharge (BDT)`
  - `Duration (hrs)`
  - `Stopovers`

## Stage 1 Deliverables
- Production-ready logging configuration with rotating file handlers
- Resilient dataset loading with retries and timeout safeguards
- Schema validation for required modeling columns
- Initial dataset profile (shape, data types, missingness, duplicates, numeric summary, and outlier flags)

## Assumptions and Limitations
- Historical fare data quality directly impacts prediction quality.
- Stage 1 focuses on understanding and validation, not model training.
- Data drift monitoring and retraining strategy will be addressed in later stages.
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def run_stage_1(
    settings: Stage1Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 1.5,
) -> Stage1Outputs:
    """Execute Stage 1 end-to-end and return generated artifact paths."""
    runtime_settings = settings or Stage1Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage1")
    try:
        logger.info("Starting Stage 1 pipeline execution.")
        logger.info("Dataset path: %s", runtime_settings.dataset_path)

        dataframe = load_flight_dataset(
            csv_path=runtime_settings.dataset_path,
            logger=logger,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )

        stage_output_dir = runtime_settings.artifacts_dir / "stage_1"
        stage_output_dir.mkdir(parents=True, exist_ok=True)

        profile = generate_initial_profile(
            dataframe=dataframe,
            logger=logger,
            head_rows=runtime_settings.profile_head_rows,
        )
        profile_path = save_profile(profile=profile, output_path=stage_output_dir / "data_profile.json")
        problem_definition_path = write_problem_definition(stage_output_dir / "problem_definition.md")

        logger.info("Stage 1 completed successfully.")
        logger.info("Profile artifact: %s", profile_path)
        logger.info("Problem definition artifact: %s", problem_definition_path)

        return Stage1Outputs(
            profile_path=profile_path,
            problem_definition_path=problem_definition_path,
        )
    finally:
        shutdown_logger_handlers(logger)


def shutdown_logger_handlers(logger: logging.Logger) -> None:
    """Flush, close, and detach handlers to release file locks reliably."""
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()
