"""Stage 7 pipeline for packaging artifacts and production inference delivery."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from .logging_utils import configure_logging, shutdown_logger_handlers
from .preprocessing import LEAKY_FEATURE_COLUMNS, TARGET_COLUMN, clean_and_engineer_features
from .settings import Stage7Settings

T = TypeVar("T")


class Stage7Error(RuntimeError):
    """Raised when Stage 7 execution fails."""


class Stage7TimeoutError(Stage7Error):
    """Raised when a Stage 7 operation exceeds timeout."""


@dataclass(frozen=True)
class Stage7Outputs:
    """Artifacts produced by Stage 7 pipeline execution."""

    package_dir: Path
    manifest_path: Path
    schema_path: Path
    smoke_test_predictions_path: Path
    package_readme_path: Path
    summary_report_path: Path


def run_with_retries_and_timeout(
    operation: Callable[[], T],
    operation_name: str,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> T:
    """Run an operation with retry and timeout protection."""
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
            last_error = Stage7TimeoutError(
                f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
            )
            logger.warning(
                "Operation '%s' timed out on attempt %s/%s.",
                operation_name,
                attempt,
                retries,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = Stage7Error(f"Operation '{operation_name}' failed: {exc}")
            logger.warning(
                "Operation '%s' failed on attempt %s/%s: %s",
                operation_name,
                attempt,
                retries,
                exc,
            )

        if attempt < retries:
            # Delay helps absorb transient file-lock and filesystem contention issues.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise Stage7Error(f"Operation '{operation_name}' failed with unknown error.")
    raise last_error


def load_json_with_retries(
    json_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    """Load JSON file with retry and timeout protection."""
    return run_with_retries_and_timeout(
        operation=lambda: json.loads(json_path.read_text(encoding="utf-8")),
        operation_name=f"load_json:{json_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def load_pickle_with_retries(
    pickle_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> Any:
    """Load pickle file with retry and timeout protection."""
    return run_with_retries_and_timeout(
        operation=lambda: pickle.loads(pickle_path.read_bytes()),
        operation_name=f"load_pickle:{pickle_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def load_csv_with_retries(
    csv_path: Path,
    logger_name: str,
    retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
) -> pd.DataFrame:
    """Load CSV file with retry and timeout protection."""
    return run_with_retries_and_timeout(
        operation=lambda: pd.read_csv(csv_path),
        operation_name=f"load_csv:{csv_path.name}",
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )


def save_json(payload: dict[str, Any], output_path: Path) -> Path:
    """Persist dictionary payload as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_dataframe_csv(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Persist dataframe as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_text(payload: str, output_path: Path) -> Path:
    """Persist plain text to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def copy_file(src_path: Path, dest_path: Path) -> Path:
    """Copy file to destination path preserving metadata."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return dest_path


def file_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    sha = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def get_expected_input_columns(preprocessing_report: dict[str, Any]) -> list[str]:
    """Extract required raw model input columns from Stage 2 preprocessing report."""
    preprocessing_section = preprocessing_report.get("preprocessing", {})
    numeric_cols = preprocessing_section.get("numeric_feature_columns", [])
    categorical_cols = preprocessing_section.get("categorical_feature_columns", [])
    expected_columns = [*numeric_cols, *categorical_cols]
    if not expected_columns:
        raise ValueError("Preprocessing report does not contain expected input columns.")
    leaky_columns = sorted(set(expected_columns).intersection(LEAKY_FEATURE_COLUMNS))
    if leaky_columns:
        leaked = ", ".join(leaky_columns)
        raise ValueError(
            f"Preprocessing report contains leaky columns ({leaked}). "
            "Re-run Stage 2 and downstream modeling without leakage."
        )
    return expected_columns


def build_input_schema(preprocessing_report: dict[str, Any], expected_columns: list[str]) -> dict[str, Any]:
    """Build model input schema payload for packaged delivery."""
    preprocessing_section = preprocessing_report.get("preprocessing", {})
    numeric_set = set(preprocessing_section.get("numeric_feature_columns", []))
    categorical_set = set(preprocessing_section.get("categorical_feature_columns", []))

    fields = []
    for column in expected_columns:
        if column in numeric_set:
            field_type = "numeric"
        elif column in categorical_set:
            field_type = "categorical"
        else:
            field_type = "unknown"
        fields.append({"name": column, "type": field_type})

    return {
        "required_fields": fields,
        "notes": [
            "For raw data inference, include at minimum the same semantic fields used in Stage 2 cleaning.",
            "Departure-derived fields (Departure Month/Day/Weekday/Season) can be generated from Departure Date & Time.",
            "Missing values are handled by the packaged preprocessor using imputation strategies from Stage 2.",
            "Do not include leaked fare components such as base fare or tax as model input fields.",
        ],
    }


def prepare_inference_features(input_df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Clean/engineer raw input and align it to the expected model feature schema."""
    cleaned_df, _ = clean_and_engineer_features(input_df)
    working_df = cleaned_df.copy()

    if TARGET_COLUMN in working_df.columns:
        working_df = working_df.drop(columns=[TARGET_COLUMN])

    for column in expected_columns:
        if column not in working_df.columns:
            working_df[column] = pd.NA

    return working_df[expected_columns]


def predict_from_package(
    package_dir: Path,
    input_df: pd.DataFrame,
    retries: int = 3,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Run batch inference using a packaged model directory and input dataframe."""
    manifest_path = package_dir / "model_manifest.json"
    schema_path = package_dir / "input_schema.json"
    model_path = package_dir / "best_model.pkl"
    preprocessor_path = package_dir / "preprocessor.pkl"

    logger_name = "flight_fare.stage7.predict"
    manifest = load_json_with_retries(
        manifest_path,
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )
    schema = load_json_with_retries(
        schema_path,
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )
    model = load_pickle_with_retries(
        model_path,
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )
    preprocessor = load_pickle_with_retries(
        preprocessor_path,
        logger_name=logger_name,
        retries=retries,
        timeout_seconds=timeout_seconds,
        retry_delay_seconds=retry_delay_seconds,
    )

    expected_columns = [field["name"] for field in schema.get("required_fields", [])]
    if not expected_columns:
        raise ValueError("Packaged schema is missing required_fields definitions.")

    feature_df = prepare_inference_features(input_df, expected_columns=expected_columns)
    transformed = preprocessor.transform(feature_df)
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    # Preserve feature-name behavior expected by the fitted model variant.
    if hasattr(model, "feature_names_in_"):
        predictions = model.predict(transformed_df)
    else:
        predictions = model.predict(transformed)

    output_df = pd.DataFrame(
        {
            "prediction_id": range(1, len(predictions) + 1),
            "predicted_total_fare_bdt": predictions,
        }
    )
    output_df["model_name"] = manifest.get("model", {}).get("model_name", "unknown")
    output_df["package_version"] = manifest.get("package_version", "unknown")
    return output_df


def build_package_readme(package_version: str, expected_columns: list[str]) -> str:
    """Build markdown instructions for using packaged artifacts in production."""
    required_columns_lines = "\n".join(f"- `{column}`" for column in expected_columns)
    return f"""# Flight Fare Model Package ({package_version})

## Contents
- `best_model.pkl`: Best tuned model from Stage 5.
- `preprocessor.pkl`: Fitted preprocessing object from Stage 2.
- `model_manifest.json`: Version, metrics, and integrity metadata.
- `input_schema.json`: Required batch inference input schema.
- `sample_input_template.csv`: Minimal input template for batch predictions.

## Batch Inference
Use the CLI entrypoint:

```powershell
python run_predict_fares.py --package-dir artifacts/stage_7/package/{package_version} --input-csv <input.csv> --output-csv <output.csv>
```

## Required Fields
{required_columns_lines}
"""


def run_stage_7(
    settings: Stage7Settings | None = None,
    retries: int = 3,
    timeout_seconds: float = 60.0,
    retry_delay_seconds: float = 1.5,
    io_retries: int = 3,
    io_timeout_seconds: float = 30.0,
    io_retry_delay_seconds: float = 1.0,
) -> Stage7Outputs:
    """Execute Stage 7 packaging and delivery pipeline end-to-end."""
    runtime_settings = settings or Stage7Settings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.stage7")
    try:
        logger.info("Starting Stage 7 packaging pipeline.")

        stage5_summary = load_json_with_retries(
            runtime_settings.stage5_summary_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        preprocessing_report = load_json_with_retries(
            runtime_settings.preprocessing_report_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        cleaned_dataset = load_csv_with_retries(
            runtime_settings.cleaned_dataset_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        _ = load_pickle_with_retries(
            runtime_settings.best_model_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        _ = load_pickle_with_retries(
            runtime_settings.preprocessor_path,
            logger_name=logger.name,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )

        expected_columns = get_expected_input_columns(preprocessing_report)
        input_schema = build_input_schema(preprocessing_report, expected_columns=expected_columns)

        stage_output_dir = runtime_settings.artifacts_dir / "stage_7"
        package_dir = stage_output_dir / "package" / runtime_settings.package_version
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        package_dir.mkdir(parents=True, exist_ok=True)

        model_copy_path = run_with_retries_and_timeout(
            operation=lambda: copy_file(runtime_settings.best_model_path, package_dir / "best_model.pkl"),
            operation_name="copy_best_model",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        preprocessor_copy_path = run_with_retries_and_timeout(
            operation=lambda: copy_file(runtime_settings.preprocessor_path, package_dir / "preprocessor.pkl"),
            operation_name="copy_preprocessor",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )
        schema_path = run_with_retries_and_timeout(
            operation=lambda: save_json(input_schema, package_dir / "input_schema.json"),
            operation_name="save_input_schema",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        template_df = cleaned_dataset.head(max(runtime_settings.smoke_test_rows, 1)).copy()
        template_features = prepare_inference_features(template_df, expected_columns=expected_columns)
        _ = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(template_features, package_dir / "sample_input_template.csv"),
            operation_name="save_sample_input_template",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        manifest_payload = {
            "package_version": runtime_settings.package_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": stage5_summary.get("best_model", {}),
            "checksums": {
                "best_model_sha256": file_sha256(model_copy_path),
                "preprocessor_sha256": file_sha256(preprocessor_copy_path),
                "schema_sha256": file_sha256(schema_path),
            },
            "source_artifacts": {
                "stage5_summary_path": str(runtime_settings.stage5_summary_path),
                "preprocessing_report_path": str(runtime_settings.preprocessing_report_path),
                "cleaned_dataset_path": str(runtime_settings.cleaned_dataset_path),
            },
        }
        manifest_path = run_with_retries_and_timeout(
            operation=lambda: save_json(manifest_payload, package_dir / "model_manifest.json"),
            operation_name="save_model_manifest",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        package_readme = build_package_readme(runtime_settings.package_version, expected_columns=expected_columns)
        package_readme_path = run_with_retries_and_timeout(
            operation=lambda: save_text(package_readme, package_dir / "README.md"),
            operation_name="save_package_readme",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        smoke_input_df = cleaned_dataset.head(runtime_settings.smoke_test_rows).copy()
        smoke_predictions_df = predict_from_package(
            package_dir=package_dir,
            input_df=smoke_input_df,
            retries=retries,
            timeout_seconds=timeout_seconds,
            retry_delay_seconds=retry_delay_seconds,
        )
        smoke_test_predictions_path = run_with_retries_and_timeout(
            operation=lambda: save_dataframe_csv(smoke_predictions_df, stage_output_dir / "smoke_test_predictions.csv"),
            operation_name="save_smoke_test_predictions",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        summary_payload = {
            "stage": 7,
            "package_version": runtime_settings.package_version,
            "package_dir": str(package_dir),
            "best_model_name": stage5_summary.get("best_model", {}).get("model_name"),
            "expected_input_field_count": len(expected_columns),
            "smoke_test_rows": int(len(smoke_predictions_df)),
            "artifacts": {
                "manifest_path": str(manifest_path),
                "schema_path": str(schema_path),
                "smoke_test_predictions_path": str(smoke_test_predictions_path),
                "package_readme_path": str(package_readme_path),
            },
        }
        summary_report_path = run_with_retries_and_timeout(
            operation=lambda: save_json(summary_payload, stage_output_dir / "stage7_summary_report.json"),
            operation_name="save_stage7_summary_report",
            logger_name=logger.name,
            retries=io_retries,
            timeout_seconds=io_timeout_seconds,
            retry_delay_seconds=io_retry_delay_seconds,
        )

        logger.info("Stage 7 completed successfully.")
        logger.info("Package dir: %s", package_dir)
        logger.info("Smoke test predictions rows: %s", len(smoke_predictions_df))

        return Stage7Outputs(
            package_dir=package_dir,
            manifest_path=manifest_path,
            schema_path=schema_path,
            smoke_test_predictions_path=smoke_test_predictions_path,
            package_readme_path=package_readme_path,
            summary_report_path=summary_report_path,
        )
    finally:
        shutdown_logger_handlers(logger)
