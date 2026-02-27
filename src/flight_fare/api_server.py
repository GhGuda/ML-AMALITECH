"""FastAPI server for packaged flight fare predictions."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


from .logging_utils import configure_logging, shutdown_logger_handlers
from .settings import ApiSettings
from .stage7_delivery import predict_from_package


class ApiRequestError(RuntimeError):
    """Raised when an API request payload is invalid."""


class ApiTimeoutError(RuntimeError):
    """Raised when prediction exceeds request timeout."""


@dataclass(frozen=True)
class PredictionService:
    """Service wrapper for packaged model predictions."""

    settings: ApiSettings
    logger: logging.Logger

    def predict_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Run batch predictions for a list of JSON records."""
        if not records:
            raise ApiRequestError("'records' must contain at least one item.")
        if not all(isinstance(item, dict) for item in records):
            raise ApiRequestError("Each element in 'records' must be an object.")

        input_df = pd.DataFrame(records)
        return predict_from_package(
            package_dir=self.settings.package_dir,
            input_df=input_df,
            retries=self.settings.predict_retries,
            timeout_seconds=self.settings.predict_timeout_seconds,
            retry_delay_seconds=self.settings.predict_retry_delay_seconds,
        )


class PredictRequest(BaseModel):
    """Request payload model for prediction endpoint."""

    records: Any = None
    record: Any = None


def run_with_timeout(operation: Callable[[], Any], timeout_seconds: float) -> Any:
    """Execute an operation under a timeout budget."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            raise ApiTimeoutError(f"Operation timed out after {timeout_seconds} seconds.") from exc


def parse_prediction_payload(payload: PredictRequest) -> list[dict[str, Any]]:
    """Normalize prediction payload supporting single-customer and batch requests."""
    has_record = payload.record is not None
    has_records = payload.records is not None

    if has_record and has_records:
        raise ApiRequestError("Provide either 'record' or 'records', not both.")
    if not has_record and not has_records:
        raise ApiRequestError("Request must include either 'record' or 'records'.")

    if has_record:
        if not isinstance(payload.record, dict):
            raise ApiRequestError("'record' must be a JSON object.")
        return [payload.record]

    if not isinstance(payload.records, list):
        raise ApiRequestError("'records' must be a JSON array.")
    return payload.records


def create_app(settings: ApiSettings, logger: logging.Logger) -> FastAPI:
    """Create and return configured FastAPI app instance."""
    service = PredictionService(settings=settings, logger=logger)
    app = FastAPI(title="Flight Fare API", version="1.0.0")

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> JSONResponse:
        """Log request method, path, status code, and duration."""
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("HTTP %s %s -> 500 (%.2f ms)", request.method, request.url.path, elapsed_ms)
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "HTTP %s %s -> %s (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(ApiRequestError)
    async def handle_request_error(request: Request, exc: ApiRequestError) -> JSONResponse:
        """Convert domain request errors into HTTP 400 responses."""
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "message": str(exc)},
        )

    @app.exception_handler(ApiTimeoutError)
    async def handle_timeout_error(request: Request, exc: ApiTimeoutError) -> JSONResponse:
        """Convert timeout errors into HTTP 504 responses."""
        return JSONResponse(
            status_code=504,
            content={"error": "timeout", "message": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return a consistent bad-request response for schema validation failures."""
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "message": "Invalid request payload structure."},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return API health status and active package location."""
        return {
            "status": "ok",
            "service": "flight_fare_api",
            "package_dir": str(service.settings.package_dir),
        }

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        """Return packaged input schema for customer input clients."""
        schema_path = service.settings.package_dir / "input_schema.json"
        try:
            schema_payload = run_with_timeout(
                operation=lambda: json.loads(schema_path.read_text(encoding="utf-8")),
                timeout_seconds=service.settings.request_timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ApiRequestError(f"Input schema file not found: {schema_path}") from exc
        except json.JSONDecodeError as exc:
            raise ApiRequestError(f"Input schema file is not valid JSON: {schema_path}") from exc

        if not isinstance(schema_payload, dict):
            raise ApiRequestError("Input schema payload must be a JSON object.")
        return schema_payload

    @app.post("/predict")
    def predict(payload: PredictRequest) -> dict[str, Any]:
        """Generate fare predictions for customer input or batch records."""
        records = parse_prediction_payload(payload)

        predictions_df = run_with_timeout(
            operation=lambda: service.predict_records(records),
            timeout_seconds=service.settings.request_timeout_seconds,
        )
        return {
            "predictions": predictions_df.to_dict(orient="records"),
            "count": int(len(predictions_df)),
        }

    return app


def create_server(settings: ApiSettings, logger: logging.Logger) -> FastAPI:
    """Backward-compatible alias that returns a FastAPI app instance."""
    return create_app(settings=settings, logger=logger)


def run_api_server(settings: ApiSettings | None = None) -> None:
    """Run the FastAPI server with uvicorn until interrupted."""
    runtime_settings = settings or ApiSettings.from_env()
    runtime_settings.ensure_directories()

    logger = configure_logging(runtime_settings, logger_name="flight_fare.api")
    app = create_app(runtime_settings, logger=logger)
    logger.info("Starting API server on http://%s:%s", runtime_settings.host, runtime_settings.port)
    try:
        uvicorn.run(
            app,
            host=runtime_settings.host,
            port=runtime_settings.port,
            log_level=runtime_settings.log_level.lower(),
            access_log=False,
        )
    finally:
        shutdown_logger_handlers(logger)
