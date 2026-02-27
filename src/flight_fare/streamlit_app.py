"""Streamlit frontend connected to the FastAPI prediction service."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd

from .logging_utils import configure_logging
from .settings import StreamlitSettings


class StreamlitApiError(RuntimeError):
    """Raised when Streamlit cannot receive a valid API response."""


class StreamlitApiTimeoutError(StreamlitApiError):
    """Raised when Streamlit request to API exceeds timeout limits."""


def _read_http_error_payload(error: urlerror.HTTPError) -> str:
    """Extract response body from HTTP errors for clearer diagnostics."""
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _request_json_with_retries(
    request: urlrequest.Request,
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Send an HTTP request and decode JSON response with retry handling."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0.")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlrequest.urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except urlerror.HTTPError as exc:
            error_payload = _read_http_error_payload(exc)
            message = f"HTTP {exc.code}: {error_payload}" if error_payload else f"HTTP {exc.code}"
            last_error = StreamlitApiError(message)
            logger.warning("HTTP request failed on attempt %s/%s: %s", attempt, retries, message)
        except TimeoutError as exc:
            last_error = StreamlitApiTimeoutError(
                f"Request timed out after {timeout_seconds} seconds."
            )
            logger.warning("Request timed out on attempt %s/%s.", attempt, retries)
        except urlerror.URLError as exc:
            reason_text = str(getattr(exc, "reason", exc))
            if "timed out" in reason_text.lower():
                last_error = StreamlitApiTimeoutError(
                    f"Request timed out after {timeout_seconds} seconds."
                )
            else:
                last_error = StreamlitApiError(f"Network error: {reason_text}")
            logger.warning("Network request failed on attempt %s/%s: %s", attempt, retries, reason_text)
        except json.JSONDecodeError as exc:
            last_error = StreamlitApiError("API response is not valid JSON.")
            logger.warning("JSON decode failed on attempt %s/%s: %s", attempt, retries, exc)

        if attempt < retries:
            # Delay between retries helps absorb short-lived API startup/network issues.
            time.sleep(retry_delay_seconds)

    if last_error is None:
        raise StreamlitApiError("API request failed with an unknown error.")
    raise last_error


def request_health(
    api_base_url: str,
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Call FastAPI health endpoint with retries and timeout protection."""
    health_url = f"{api_base_url.rstrip('/')}/health"
    request = urlrequest.Request(health_url, method="GET")
    return _request_json_with_retries(
        request=request,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        logger=logger,
    )


def request_predictions(
    api_base_url: str,
    records: list[dict[str, Any]],
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Call FastAPI prediction endpoint with retries and timeout protection."""
    if not records:
        raise StreamlitApiError("No rows provided for prediction.")

    payload = json.dumps({"records": records}).encode("utf-8")
    predict_url = f"{api_base_url.rstrip('/')}/predict"
    request = urlrequest.Request(
        predict_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _request_json_with_retries(
        request=request,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        logger=logger,
    )


def request_customer_prediction(
    api_base_url: str,
    record: dict[str, Any],
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Call FastAPI prediction endpoint for a single-customer record."""
    if not isinstance(record, dict) or not record:
        raise StreamlitApiError("Customer input record must be a non-empty object.")

    payload = json.dumps({"record": record}).encode("utf-8")
    predict_url = f"{api_base_url.rstrip('/')}/predict"
    request = urlrequest.Request(
        predict_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _request_json_with_retries(
        request=request,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        logger=logger,
    )


def request_schema(
    api_base_url: str,
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Call FastAPI schema endpoint with retries and timeout protection."""
    schema_url = f"{api_base_url.rstrip('/')}/schema"
    request = urlrequest.Request(schema_url, method="GET")
    return _request_json_with_retries(
        request=request,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        logger=logger,
    )


def load_local_input_schema(
    package_dir: Path,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Load packaged input schema locally when API schema endpoint is unavailable."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")

    schema_path = package_dir / "input_schema.json"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema_payload, dict):
                raise ValueError("input_schema.json payload must be a JSON object.")
            return schema_payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Failed reading local schema on attempt %s/%s: %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                # Delay helps absorb temporary file-access contention.
                time.sleep(retry_delay_seconds)

    message = f"Unable to load input schema at {schema_path}."
    raise StreamlitApiError(message) from last_error


def coerce_customer_input(raw_value: str, field_type: str, field_name: str) -> Any:
    """Coerce a raw text field value into model-ready numeric or categorical types."""
    normalized = raw_value.strip()
    if not normalized:
        return None

    if field_type.lower() == "numeric":
        try:
            value = float(normalized)
        except ValueError as exc:
            raise StreamlitApiError(f"Field '{field_name}' expects a numeric value.") from exc
        return int(value) if value.is_integer() else value
    return normalized


def load_default_input_template(
    package_dir: Path,
    retries: int,
    retry_delay_seconds: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load packaged sample input template with retry handling."""
    if retries < 1:
        raise ValueError("retries must be at least 1.")

    template_path = package_dir / "sample_input_template.csv"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return pd.read_csv(template_path)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Failed reading sample template on attempt %s/%s: %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                # Delay helps absorb temporary file-access contention.
                time.sleep(retry_delay_seconds)

    message = f"Unable to load sample template at {template_path}."
    raise StreamlitApiError(message) from last_error


def render_streamlit_app(settings: StreamlitSettings | None = None) -> None:
    """Render and run the Streamlit UI for API-driven fare predictions."""
    import streamlit as st

    runtime_settings = settings or StreamlitSettings.from_env()
    logger = configure_logging(runtime_settings, logger_name="flight_fare.streamlit")

    st.set_page_config(page_title="Flight Fare Predictor", layout="wide")
    st.title("Flight Fare Predictor")
    st.caption("This frontend sends batch records to the FastAPI `/predict` endpoint.")

    with st.sidebar:
        st.header("Connection")
        api_base_url = st.text_input("API base URL", value=runtime_settings.api_base_url)
        timeout_seconds = st.number_input(
            "Request timeout (seconds)",
            min_value=1.0,
            value=float(runtime_settings.request_timeout_seconds),
            step=1.0,
        )
        retries = st.number_input(
            "Request retries",
            min_value=1,
            value=int(runtime_settings.request_retries),
            step=1,
        )
        retry_delay_seconds = st.number_input(
            "Retry delay (seconds)",
            min_value=0.0,
            value=float(runtime_settings.request_retry_delay_seconds),
            step=0.5,
        )

        if st.button("Check API Health", use_container_width=True):
            try:
                health_payload = request_health(
                    api_base_url=api_base_url,
                    timeout_seconds=float(timeout_seconds),
                    retries=int(retries),
                    retry_delay_seconds=float(retry_delay_seconds),
                    logger=logger,
                )
                st.success(f"API healthy: {health_payload}")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Health check failed: %s", exc)
                st.error(f"Health check failed: {exc}")

    input_mode = st.radio(
        "Prediction Mode",
        options=["Manual Customer Input", "Batch CSV Input"],
        horizontal=True,
    )

    if input_mode == "Manual Customer Input":
        st.subheader("Customer Input Form")
        try:
            schema_payload = request_schema(
                api_base_url=api_base_url,
                timeout_seconds=float(timeout_seconds),
                retries=int(retries),
                retry_delay_seconds=float(retry_delay_seconds),
                logger=logger,
            )
            st.caption("Input schema loaded from API.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema endpoint unavailable; falling back to local schema: %s", exc)
            schema_payload = load_local_input_schema(
                package_dir=runtime_settings.package_dir,
                retries=int(retries),
                retry_delay_seconds=float(retry_delay_seconds),
                logger=logger,
            )
            st.caption("Input schema loaded from local package.")

        required_fields = schema_payload.get("required_fields", [])
        notes = schema_payload.get("notes", [])
        if notes:
            st.info("\n".join(str(note) for note in notes))

        if not isinstance(required_fields, list) or not required_fields:
            st.error("Schema does not contain required_fields.")
            return

        with st.form("customer_input_form"):
            raw_inputs: list[tuple[str, str, str]] = []
            for index, field in enumerate(required_fields):
                field_name = str(field.get("name", "")).strip()
                field_type = str(field.get("type", "categorical")).strip().lower()
                if not field_name:
                    continue

                raw_value = st.text_input(
                    label=f"{field_name} ({field_type})",
                    value="",
                    key=f"customer_field_{index}",
                )
                raw_inputs.append((field_name, field_type, raw_value))

            submit_customer = st.form_submit_button(
                "Predict Customer Fare",
                type="primary",
                use_container_width=True,
            )

        if submit_customer:
            try:
                customer_record: dict[str, Any] = {}
                for field_name, field_type, raw_value in raw_inputs:
                    customer_record[field_name] = coerce_customer_input(
                        raw_value=raw_value,
                        field_type=field_type,
                        field_name=field_name,
                    )

                prediction_payload = request_customer_prediction(
                    api_base_url=api_base_url,
                    record=customer_record,
                    timeout_seconds=float(timeout_seconds),
                    retries=int(retries),
                    retry_delay_seconds=float(retry_delay_seconds),
                    logger=logger,
                )
                predictions_df = pd.DataFrame(prediction_payload.get("predictions", []))
                st.success("Customer fare prediction completed.")
                st.dataframe(predictions_df, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Customer prediction failed: %s", exc)
                st.error(f"Customer prediction failed: {exc}")
        return

    st.subheader("Batch Input Data")
    uploaded_file = st.file_uploader("Upload prediction CSV", type=["csv"])

    input_df: pd.DataFrame | None = None
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed reading uploaded CSV: %s", exc)
            st.error(f"Could not read uploaded CSV: {exc}")
    else:
        try:
            input_df = load_default_input_template(
                package_dir=runtime_settings.package_dir,
                retries=int(retries),
                retry_delay_seconds=float(retry_delay_seconds),
                logger=logger,
            )
            st.info("Using packaged sample input template.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("No default template available: %s", exc)
            st.warning("Upload a CSV file to begin batch predictions.")

    if input_df is None:
        return

    if len(input_df) > runtime_settings.max_upload_rows:
        st.warning(
            f"Input has {len(input_df):,} rows; showing only first "
            f"{runtime_settings.max_upload_rows:,} rows."
        )
        input_df = input_df.head(runtime_settings.max_upload_rows).copy()

    preview_df = input_df.head(runtime_settings.preview_rows).copy()
    st.caption(
        f"Preview/edit mode is limited to first {len(preview_df):,} rows "
        f"to keep the UI responsive."
    )
    edited_df = st.data_editor(preview_df, num_rows="dynamic", use_container_width=True)

    if st.button("Predict Batch Fares", type="primary", use_container_width=True):
        try:
            records = edited_df.to_dict(orient="records")
            prediction_payload = request_predictions(
                api_base_url=api_base_url,
                records=records,
                timeout_seconds=float(timeout_seconds),
                retries=int(retries),
                retry_delay_seconds=float(retry_delay_seconds),
                logger=logger,
            )
            predictions_df = pd.DataFrame(prediction_payload.get("predictions", []))
            st.success(f"Predictions completed for {prediction_payload.get('count', 0)} rows.")
            st.dataframe(predictions_df, use_container_width=True)

            csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch prediction request failed: %s", exc)
            st.error(f"Batch prediction failed: {exc}")
