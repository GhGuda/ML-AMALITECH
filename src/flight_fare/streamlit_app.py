"""Streamlit frontend connected to the FastAPI prediction service."""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
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


def _inject_app_styles() -> None:
    """Inject lightweight custom CSS for a more polished dashboard layout."""
    import streamlit as st

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap');
        :root {
            --ff-ink: #0f172a;
            --ff-muted: #475569;
            --ff-accent: #0e7490;
            --ff-accent-strong: #155e75;
            --ff-panel: #ffffff;
            --ff-border: #d8e5ef;
            --ff-shadow: rgba(15, 23, 42, 0.08);
        }

        html, body, [class*="css"] {
            font-family: "Manrope", "Trebuchet MS", "Segoe UI", sans-serif;
            color: var(--ff-ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1100px 420px at 90% -10%, #dff4ff 0%, rgba(223, 244, 255, 0) 72%),
                radial-gradient(760px 320px at -10% 8%, #f0fff7 0%, rgba(240, 255, 247, 0) 72%),
                linear-gradient(180deg, #f7fbff 0%, #ffffff 45%);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 2.4rem;
        }

        .ff-hero {
            border: 1px solid var(--ff-border);
            border-radius: 16px;
            padding: 1.15rem 1.3rem 1.05rem;
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.98) 0%, rgba(236, 251, 255, 0.94) 100%);
            box-shadow: 0 12px 26px var(--ff-shadow);
            margin-bottom: 0.9rem;
            animation: ff-fade-in 360ms ease-out both;
        }

        .ff-hero h1 {
            font-family: "Sora", "Verdana", sans-serif;
            font-size: 1.8rem;
            margin: 0;
            color: var(--ff-ink);
            letter-spacing: -0.02em;
        }

        .ff-hero p {
            margin: 0.35rem 0 0;
            color: #334155;
            font-size: 0.94rem;
        }

        .ff-panel {
            border: 1px solid var(--ff-border);
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            background: var(--ff-panel);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.75rem;
        }

        .ff-section-title {
            margin: 0.45rem 0 0.35rem;
            font-family: "Sora", "Verdana", sans-serif;
            color: #0b3b4b;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
        }

        .ff-readonly-chip {
            margin-top: 0.2rem;
            padding: 0.45rem 0.65rem;
            border: 1px solid #d3e6f1;
            background: #f2fbff;
            border-radius: 10px;
            color: #134152;
            font-size: 0.86rem;
            line-height: 1.3;
            min-height: 2.1rem;
        }

        /* Keep Streamlit labels fully visible in both light/dark browser modes */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {
            color: #0b1324 !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }

        [data-testid="stWidgetLabel"] {
            font-weight: 600 !important;
            letter-spacing: 0.015em;
            margin-bottom: 0.15rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%);
            border-right: 1px solid #e4ebf5;
        }

        [data-testid="stMetricValue"] {
            font-family: "Sora", "Verdana", sans-serif;
            letter-spacing: -0.02em;
        }

        [data-baseweb="select"] > div,
        [data-testid="stDateInput"] > div > div,
        [data-testid="stNumberInput"] > div > div,
        [data-testid="stTextInput"] > div > div {
            border-radius: 10px !important;
            border: 1px solid #c8d8e6 !important;
            box-shadow: none !important;
        }

        [data-testid="stButton"] > button {
            border-radius: 10px;
            border: 1px solid #bdd5e7;
            font-weight: 600;
            transition: all 0.18s ease;
        }

        [data-testid="stButton"] > button:hover {
            border-color: var(--ff-accent);
            color: var(--ff-accent-strong);
        }

        [data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            font-weight: 600;
        }

        div[data-testid="stForm"] {
            border: 1px solid #d6e6f1;
            border-radius: 14px;
            padding: 0.65rem 0.8rem 0.9rem;
            background: #ffffff;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        @keyframes ff-fade-in {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sorted_unique_strings(frame: pd.DataFrame, column_name: str) -> list[str]:
    """Return sorted unique non-empty string values from a dataframe column."""
    if column_name not in frame.columns:
        return []

    values: set[str] = set()
    for raw_value in frame[column_name].dropna().tolist():
        text_value = str(raw_value).strip()
        if text_value and text_value.lower() != "nan":
            values.add(text_value)
    return sorted(values)


def _with_select_placeholder(options: list[str], field_label: str) -> tuple[str, list[str]]:
    """Return select options prefixed with a placeholder entry."""
    placeholder = f"Select {field_label}"
    cleaned = [value for value in options if str(value).strip() and value != placeholder]
    return placeholder, [placeholder, *cleaned]


def _build_code_to_name_map(frame: pd.DataFrame, code_column: str, name_column: str) -> dict[str, str]:
    """Build a code->display name map from paired columns in template data."""
    if code_column not in frame.columns or name_column not in frame.columns:
        return {}

    mapping: dict[str, str] = {}
    pairs = frame[[code_column, name_column]].dropna()
    for code_raw, name_raw in pairs.itertuples(index=False):
        code_value = str(code_raw).strip()
        name_value = str(name_raw).strip()
        if code_value and name_value and code_value not in mapping:
            mapping[code_value] = name_value
    return mapping


def _build_routes_lookup(frame: pd.DataFrame, source_column: str, destination_column: str) -> dict[str, list[str]]:
    """Build source->destination lookup from template rows for guided route selection."""
    if source_column not in frame.columns or destination_column not in frame.columns:
        return {}

    lookup: dict[str, set[str]] = {}
    for source_raw, destination_raw in frame[[source_column, destination_column]].dropna().itertuples(index=False):
        source_value = str(source_raw).strip()
        destination_value = str(destination_raw).strip()
        if not source_value or not destination_value:
            continue
        lookup.setdefault(source_value, set()).add(destination_value)

    return {source: sorted(destinations) for source, destinations in lookup.items()}


def _build_float_slider_values(
    frame: pd.DataFrame,
    column_name: str,
    fallback_min: float,
    fallback_max: float,
    fallback_value: float,
    step: float,
) -> tuple[float, float, float]:
    """Build slider bounds/default for float-like fields using template data."""
    if column_name in frame.columns:
        numeric_series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    else:
        numeric_series = pd.Series(dtype=float)

    if numeric_series.empty:
        return fallback_min, fallback_max, fallback_value

    minimum = float(numeric_series.min())
    maximum = float(numeric_series.max())
    median = float(numeric_series.median())

    minimum = min(minimum, fallback_min)
    maximum = max(maximum, fallback_max)
    if minimum >= maximum:
        maximum = minimum + step

    value = min(max(median, minimum), maximum)
    return minimum, maximum, value


def _build_int_slider_values(
    frame: pd.DataFrame,
    column_name: str,
    fallback_min: int,
    fallback_max: int,
    fallback_value: int,
) -> tuple[int, int, int]:
    """Build slider bounds/default for integer fields using template data."""
    if column_name in frame.columns:
        numeric_series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    else:
        numeric_series = pd.Series(dtype=float)

    if numeric_series.empty:
        return fallback_min, fallback_max, fallback_value

    minimum = int(numeric_series.min())
    maximum = int(numeric_series.max())
    median = int(numeric_series.median())

    minimum = min(minimum, fallback_min)
    maximum = max(maximum, fallback_max)
    if minimum >= maximum:
        maximum = minimum + 1

    value = min(max(median, minimum), maximum)
    return minimum, maximum, value


def _infer_departure_season(month_value: int) -> str:
    """Infer Bangladesh season label from a calendar month value."""
    if month_value in (12, 1, 2):
        return "winter"
    if month_value in (3, 4, 5):
        return "pre_monsoon"
    if month_value in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def _require_field_names(schema_payload: dict[str, Any]) -> list[str]:
    """Extract ordered required field names from schema payload."""
    required_fields = schema_payload.get("required_fields", [])
    if not isinstance(required_fields, list):
        return []

    field_names: list[str] = []
    for field in required_fields:
        field_name = str(field.get("name", "")).strip()
        if field_name:
            field_names.append(field_name)
    return field_names


def render_streamlit_app(settings: StreamlitSettings | None = None) -> None:
    """Render and run the Streamlit UI for API-driven fare predictions."""
    import streamlit as st

    runtime_settings = settings or StreamlitSettings.from_env()
    logger = configure_logging(runtime_settings, logger_name="flight_fare.streamlit")

    st.set_page_config(page_title="Flight Fare Intelligence", layout="wide")
    _inject_app_styles()
    st.markdown(
        """
        <div class="ff-hero">
            <h1>Flight Fare Intelligence</h1>
            <p>Selector-driven fare prediction workspace powered by your packaged model API.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Connection")
        api_base_url = runtime_settings.api_base_url
        st.caption("API Endpoint")
        st.code(api_base_url, language=None)

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

        check_health = st.button("Check API Health", use_container_width=True, key="sidebar_health_check")

        if check_health:
            try:
                health_payload = request_health(
                    api_base_url=api_base_url,
                    timeout_seconds=float(timeout_seconds),
                    retries=int(retries),
                    retry_delay_seconds=float(retry_delay_seconds),
                    logger=logger,
                )
                st.success(f"API healthy ({health_payload.get('status', 'ok')})")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Health check failed: %s", exc)
                st.error(f"Health check failed: {exc}")

    try:
        schema_payload = request_schema(
            api_base_url=api_base_url,
            timeout_seconds=float(timeout_seconds),
            retries=int(retries),
            retry_delay_seconds=float(retry_delay_seconds),
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Schema endpoint unavailable; falling back to local schema: %s", exc)
        schema_payload = load_local_input_schema(
            package_dir=runtime_settings.package_dir,
            retries=int(retries),
            retry_delay_seconds=float(retry_delay_seconds),
            logger=logger,
        )

    notes = schema_payload.get("notes", [])
    required_field_names = _require_field_names(schema_payload)
    if not required_field_names:
        st.error("Schema does not contain any required input fields.")
        return

    try:
        template_df = load_default_input_template(
            package_dir=runtime_settings.package_dir,
            retries=int(retries),
            retry_delay_seconds=float(retry_delay_seconds),
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Template file unavailable: %s", exc)
        template_df = pd.DataFrame(columns=required_field_names)

    with st.container():
        st.markdown('<div class="ff-panel">', unsafe_allow_html=True)
        st.subheader("Passenger Quote Form")
        st.caption("All customer-facing fields are controlled inputs (no raw text entry).")
        st.markdown("</div>", unsafe_allow_html=True)

        if notes:
            with st.expander("Model Input Notes", expanded=False):
                st.write("\n".join(str(note) for note in notes))

        fallback_categorical_options: dict[str, list[str]] = {
            "Stopovers": ["Direct", "1 Stop", "2 Stops"],
            "Class": ["Economy", "Business", "First Class"],
            "Seasonality": ["Regular", "Holiday"],
            "Departure Season": ["winter", "pre_monsoon", "monsoon", "post_monsoon"],
        }

        categorical_options: dict[str, list[str]] = {}
        for field_name in required_field_names:
            options = _sorted_unique_strings(template_df, field_name)
            fallback_options = fallback_categorical_options.get(field_name, [])
            merged_options = options if options else fallback_options
            if merged_options:
                categorical_options[field_name] = merged_options

        source_map = _build_code_to_name_map(template_df, "Source", "Source Name")
        destination_map = _build_code_to_name_map(template_df, "Destination", "Destination Name")
        source_codes = sorted(set(categorical_options.get("Source", [])) | set(source_map.keys())) or ["DAC"]
        destination_codes = (
            sorted(set(categorical_options.get("Destination", [])) | set(destination_map.keys()))
            or ["CXB"]
        )
        airline_options = categorical_options.get("Airline", ["Biman Bangladesh Airlines"])
        class_options = categorical_options.get("Class", ["Economy"])
        stopover_options = categorical_options.get("Stopovers", ["Direct"])
        aircraft_options = categorical_options.get("Aircraft Type", ["Airbus A320"])
        booking_options = categorical_options.get("Booking Source", ["Online Website"])
        seasonality_options = categorical_options.get("Seasonality", ["Regular"])
        departure_season_options = categorical_options.get(
            "Departure Season",
            ["winter", "pre_monsoon", "monsoon", "post_monsoon"],
        )

        source_placeholder, source_select_options = _with_select_placeholder(source_codes, "Source Airport Code")
        destination_placeholder, destination_select_options = _with_select_placeholder(
            destination_codes,
            "Destination Airport Code",
        )
        airline_placeholder, airline_select_options = _with_select_placeholder(airline_options, "Airline")
        class_placeholder, class_select_options = _with_select_placeholder(class_options, "Cabin Class")
        stopovers_placeholder, stopovers_select_options = _with_select_placeholder(stopover_options, "Stopovers")
        aircraft_placeholder, aircraft_select_options = _with_select_placeholder(aircraft_options, "Aircraft Type")
        booking_placeholder, booking_select_options = _with_select_placeholder(booking_options, "Booking Channel")
        seasonality_placeholder, seasonality_select_options = _with_select_placeholder(
            seasonality_options,
            "Seasonality",
        )
        departure_season_placeholder, departure_season_select_options = _with_select_placeholder(
            departure_season_options,
            "Departure Season",
        )

        duration_min, duration_max, duration_default = _build_float_slider_values(
            frame=template_df,
            column_name="Duration (hrs)",
            fallback_min=0.5,
            fallback_max=8.0,
            fallback_value=1.5,
            step=0.25,
        )
        _, _, days_default = _build_int_slider_values(
            frame=template_df,
            column_name="Days Before Departure",
            fallback_min=1,
            fallback_max=180,
            fallback_value=30,
        )

        with st.form("single_customer_selector_form"):
            st.markdown('<p class="ff-section-title">Route Details</p>', unsafe_allow_html=True)
            route_col1, route_col2 = st.columns(2)
            with route_col1:
                source_code = st.selectbox("Source Airport Code", options=source_select_options, index=0)
                source_name = source_map.get(source_code) if source_code != source_placeholder else None
                if source_name:
                    st.markdown(
                        f'<div class="ff-readonly-chip"><strong>Source Airport Name:</strong> {source_name}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ff-readonly-chip"><strong>Source Airport Name:</strong> Not selected</div>',
                        unsafe_allow_html=True,
                    )

            with route_col2:
                destination_code = st.selectbox(
                    "Destination Airport Code",
                    options=destination_select_options,
                    index=0,
                )
                destination_name = (
                    destination_map.get(destination_code) if destination_code != destination_placeholder else None
                )
                if destination_name:
                    st.markdown(
                        f'<div class="ff-readonly-chip"><strong>Destination Airport Name:</strong> {destination_name}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="ff-readonly-chip"><strong>Destination Airport Name:</strong> Not selected</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<p class="ff-section-title">Flight Details</p>', unsafe_allow_html=True)
            travel_col1, travel_col2, travel_col3 = st.columns(3)
            with travel_col1:
                airline = st.selectbox("Airline", options=airline_select_options, index=0)
                seat_class = st.selectbox("Cabin Class", options=class_select_options, index=0)
            with travel_col2:
                stopovers = st.selectbox("Stopovers", options=stopovers_select_options, index=0)
                aircraft_type = st.selectbox("Aircraft Type", options=aircraft_select_options, index=0)
            with travel_col3:
                booking_source = st.selectbox("Booking Channel", options=booking_select_options, index=0)
                seasonality = st.selectbox("Seasonality", options=seasonality_select_options, index=0)

            st.markdown('<p class="ff-section-title">Timing Details</p>', unsafe_allow_html=True)
            schedule_col1, schedule_col2 = st.columns(2)
            with schedule_col1:
                default_departure_date = date.today() + timedelta(days=max(days_default, 1))
                departure_date = st.date_input(
                    "Departure Date",
                    value=default_departure_date,
                    min_value=date.today(),
                )
                derived_days_before_departure = max((departure_date - date.today()).days, 1)
                st.metric("Days Before Departure", f"{derived_days_before_departure}")
            with schedule_col2:
                duration_hours = st.slider(
                    "Flight Duration (hours)",
                    min_value=float(duration_min),
                    max_value=float(duration_max),
                    value=float(duration_default),
                    step=0.25,
                )
                inferred_departure_season = _infer_departure_season(departure_date.month)
                departure_season = st.selectbox(
                    "Departure Season",
                    options=departure_season_select_options,
                    index=0,
                    format_func=lambda value: str(value).replace("_", " ").title(),
                )
                st.caption(f"Suggested from date: {inferred_departure_season.replace('_', ' ').title()}")

            st.markdown('<p class="ff-section-title">Derived Schedule Fields</p>', unsafe_allow_html=True)
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.metric("Departure Month", departure_date.month)
            with kpi_col2:
                st.metric("Departure Day", departure_date.day)
            with kpi_col3:
                st.metric("Departure Weekday", departure_date.weekday())

            submit_single = st.form_submit_button(
                "Estimate Fare",
                type="primary",
                use_container_width=True,
            )

        # Keep result directly below the submit button so users can see it immediately.
        st.markdown('<p class="ff-section-title">Prediction Result</p>', unsafe_allow_html=True)
        fare_result_placeholder = st.empty()

        if submit_single:
            missing_fields: list[str] = []
            required_selects = [
                ("Source Airport Code", source_code, source_placeholder),
                ("Destination Airport Code", destination_code, destination_placeholder),
                ("Airline", airline, airline_placeholder),
                ("Cabin Class", seat_class, class_placeholder),
                ("Stopovers", stopovers, stopovers_placeholder),
                ("Aircraft Type", aircraft_type, aircraft_placeholder),
                ("Booking Channel", booking_source, booking_placeholder),
                ("Seasonality", seasonality, seasonality_placeholder),
                ("Departure Season", departure_season, departure_season_placeholder),
            ]
            for field_label, selected_value, placeholder_value in required_selects:
                if selected_value == placeholder_value:
                    missing_fields.append(field_label)

            if missing_fields:
                st.error("Please select values for: " + ", ".join(missing_fields))
                return

            selected_source_name = source_name if source_name else "Not Available"
            selected_destination_name = destination_name if destination_name else "Not Available"

            template_defaults: dict[str, Any] = {}
            if not template_df.empty:
                template_defaults = template_df.iloc[0].to_dict()

            customer_record = {field_name: template_defaults.get(field_name) for field_name in required_field_names}
            customer_record.update(
                {
                    "Duration (hrs)": float(duration_hours),
                    "Days Before Departure": int(derived_days_before_departure),
                    "Departure Month": int(departure_date.month),
                    "Departure Day": int(departure_date.day),
                    "Departure Weekday": int(departure_date.weekday()),
                    "Airline": airline,
                    "Source": source_code,
                    "Source Name": selected_source_name,
                    "Destination": destination_code,
                    "Destination Name": selected_destination_name,
                    "Stopovers": stopovers,
                    "Aircraft Type": aircraft_type,
                    "Class": seat_class,
                    "Booking Source": booking_source,
                    "Seasonality": seasonality,
                    "Departure Season": departure_season,
                }
            )

            try:
                with st.spinner("Estimating fare..."):
                    prediction_payload = request_customer_prediction(
                        api_base_url=api_base_url,
                        record=customer_record,
                        timeout_seconds=float(timeout_seconds),
                        retries=int(retries),
                        retry_delay_seconds=float(retry_delay_seconds),
                        logger=logger,
                    )
                predictions_df = pd.DataFrame(prediction_payload.get("predictions", []))
                if predictions_df.empty:
                    st.warning("Prediction completed, but no rows were returned by the API.")
                else:
                    first_prediction = predictions_df.iloc[0].to_dict()
                    if "predicted_total_fare_bdt" not in first_prediction:
                        st.error("Prediction response missing 'predicted_total_fare_bdt'.")
                        return
                    predicted_fare = first_prediction.get("predicted_total_fare_bdt")
                    if pd.notna(predicted_fare):
                        fare_value = float(predicted_fare)
                        st.session_state["ff_last_predicted_fare_bdt"] = fare_value
                    else:
                        st.warning("Prediction returned an invalid fare value.")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Customer prediction failed: %s", exc)
                st.error(f"Customer prediction failed: {exc}")

        last_fare = st.session_state.get("ff_last_predicted_fare_bdt")
        with fare_result_placeholder.container():
            if last_fare is not None:
                try:
                    parsed_fare = float(last_fare)
                except (TypeError, ValueError):
                    st.error("Predicted fare value could not be displayed.")
                else:
                    st.success(f"Predicted Fare (BDT): {parsed_fare:,.2f}")
            else:
                st.info("Complete the form and click `Estimate Fare` to see the predicted fare.")
