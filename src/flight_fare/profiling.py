"""Dataset profiling utilities for Stage 1 data understanding outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    """Convert non-JSON-native values to serializable Python primitives."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def detect_outliers_iqr(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count outliers per numeric feature using the IQR rule."""
    outlier_counts: dict[str, int] = {}
    numeric_df = dataframe.select_dtypes(include=["number"])
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            outlier_counts[column] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        mask = (series < lower_bound) | (series > upper_bound)
        outlier_counts[column] = int(mask.sum())
    return outlier_counts


def generate_initial_profile(
    dataframe: pd.DataFrame,
    logger: logging.Logger,
    head_rows: int = 5,
) -> dict[str, Any]:
    """Generate a structured profile for initial dataset understanding."""
    logger.info("Generating Stage 1 dataset profile.")

    numeric_df = dataframe.select_dtypes(include=["number"])
    missing_counts = dataframe.isna().sum()

    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
        "columns": dataframe.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "head": dataframe.head(head_rows).to_dict(orient="records"),
        "missing": {
            "count": missing_counts.to_dict(),
            "percentage": ((missing_counts / max(len(dataframe), 1)) * 100).round(4).to_dict(),
        },
        "duplicates": {
            "count": int(dataframe.duplicated().sum()),
            "percentage": round((dataframe.duplicated().mean() * 100), 4),
        },
        "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else {},
        "outlier_flags_iqr": detect_outliers_iqr(dataframe),
    }
    return profile


def save_profile(profile: dict[str, Any], output_path: Path) -> Path:
    """Persist the generated profile to a JSON file on disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(profile, file_obj, indent=2, default=json_default)
    return output_path
