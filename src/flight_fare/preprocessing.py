"""Data cleaning, feature engineering, and preprocessing for Stage 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

TARGET_COLUMN = "Total Fare (BDT)"
DATE_COLUMN = "Departure Date & Time"
ARRIVAL_DATE_COLUMN = "Arrival Date & Time"
FARE_COLUMNS = ("Base Fare (BDT)", "Tax & Surcharge (BDT)", TARGET_COLUMN)
POTENTIAL_NUMERIC_COLUMNS = ("Base Fare (BDT)", "Tax & Surcharge (BDT)", "Duration (hrs)", "Days Before Departure")
POTENTIAL_CATEGORICAL_COLUMNS = (
    "Airline",
    "Source",
    "Source Name",
    "Destination",
    "Destination Name",
    "Stopovers",
    "Aircraft Type",
    "Class",
    "Booking Source",
    "Seasonality",
)
CITY_NORMALIZATION_MAP = {
    "dacca": "Dhaka",
    "chittagong": "Chattogram",
    "barisal": "Barishal",
    "sylhet ": "Sylhet",
}


@dataclass(frozen=True)
class CleaningReport:
    """Summary of cleaning outcomes for auditability."""

    input_rows: int
    output_rows: int
    dropped_duplicates: int
    dropped_invalid_target_rows: int
    dropped_invalid_datetime_rows: int
    dropped_irrelevant_columns: list[str]
    negative_values_replaced: dict[str, int]
    newly_introduced_numeric_nulls: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary serialization for JSON report persistence."""
        return asdict(self)


@dataclass(frozen=True)
class PreprocessingSplit:
    """Container for transformed train/test data and fitted preprocessor."""

    x_train_processed: pd.DataFrame
    x_test_processed: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]


def month_to_season(month: int) -> str:
    """Map month number to a season label relevant to Bangladesh travel patterns."""
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "pre_monsoon"
    if month in {6, 7, 8, 9}:
        return "monsoon"
    return "post_monsoon"


def normalize_city_name(value: Any) -> Any:
    """Normalize city names with a lightweight mapping and clean formatting."""
    if pd.isna(value):
        return np.nan
    text_value = str(value).strip()
    if not text_value:
        return np.nan
    normalized = CITY_NORMALIZATION_MAP.get(text_value.lower(), text_value)
    return normalized.title()


def normalize_airport_code(value: Any) -> Any:
    """Normalize airport/city codes to uppercase without surrounding spaces."""
    if pd.isna(value):
        return np.nan
    text_value = str(value).strip()
    if not text_value:
        return np.nan
    return text_value.upper()


def drop_irrelevant_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop generic irrelevant columns such as unnamed index artifacts."""
    dropped_columns: list[str] = []
    filtered_df = dataframe.copy()
    for column in list(filtered_df.columns):
        normalized = column.strip().lower()
        if normalized.startswith("unnamed") or normalized == "index":
            dropped_columns.append(column)
    if dropped_columns:
        filtered_df = filtered_df.drop(columns=dropped_columns)
    return filtered_df, dropped_columns


def clean_and_engineer_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean raw flight fare data and engineer deterministic time-based features."""
    working_df = dataframe.copy()
    input_rows = int(len(working_df))

    # Remove non-business columns often introduced by CSV exports.
    working_df, dropped_irrelevant_columns = drop_irrelevant_columns(working_df)

    duplicate_count = int(working_df.duplicated().sum())
    working_df = working_df.drop_duplicates().reset_index(drop=True)

    # Normalize airport codes and city names for better category consistency.
    if "Source" in working_df.columns:
        working_df["Source"] = working_df["Source"].apply(normalize_airport_code)
    if "Destination" in working_df.columns:
        working_df["Destination"] = working_df["Destination"].apply(normalize_airport_code)
    if "Source Name" in working_df.columns:
        working_df["Source Name"] = working_df["Source Name"].apply(normalize_city_name)
    if "Destination Name" in working_df.columns:
        working_df["Destination Name"] = working_df["Destination Name"].apply(normalize_city_name)

    negative_values_replaced: dict[str, int] = {}
    newly_introduced_numeric_nulls: dict[str, int] = {}
    numeric_columns_to_coerce = set(POTENTIAL_NUMERIC_COLUMNS).union(FARE_COLUMNS)
    for column in numeric_columns_to_coerce:
        if column not in working_df.columns:
            continue
        null_count_before = int(working_df[column].isna().sum())
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
        null_count_after = int(working_df[column].isna().sum())
        newly_introduced_numeric_nulls[column] = max(0, null_count_after - null_count_before)

    for column in FARE_COLUMNS:
        if column not in working_df.columns:
            continue
        negative_mask = working_df[column] < 0
        negative_count = int(negative_mask.sum())
        negative_values_replaced[column] = negative_count
        if negative_count > 0:
            working_df.loc[negative_mask, column] = np.nan

    if TARGET_COLUMN in working_df.columns:
        missing_total_fare = working_df[TARGET_COLUMN].isna()
        can_reconstruct_total = missing_total_fare & working_df["Base Fare (BDT)"].notna() & working_df["Tax & Surcharge (BDT)"].notna()
        working_df.loc[can_reconstruct_total, TARGET_COLUMN] = (
            working_df.loc[can_reconstruct_total, "Base Fare (BDT)"]
            + working_df.loc[can_reconstruct_total, "Tax & Surcharge (BDT)"]
        )

    if DATE_COLUMN in working_df.columns:
        working_df[DATE_COLUMN] = pd.to_datetime(working_df[DATE_COLUMN], errors="coerce")
    if ARRIVAL_DATE_COLUMN in working_df.columns:
        working_df[ARRIVAL_DATE_COLUMN] = pd.to_datetime(working_df[ARRIVAL_DATE_COLUMN], errors="coerce")

    invalid_datetime_rows = int(working_df[DATE_COLUMN].isna().sum()) if DATE_COLUMN in working_df.columns else 0
    if DATE_COLUMN in working_df.columns:
        working_df = working_df.dropna(subset=[DATE_COLUMN]).reset_index(drop=True)

    dropped_invalid_target_rows = int(working_df[TARGET_COLUMN].isna().sum()) if TARGET_COLUMN in working_df.columns else 0
    if TARGET_COLUMN in working_df.columns:
        working_df = working_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    if DATE_COLUMN in working_df.columns:
        working_df["Departure Month"] = working_df[DATE_COLUMN].dt.month.astype(int)
        working_df["Departure Day"] = working_df[DATE_COLUMN].dt.day.astype(int)
        working_df["Departure Weekday"] = working_df[DATE_COLUMN].dt.weekday.astype(int)
        working_df["Departure Season"] = working_df["Departure Month"].apply(month_to_season)

    output_rows = int(len(working_df))
    report = CleaningReport(
        input_rows=input_rows,
        output_rows=output_rows,
        dropped_duplicates=duplicate_count,
        dropped_invalid_target_rows=dropped_invalid_target_rows,
        dropped_invalid_datetime_rows=invalid_datetime_rows,
        dropped_irrelevant_columns=dropped_irrelevant_columns,
        negative_values_replaced=negative_values_replaced,
        newly_introduced_numeric_nulls=newly_introduced_numeric_nulls,
    )
    return working_df, report


def select_modeling_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature frame and regression target for modeling."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    y = dataframe[TARGET_COLUMN].astype(float)
    x = dataframe.drop(columns=[TARGET_COLUMN])

    # Raw timestamp columns are replaced by engineered features.
    for date_column in (DATE_COLUMN, ARRIVAL_DATE_COLUMN):
        if date_column in x.columns:
            x = x.drop(columns=[date_column])
    return x, y


def build_one_hot_encoder() -> OneHotEncoder:
    """Create an OneHotEncoder compatible with multiple scikit-learn versions."""
    # Support both new and legacy constructor APIs for sparse output.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    scaler_strategy: str = "standard",
) -> ColumnTransformer:
    """Build a column transformer with robust imputing, encoding, and scaling."""
    if scaler_strategy not in {"standard", "minmax"}:
        raise ValueError("scaler_strategy must be either 'standard' or 'minmax'.")

    scaler = StandardScaler() if scaler_strategy == "standard" else MinMaxScaler()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_impute_strategy)),
            ("scaler", scaler),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
            ("encoder", build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def split_and_preprocess(
    dataframe: pd.DataFrame,
    test_size: float,
    random_state: int,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    scaler_strategy: str = "standard",
) -> PreprocessingSplit:
    """Split cleaned data and produce fully transformed train/test matrices."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(dataframe) < 5:
        raise ValueError("At least 5 rows are required for train/test split.")

    x, y = select_modeling_features(dataframe)
    numeric_columns = [column for column in x.columns if pd.api.types.is_numeric_dtype(x[column])]
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = create_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_impute_strategy=categorical_impute_strategy,
        scaler_strategy=scaler_strategy,
    )
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    x_train_df = pd.DataFrame(x_train_processed, columns=feature_names, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_processed, columns=feature_names, index=x_test.index)

    return PreprocessingSplit(
        x_train_processed=x_train_df,
        x_test_processed=x_test_df,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        numeric_feature_columns=numeric_columns,
        categorical_feature_columns=categorical_columns,
    )
