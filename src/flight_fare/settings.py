"""Configuration models for stage-based flight fare project execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_project_root() -> Path:
    """Return the repository root based on this module location."""
    return Path(__file__).resolve().parents[2]


def read_env_int(name: str, default: int) -> int:
    """Read an integer value from environment variables safely."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        message = f"Environment variable '{name}' must be an integer."
        raise ValueError(message) from exc


def read_env_float(name: str, default: float) -> float:
    """Read a float value from environment variables safely."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        message = f"Environment variable '{name}' must be a float."
        raise ValueError(message) from exc


@dataclass(frozen=True)
class Stage1Settings:
    """Configuration container for Stage 1 pipeline behavior."""

    project_root: Path
    dataset_path: Path
    logs_dir: Path
    artifacts_dir: Path
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    profile_head_rows: int = 5

    @classmethod
    def from_env(cls) -> "Stage1Settings":
        """Build settings from environment variables with safe defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        dataset_default = project_root / "notebook" / "raw_bangladesh_data" / "Flight_Price_Dataset_of_Bangladesh.csv"
        dataset_path = Path(os.getenv("FF_DATASET_PATH", dataset_default)).resolve()
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            dataset_path=dataset_path,
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
            profile_head_rows=read_env_int("FF_PROFILE_HEAD_ROWS", 5),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage2Settings:
    """Configuration container for Stage 2 preprocessing behavior."""

    project_root: Path
    dataset_path: Path
    logs_dir: Path
    artifacts_dir: Path
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    test_size: float = 0.2
    random_state: int = 42
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scaler_strategy: str = "standard"

    @classmethod
    def from_env(cls) -> "Stage2Settings":
        """Build Stage 2 settings from environment variables with defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        dataset_default = project_root / "notebook" / "raw_bangladesh_data" / "Flight_Price_Dataset_of_Bangladesh.csv"
        dataset_path = Path(os.getenv("FF_DATASET_PATH", dataset_default)).resolve()
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            dataset_path=dataset_path,
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
            test_size=read_env_float("FF_STAGE2_TEST_SIZE", 0.2),
            random_state=read_env_int("FF_STAGE2_RANDOM_STATE", 42),
            numeric_impute_strategy=os.getenv("FF_STAGE2_NUMERIC_IMPUTE", "median").strip(),
            categorical_impute_strategy=os.getenv("FF_STAGE2_CATEGORICAL_IMPUTE", "most_frequent").strip(),
            scaler_strategy=os.getenv("FF_STAGE2_SCALER", "standard").strip().lower(),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage3Settings:
    """Configuration container for Stage 3 exploratory analysis behavior."""

    project_root: Path
    dataset_path: Path
    logs_dir: Path
    artifacts_dir: Path
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    top_n_routes: int = 5
    max_airlines_boxplot: int = 15

    @classmethod
    def from_env(cls) -> "Stage3Settings":
        """Build Stage 3 settings from environment variables with defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        dataset_default = project_root / "artifacts" / "stage_2" / "cleaned_dataset.csv"
        dataset_path = Path(os.getenv("FF_STAGE3_DATASET_PATH", dataset_default)).resolve()
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            dataset_path=dataset_path,
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
            top_n_routes=read_env_int("FF_STAGE3_TOP_N_ROUTES", 5),
            max_airlines_boxplot=read_env_int("FF_STAGE3_MAX_AIRLINES_BOXPLOT", 15),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage4Settings:
    """Configuration container for Stage 4 baseline modeling behavior."""

    project_root: Path
    x_train_path: Path
    x_test_path: Path
    y_train_path: Path
    y_test_path: Path
    logs_dir: Path
    artifacts_dir: Path
    target_column: str = "Total Fare (BDT)"
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5

    @classmethod
    def from_env(cls) -> "Stage4Settings":
        """Build Stage 4 settings from environment variables with defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        stage2_dir = project_root / "artifacts" / "stage_2"
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            x_train_path=Path(os.getenv("FF_STAGE4_X_TRAIN_PATH", stage2_dir / "x_train_processed.csv")).resolve(),
            x_test_path=Path(os.getenv("FF_STAGE4_X_TEST_PATH", stage2_dir / "x_test_processed.csv")).resolve(),
            y_train_path=Path(os.getenv("FF_STAGE4_Y_TRAIN_PATH", stage2_dir / "y_train.csv")).resolve(),
            y_test_path=Path(os.getenv("FF_STAGE4_Y_TEST_PATH", stage2_dir / "y_test.csv")).resolve(),
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            target_column=os.getenv("FF_STAGE4_TARGET_COLUMN", "Total Fare (BDT)").strip(),
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage5Settings:
    """Configuration container for Stage 5 advanced model tuning behavior."""

    project_root: Path
    x_train_path: Path
    x_test_path: Path
    y_train_path: Path
    y_test_path: Path
    logs_dir: Path
    artifacts_dir: Path
    target_column: str = "Total Fare (BDT)"
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    cv_folds: int = 3
    random_state: int = 42
    tuning_sample_fraction: float = 0.25
    tree_search_iterations: int = 8
    forest_search_iterations: int = 6
    gb_search_iterations: int = 6
    n_jobs: int = -1

    @classmethod
    def from_env(cls) -> "Stage5Settings":
        """Build Stage 5 settings from environment variables with defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        stage2_dir = project_root / "artifacts" / "stage_2"
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            x_train_path=Path(os.getenv("FF_STAGE5_X_TRAIN_PATH", stage2_dir / "x_train_processed.csv")).resolve(),
            x_test_path=Path(os.getenv("FF_STAGE5_X_TEST_PATH", stage2_dir / "x_test_processed.csv")).resolve(),
            y_train_path=Path(os.getenv("FF_STAGE5_Y_TRAIN_PATH", stage2_dir / "y_train.csv")).resolve(),
            y_test_path=Path(os.getenv("FF_STAGE5_Y_TEST_PATH", stage2_dir / "y_test.csv")).resolve(),
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            target_column=os.getenv("FF_STAGE5_TARGET_COLUMN", "Total Fare (BDT)").strip(),
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
            cv_folds=read_env_int("FF_STAGE5_CV_FOLDS", 3),
            random_state=read_env_int("FF_STAGE5_RANDOM_STATE", 42),
            tuning_sample_fraction=read_env_float("FF_STAGE5_TUNING_SAMPLE_FRACTION", 0.25),
            tree_search_iterations=read_env_int("FF_STAGE5_TREE_SEARCH_ITERATIONS", 8),
            forest_search_iterations=read_env_int("FF_STAGE5_FOREST_SEARCH_ITERATIONS", 6),
            gb_search_iterations=read_env_int("FF_STAGE5_GB_SEARCH_ITERATIONS", 6),
            n_jobs=read_env_int("FF_STAGE5_N_JOBS", -1),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage6Settings:
    """Configuration container for Stage 6 interpretation and insight generation."""

    project_root: Path
    cleaned_dataset_path: Path
    stage5_summary_path: Path
    feature_impact_path: Path
    logs_dir: Path
    artifacts_dir: Path
    target_column: str = "Total Fare (BDT)"
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5
    top_feature_count: int = 20
    top_route_season_count: int = 15
    min_route_season_flights: int = 30

    @classmethod
    def from_env(cls) -> "Stage6Settings":
        """Build Stage 6 settings from environment variables with defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        stage2_dir = project_root / "artifacts" / "stage_2"
        stage5_dir = project_root / "artifacts" / "stage_5"
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            cleaned_dataset_path=Path(
                os.getenv("FF_STAGE6_CLEANED_DATASET_PATH", stage2_dir / "cleaned_dataset.csv")
            ).resolve(),
            stage5_summary_path=Path(
                os.getenv("FF_STAGE6_STAGE5_SUMMARY_PATH", stage5_dir / "stage5_summary_report.json")
            ).resolve(),
            feature_impact_path=Path(
                os.getenv("FF_STAGE6_FEATURE_IMPACT_PATH", stage5_dir / "best_model_feature_impact.csv")
            ).resolve(),
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            target_column=os.getenv("FF_STAGE6_TARGET_COLUMN", "Total Fare (BDT)").strip(),
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
            top_feature_count=read_env_int("FF_STAGE6_TOP_FEATURE_COUNT", 20),
            top_route_season_count=read_env_int("FF_STAGE6_TOP_ROUTE_SEASON_COUNT", 15),
            min_route_season_flights=read_env_int("FF_STAGE6_MIN_ROUTE_SEASON_FLIGHTS", 30),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage7Settings:
    """Configuration container for Stage 7 packaging and delivery behavior."""

    project_root: Path
    best_model_path: Path
    preprocessor_path: Path
    stage5_summary_path: Path
    preprocessing_report_path: Path
    cleaned_dataset_path: Path
    logs_dir: Path
    artifacts_dir: Path
    package_version: str = "v1"
    smoke_test_rows: int = 20
    log_file_name: str = "flight_fare_pipeline.log"
    log_level: str = "INFO"
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 5

    @classmethod
    def from_env(cls) -> "Stage7Settings":
        """Build Stage 7 settings from environment variables with defaults."""
        project_root = Path(os.getenv("FF_PROJECT_ROOT", default_project_root())).resolve()
        stage2_dir = project_root / "artifacts" / "stage_2"
        stage5_dir = project_root / "artifacts" / "stage_5"
        logs_dir = Path(os.getenv("FF_LOGS_DIR", project_root / "logs")).resolve()
        artifacts_dir = Path(os.getenv("FF_ARTIFACTS_DIR", project_root / "artifacts")).resolve()

        return cls(
            project_root=project_root,
            best_model_path=Path(os.getenv("FF_STAGE7_BEST_MODEL_PATH", stage5_dir / "best_model.pkl")).resolve(),
            preprocessor_path=Path(os.getenv("FF_STAGE7_PREPROCESSOR_PATH", stage2_dir / "preprocessor.pkl")).resolve(),
            stage5_summary_path=Path(
                os.getenv("FF_STAGE7_STAGE5_SUMMARY_PATH", stage5_dir / "stage5_summary_report.json")
            ).resolve(),
            preprocessing_report_path=Path(
                os.getenv("FF_STAGE7_PREPROCESSING_REPORT_PATH", stage2_dir / "preprocessing_report.json")
            ).resolve(),
            cleaned_dataset_path=Path(
                os.getenv("FF_STAGE7_CLEANED_DATASET_PATH", stage2_dir / "cleaned_dataset.csv")
            ).resolve(),
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            package_version=os.getenv("FF_STAGE7_PACKAGE_VERSION", "v1").strip(),
            smoke_test_rows=read_env_int("FF_STAGE7_SMOKE_TEST_ROWS", 20),
            log_file_name=os.getenv("FF_LOG_FILE_NAME", "flight_fare_pipeline.log"),
            log_level=os.getenv("FF_LOG_LEVEL", "INFO").upper(),
            log_max_bytes=read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=read_env_int("FF_LOG_BACKUP_COUNT", 5),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
