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
