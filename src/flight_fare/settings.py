"""Configuration models for stage-based flight fare project execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_project_root() -> Path:
    """Return the repository root based on this module location."""
    return Path(__file__).resolve().parents[2]


def _read_env_int(name: str, default: int) -> int:
    """Read an integer value from environment variables safely."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        message = f"Environment variable '{name}' must be an integer."
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
        project_root = Path(os.getenv("FF_PROJECT_ROOT", _default_project_root())).resolve()
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
            log_max_bytes=_read_env_int("FF_LOG_MAX_BYTES", 5_000_000),
            log_backup_count=_read_env_int("FF_LOG_BACKUP_COUNT", 5),
            profile_head_rows=_read_env_int("FF_PROFILE_HEAD_ROWS", 5),
        )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not already exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
