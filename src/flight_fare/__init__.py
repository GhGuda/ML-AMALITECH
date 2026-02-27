"""Flight fare prediction package."""

from .settings import (
    ApiSettings,
    Stage1Settings,
    Stage2Settings,
    Stage3Settings,
    Stage4Settings,
    Stage5Settings,
    Stage6Settings,
    Stage7Settings,
    StreamlitSettings,
)
from .stage1_pipeline import Stage1Outputs, run_stage_1
from .stage2_pipeline import Stage2Outputs, run_stage_2
from .stage3_eda import Stage3Outputs, run_stage_3
from .stage4_baseline import Stage4Outputs, run_stage_4
from .stage5_advanced import Stage5Outputs, run_stage_5
from .stage6_interpretation import Stage6Outputs, run_stage_6
from .stage7_delivery import Stage7Outputs, run_stage_7

try:
    from .api_server import create_server, run_api_server
except ImportError as exc:
    _api_import_error = exc

    def create_server(*args, **kwargs):  # type: ignore[no-redef]
        """Raise a clear error when optional API dependencies are missing."""
        raise ImportError(
            "API dependencies are missing. Install FastAPI/Uvicorn to use create_server()."
        ) from _api_import_error

    def run_api_server(*args, **kwargs):  # type: ignore[no-redef]
        """Raise a clear error when optional API dependencies are missing."""
        raise ImportError(
            "API dependencies are missing. Install FastAPI/Uvicorn to use run_api_server()."
        ) from _api_import_error

try:
    from .streamlit_app import render_streamlit_app
except ImportError as exc:
    _streamlit_import_error = exc

    def render_streamlit_app(*args, **kwargs):  # type: ignore[no-redef]
        """Raise a clear error when optional Streamlit dependency is missing."""
        raise ImportError(
            "Streamlit dependency is missing. Install streamlit to use render_streamlit_app()."
        ) from _streamlit_import_error

__all__ = [
    "Stage1Outputs",
    "Stage1Settings",
    "Stage2Outputs",
    "Stage2Settings",
    "Stage3Outputs",
    "Stage3Settings",
    "Stage4Outputs",
    "Stage4Settings",
    "Stage5Outputs",
    "Stage5Settings",
    "Stage6Outputs",
    "Stage6Settings",
    "Stage7Outputs",
    "Stage7Settings",
    "ApiSettings",
    "StreamlitSettings",
    "run_stage_1",
    "run_stage_2",
    "run_stage_3",
    "run_stage_4",
    "run_stage_5",
    "run_stage_6",
    "run_stage_7",
    "render_streamlit_app",
    "create_server",
    "run_api_server",
]
