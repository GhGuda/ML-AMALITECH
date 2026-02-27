"""Entrypoint script for Streamlit UI connected to the FastAPI service."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_fare.streamlit_app import render_streamlit_app  # noqa: E402


def main() -> None:
    """Render Streamlit app UI for live fare predictions."""
    render_streamlit_app()


if __name__ == "__main__":
    main()
