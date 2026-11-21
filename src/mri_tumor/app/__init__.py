"""Application entry points: CLI and Gradio UI."""

from mri_tumor.app.cli import main
from mri_tumor.app.web_ui import (
    RESEARCH_DISCLAIMER,
    analyze_scan,
    build_gradio_app,
    launch_ui,
)

__all__ = [
    "RESEARCH_DISCLAIMER",
    "analyze_scan",
    "build_gradio_app",
    "launch_ui",
    "main",
]
