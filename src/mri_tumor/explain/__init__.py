"""Explainability and attribution modules (Grad-CAM, masking overlap metrics)."""

from mri_tumor.explain.gradcam import (
    GradCAM,
    GradCAMResult,
    compute_energy_inside_mask,
    compute_pointing_game_hit,
    evaluate_gradcam_against_masks,
    overlay_gradcam,
)

__all__ = [
    "GradCAMResult",
    "GradCAM",
    "compute_pointing_game_hit",
    "compute_energy_inside_mask",
    "evaluate_gradcam_against_masks",
    "overlay_gradcam",
]
