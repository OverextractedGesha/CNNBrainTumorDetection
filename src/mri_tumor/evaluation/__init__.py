"""Evaluation metrics, calibration curves, threshold analysis, and reporting."""

from mri_tumor.evaluation.metrics import (
    CascadeMetrics,
    Stage1Metrics,
    Stage2Metrics,
    compute_calibration_bins,
    compute_cascade_metrics,
    compute_ece,
    compute_stage1_metrics,
    compute_stage2_metrics,
)
from mri_tumor.evaluation.report import (
    generate_evaluation_report,
    plot_ablation_comparison,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_gradcam_examples,
    plot_roc_curves,
)

__all__ = [
    "Stage1Metrics",
    "Stage2Metrics",
    "CascadeMetrics",
    "compute_calibration_bins",
    "compute_ece",
    "compute_stage1_metrics",
    "compute_stage2_metrics",
    "compute_cascade_metrics",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_calibration_curve",
    "plot_ablation_comparison",
    "plot_gradcam_examples",
    "generate_evaluation_report",
]
