"""Visualization plots and Markdown/JSON evaluation report generation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import auc, roc_curve

from mri_tumor.evaluation.metrics import (
    _validate_integer_targets,
    _validate_probabilities,
    compute_calibration_bins,
    compute_ece,
)
from mri_tumor.explain.gradcam import _to_rgb_uint8, overlay_gradcam


def _save_figure(fig: Figure, output_path: str | Path) -> Path:
    """Renders a Matplotlib Figure using headless Agg canvas and closes it cleanly."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    try:
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
    finally:
        fig.clf()
        fig.set_canvas(None)
    return out


def plot_confusion_matrix(
    cm: np.ndarray | Sequence[Sequence[int | float]],
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Path:
    """Plots a confusion matrix (or 4x5 cascade routing matrix) and saves it to disk."""
    cm_arr = np.asarray(cm, dtype=np.float64)
    if cm_arr.ndim != 2 or cm_arr.shape[0] == 0 or cm_arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D confusion matrix, got shape {cm_arr.shape}.")
    if not np.all(np.isfinite(cm_arr)):
        raise ValueError("cm must contain only finite values.")
    if np.any(cm_arr < 0.0):
        raise ValueError("cm must contain non-negative values.")

    names = [str(name) for name in class_names]
    if cm_arr.shape[1] != len(names):
        raise ValueError(
            f"Column count ({cm_arr.shape[1]}) must match len(class_names) ({len(names)})."
        )
    if cm_arr.shape[0] == len(names):
        y_names = names
    elif cm_arr.shape[0] < len(names):
        y_names = names[: cm_arr.shape[0]]
    else:
        raise ValueError(
            f"Row count ({cm_arr.shape[0]}) exceeds len(class_names) ({len(names)})."
        )

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        display_arr = np.divide(
            cm_arr,
            np.maximum(row_sums, 1e-12),
            out=np.zeros_like(cm_arr),
            where=row_sums > 0,
        )
    else:
        display_arr = cm_arr

    fig = Figure(figsize=(max(5.0, 1.3 * len(names)), max(4.5, 1.1 * len(y_names))))
    ax = fig.add_subplot(111)
    im = ax.imshow(display_arr, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(y_names)),
        xticklabels=names,
        yticklabels=y_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    ax.tick_params(axis="x", labelrotation=25)

    vmax = float(np.max(display_arr)) if display_arr.size > 0 else 0.0
    thresh = vmax * 0.5
    for row_idx in range(display_arr.shape[0]):
        for col_idx in range(display_arr.shape[1]):
            val = float(display_arr[row_idx, col_idx])
            text = f"{val:.2f}" if normalize else f"{int(round(val))}"
            color = "white" if val > thresh and vmax > 0.0 else "#111111"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=10)

    return _save_figure(fig, output_path)


def plot_roc_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "ROC Curves",
) -> Path:
    """Plots binary or One-vs-Rest multi-class ROC curves with AUROC scores."""
    probs_arr = _validate_probabilities(probs, name="probs")
    names = [str(name) for name in class_names]

    fig = Figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)

    if probs_arr.ndim == 1 or (probs_arr.ndim == 2 and probs_arr.shape[1] == 1):
        probs_1d = probs_arr.ravel()
        targets_arr = _validate_integer_targets(targets, num_classes=2)
        if targets_arr.shape[0] != probs_1d.shape[0]:
            raise ValueError("targets and probs sample counts do not match.")
        if len(names) > 2:
            raise ValueError(
                f"Expected at most 2 class_names for binary ROC curves, got {len(names)}."
            )
        pos_name = names[-1] if names else "positive"
        if np.unique(targets_arr).size >= 2:
            fpr, tpr, _ = roc_curve(targets_arr, probs_1d)
            roc_auc = float(auc(fpr, tpr))
            ax.plot(fpr, tpr, lw=2.0, label=f"{pos_name} (AUC = {roc_auc:.3f})")
        else:
            ax.plot([], [], lw=2.0, label=f"{pos_name} (AUC = N/A: single class)")
    elif probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
        num_classes = probs_arr.shape[1]
        if len(names) != num_classes:
            raise ValueError(
                f"len(class_names) ({len(names)}) must match probs.shape[1] ({num_classes})."
            )
        targets_arr = _validate_integer_targets(targets, num_classes=num_classes)
        if targets_arr.shape[0] != probs_arr.shape[0]:
            raise ValueError("targets and probs sample counts do not match.")

        if num_classes == 2:
            if np.unique(targets_arr).size >= 2:
                fpr, tpr, _ = roc_curve(targets_arr, probs_arr[:, 1])
                roc_auc = float(auc(fpr, tpr))
                ax.plot(fpr, tpr, lw=2.0, label=f"{names[1]} (AUC = {roc_auc:.3f})")
            else:
                ax.plot([], [], lw=2.0, label=f"{names[1]} (AUC = N/A: single class)")
        else:
            aucs: list[float] = []
            for idx, name in enumerate(names):
                bin_true = (targets_arr == idx).astype(np.int64)
                if np.any(bin_true == 1) and np.any(bin_true == 0):
                    fpr, tpr, _ = roc_curve(bin_true, probs_arr[:, idx])
                    roc_auc = float(auc(fpr, tpr))
                    aucs.append(roc_auc)
                    ax.plot(fpr, tpr, lw=2.0, label=f"{name} (AUC = {roc_auc:.3f})")
                else:
                    ax.plot([], [], lw=2.0, label=f"{name} (AUC = N/A)")
            if aucs:
                ax.plot([], [], color="none", label=f"Macro AUC = {float(np.mean(aucs)):.3f}")
    else:
        raise ValueError(f"Unsupported probs shape {probs_arr.shape}.")

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", lw=1.2, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return _save_figure(fig, output_path)


def plot_calibration_curve(
    targets: np.ndarray,
    probs: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> Path:
    """Plots a confidence reliability diagram annotated with Expected Calibration Error (ECE)."""
    bins = compute_calibration_bins(probs=probs, targets=targets, n_bins=n_bins)
    ece = compute_ece(probs=probs, targets=targets, n_bins=n_bins)

    non_empty = [b for b in bins if int(b["count"]) > 0]
    confs = [float(b["confidence"]) for b in non_empty]
    accs = [float(b["accuracy"]) for b in non_empty]

    fig = Figure(figsize=(6.0, 5.5))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", lw=1.2, label="Perfect Calibration")

    if non_empty:
        centers = [0.5 * (float(b["bin_low"]) + float(b["bin_high"])) for b in non_empty]
        width = 0.85 / float(max(1, n_bins))
        ax.bar(
            centers,
            accs,
            width=width,
            alpha=0.35,
            edgecolor="#1f77b4",
            color="#1f77b4",
            label="Bin Accuracy",
        )
        ax.plot(confs, accs, marker="o", color="#d62728", lw=2.0, label=f"Model (ECE = {ece:.4f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    return _save_figure(fig, output_path)


def plot_ablation_comparison(
    results: dict[str, dict[str, float]],
    metrics: Sequence[str],
    output_path: str | Path,
    title: str = "Model Ablation Comparison",
) -> Path:
    """Plots grouped bar comparison across model variants and selected evaluation metrics."""
    if not results:
        raise ValueError("results must not be empty.")
    metric_keys = [str(m) for m in metrics]
    if not metric_keys:
        raise ValueError("metrics must not be empty.")

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)
    x = np.arange(n_models, dtype=np.float64)
    total_width = 0.78
    bar_width = total_width / float(n_metrics)

    fig = Figure(figsize=(max(6.5, 1.8 * n_models), 5.0))
    ax = fig.add_subplot(111)

    all_vals: list[float] = []
    for m_idx, metric_name in enumerate(metric_keys):
        vals: list[float] = []
        labels_txt: list[str] = []
        for model in model_names:
            raw_val = results[model].get(metric_name, 0.0)
            if raw_val is None:
                vals.append(0.0)
                labels_txt.append("N/A")
            else:
                val_f = float(raw_val)
                if math.isfinite(val_f):
                    vals.append(val_f)
                    labels_txt.append(f"{val_f:.3f}")
                else:
                    vals.append(0.0)
                    labels_txt.append("N/A")

        all_vals.extend(vals)
        offset = (m_idx - 0.5 * (n_metrics - 1)) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width, label=metric_name)
        for bar, val, txt in zip(bars, vals, labels_txt):
            y_pos = bar.get_height() + 0.015 if val >= 0.0 else bar.get_height() - 0.015
            va_align = "bottom" if val >= 0.0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                y_pos,
                txt,
                ha="center",
                va=va_align,
                fontsize=8,
            )

    min_v = min(all_vals) if all_vals else 0.0
    max_v = max(all_vals) if all_vals else 1.0
    if 0.0 <= min_v and max_v <= 1.0:
        ax.set_ylim(0.0, 1.15)
    else:
        ax.set_ylim(min(0.0, min_v * 1.1), max_v * 1.18 if max_v > 0 else 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return _save_figure(fig, output_path)


def _prepare_display_image(image: Any) -> np.ndarray:
    """Converts (C, H, W) or (H, W, C) tensor/array into uint8 RGB (H, W, 3) for display."""
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return _to_rgb_uint8(arr)


def plot_gradcam_examples(
    images: Sequence[Any],
    masks: Sequence[Any | None] | None,
    heatmaps: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Plots a multi-row grid of Original MRI | Ground-Truth Mask | Grad-CAM Overlay."""
    n_items = len(images)
    if n_items == 0:
        raise ValueError("images must not be empty.")
    if len(heatmaps) != n_items or len(titles) != n_items:
        raise ValueError(
            f"Length mismatch: images={n_items}, heatmaps={len(heatmaps)}, titles={len(titles)}."
        )
    if masks is not None and len(masks) != n_items:
        raise ValueError(f"masks length ({len(masks)}) must match images ({n_items}).")

    fig = Figure(figsize=(10.5, 3.3 * n_items))
    axes = fig.subplots(n_items, 3, squeeze=False)
    col_headers = ("Original MRI", "Ground-Truth Mask", "Grad-CAM Overlay")

    for row_idx in range(n_items):
        img_rgb = _prepare_display_image(images[row_idx])
        hm = np.asarray(heatmaps[row_idx], dtype=np.float32)
        overlay = overlay_gradcam(img_rgb, hm)
        raw_mask = masks[row_idx] if masks is not None else None

        ax_orig, ax_mask, ax_cam = axes[row_idx]
        ax_orig.imshow(img_rgb)
        ax_orig.set_ylabel(str(titles[row_idx]), fontsize=9)

        if raw_mask is not None:
            mask_np = raw_mask.detach().cpu().numpy() if hasattr(raw_mask, "detach") else np.asarray(raw_mask)
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            elif mask_np.ndim == 3 and mask_np.shape[2] == 1:
                mask_np = mask_np[:, :, 0]
            if mask_np.ndim != 2:
                raise ValueError(f"Expected 2D mask after squeezing singleton channel, got shape {mask_np.shape}.")
            ax_mask.imshow((mask_np > 0).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax_mask.imshow(np.zeros(img_rgb.shape[:2], dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)
            ax_mask.text(
                0.5,
                0.5,
                "No Mask",
                transform=ax_mask.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

        ax_cam.imshow(overlay)

        for col_idx, ax in enumerate((ax_orig, ax_mask, ax_cam)):
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=10)

    return _save_figure(fig, output_path)


def _to_serializable(obj: Any) -> Any:
    """Recursively converts dataclasses, numpy types, and Paths into JSON-safe Python values."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _to_serializable(obj.to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_serializable(asdict(obj))
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _to_serializable(obj.item())
        return [_to_serializable(item) for item in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    return str(obj)


def _fmt(val: Any, digits: int = 4) -> str:
    """Formats scalar metrics cleanly for Markdown tables."""
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.{digits}f}" if math.isfinite(val) else "N/A"
    return str(val)


def _render_kv_table(rows: Sequence[tuple[str, Any]]) -> list[str]:
    """Formats a 2-column Metric | Value Markdown table."""
    lines = ["| Metric | Value |", "| :--- | :--- |"]
    for key, val in rows:
        lines.append(f"| {key} | {_fmt(val)} |")
    return lines


def generate_evaluation_report(
    output_dir: str | Path,
    stage1_metrics: Any = None,
    stage2_metrics: Any = None,
    cascade_metrics: Any = None,
    gradcam_metrics: dict[str, Any] | None = None,
    ablation_table: dict[str, dict[str, float]] | None = None,
    plot_paths: dict[str, str | Path] | None = None,
    title: str = "MRITumorDetection-CNN Evaluation Report",
) -> Path:
    """Writes ``evaluation_report.md`` and ``metrics_summary.json`` into ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"title": str(title)}
    if stage1_metrics is not None:
        summary["stage1"] = _to_serializable(stage1_metrics)
    if stage2_metrics is not None:
        summary["stage2"] = _to_serializable(stage2_metrics)
    if cascade_metrics is not None:
        summary["cascade"] = _to_serializable(cascade_metrics)
    if gradcam_metrics is not None:
        summary["gradcam"] = _to_serializable(gradcam_metrics)
    if ablation_table is not None:
        summary["ablation"] = _to_serializable(ablation_table)
    if plot_paths is not None:
        summary["plots"] = {str(k): Path(v).as_posix() for k, v in plot_paths.items()}

    json_path = out_dir / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    lines: list[str] = [f"# {title}", ""]

    if "stage1" in summary:
        s1 = summary["stage1"]
        lines.extend(["## Stage 1: Binary Tumor Screening", ""])
        lines.extend(
            _render_kv_table(
                [
                    ("Samples", s1.get("n_samples")),
                    ("Decision Threshold", s1.get("threshold")),
                    ("Accuracy", s1.get("accuracy")),
                    ("Balanced Accuracy", s1.get("balanced_accuracy")),
                    ("Sensitivity (Recall+)", s1.get("sensitivity")),
                    ("Specificity", s1.get("specificity")),
                    ("Precision", s1.get("precision")),
                    ("NPV", s1.get("npv")),
                    ("F1 Score", s1.get("f1")),
                    ("AUROC", s1.get("auroc")),
                    ("AUPRC", s1.get("auprc")),
                ]
            )
        )
        lines.append("")

    if "stage2" in summary:
        s2 = summary["stage2"]
        lines.extend(["## Stage 2: Subtype Classification & Calibration", ""])
        s2_rows: list[tuple[str, Any]] = [
            ("Samples", s2.get("n_samples")),
            ("Accuracy", s2.get("accuracy")),
            ("Balanced Accuracy", s2.get("balanced_accuracy")),
            ("Macro F1", s2.get("macro_f1")),
            ("Weighted F1", s2.get("weighted_f1")),
            ("Macro AUROC (OvR)", s2.get("macro_auroc")),
            ("ECE", s2.get("ece")),
            ("Brier Score", s2.get("brier_score")),
        ]
        if s2.get("uncertainty_threshold") is not None:
            s2_rows.extend(
                [
                    ("Uncertainty Threshold", s2.get("uncertainty_threshold")),
                    ("Abstention Rate", s2.get("abstention_rate")),
                    ("Selective Accuracy", s2.get("selective_accuracy")),
                    ("Selective Macro F1", s2.get("selective_macro_f1")),
                ]
            )
        lines.extend(_render_kv_table(s2_rows))
        lines.append("")

        per_class = s2.get("per_class", {})
        if per_class:
            lines.extend(
                [
                    "### Stage 2 Per-Class Metrics",
                    "",
                    "| Class | Precision | Recall | F1 | Support |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for cls_name, cls_stats in per_class.items():
                lines.append(
                    f"| {cls_name} | {_fmt(cls_stats.get('precision'))} | "
                    f"{_fmt(cls_stats.get('recall'))} | {_fmt(cls_stats.get('f1'))} | "
                    f"{_fmt(cls_stats.get('support'))} |"
                )
            lines.append("")

    if "cascade" in summary:
        cas = summary["cascade"]
        lines.extend(["## End-to-End Cascade Evaluation", ""])
        lines.extend(
            _render_kv_table(
                [
                    ("Samples", cas.get("n_samples")),
                    ("Overall 4-Way Accuracy", cas.get("overall_4way_accuracy")),
                    ("Overall 4-Way Macro F1", cas.get("overall_4way_macro_f1")),
                    ("Selective 4-Way Accuracy", cas.get("selective_4way_accuracy")),
                    ("Selective 4-Way Macro F1", cas.get("selective_4way_macro_f1")),
                    ("Abstention Rate", cas.get("abstention_rate")),
                    ("Cascade Tumor Sensitivity", cas.get("cascade_tumor_sensitivity")),
                    ("Cascade Healthy Specificity", cas.get("cascade_healthy_specificity")),
                ]
            )
        )
        lines.append("")

        routing = cas.get("routing_matrix_4x5", [])
        cols = cas.get("routing_columns", [])
        rows = cas.get("class_names", [])
        if routing and cols and rows:
            lines.extend(["### 4x5 Clinical Routing Matrix", ""])
            header = "| True \\ Routed | " + " | ".join(str(c) for c in cols) + " |"
            sep = "| :--- | " + " | ".join(":---" for _ in cols) + " |"
            lines.extend([header, sep])
            for r_name, r_vals in zip(rows, routing):
                lines.append(f"| {r_name} | " + " | ".join(str(v) for v in r_vals) + " |")
            lines.append("")

    if "gradcam" in summary:
        gc = summary["gradcam"]
        lines.extend(["## Grad-CAM Localization Overlap", ""])
        lines.extend(
            _render_kv_table(
                [
                    ("Evaluated Masked Samples", gc.get("n_evaluated")),
                    ("Pointing Game Hit Rate", gc.get("pointing_game_hit_rate")),
                    ("Mean Energy Inside Mask", gc.get("mean_energy_in_mask")),
                    ("Median Energy Inside Mask", gc.get("median_energy_in_mask")),
                    ("Std Energy Inside Mask", gc.get("std_energy_in_mask")),
                ]
            )
        )
        lines.append("")

    if "ablation" in summary and summary["ablation"]:
        ab = summary["ablation"]
        metric_cols: list[str] = []
        for model_stats in ab.values():
            for key in model_stats.keys():
                if key not in metric_cols:
                    metric_cols.append(key)
        lines.extend(["## Model Ablation Comparison", ""])
        lines.append("| Model | " + " | ".join(metric_cols) + " |")
        lines.append("| :--- | " + " | ".join(":---" for _ in metric_cols) + " |")
        for model_name, model_stats in ab.items():
            row_str = " | ".join(_fmt(model_stats.get(col)) for col in metric_cols)
            lines.append(f"| {model_name} | {row_str} |")
        lines.append("")

    if plot_paths:
        lines.extend(["## Figures", ""])
        out_resolved = out_dir.resolve()
        for label, raw_path in plot_paths.items():
            p = Path(raw_path)
            try:
                rel = p.resolve().relative_to(out_resolved).as_posix()
            except ValueError:
                rel = p.as_posix()
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{label}]({rel})")
            lines.append("")

    report_path = out_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
