"""Interactive Gradio web interface for MRITumorDetection-CNN cascade inference."""

from __future__ import annotations

import html
import math
from pathlib import Path
import threading
from typing import Any
import weakref

import cv2
import gradio as gr
import numpy as np
import torch

from mri_tumor.data.figshare import read_figshare_mat
from mri_tumor.data.preprocess import crop_brain_contour
from mri_tumor.explain.gradcam import _to_rgb_uint8
from mri_tumor.inference.cascade import CascadePredictor, CascadeResult, load_yaml_config
from mri_tumor.training.calibrate import ABSTAIN_LABEL

RESEARCH_DISCLAIMER: str = (
    "RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DIAGNOSIS. "
    "Automated predictions from MRITumorDetection-CNN require confirmation "
    "by a board-certified neuroradiologist."
)

EMPTY_STATUS_MARKDOWN: str = (
    "### Awaiting MRI Slice\n\n"
    "No scan has been analyzed yet. Upload a 2D axial MRI slice (.jpg/.png) "
    "or a Figshare .mat file on the left and select **Run Screening**."
)

CUSTOM_CSS: str = """
body,
.gradio-container {
    background-color: #0B1120 !important;
    color: #F1F5F9 !important;
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 16px 20px 28px !important;
    overflow-x: clip;
}

.gradio-container code,
.gradio-container pre {
    background-color: #1E293B !important;
    color: #E2E8F0 !important;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 1px 5px;
}

.clinical-masthead {
    border-bottom: 1px solid #334155;
    padding-bottom: 12px;
    margin-bottom: 12px;
}

.clinical-masthead h1 {
    color: #F1F5F9;
    font-size: 1.625rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 4px 0;
}

.clinical-masthead p {
    color: #CBD5E1;
    font-size: 0.95rem;
    line-height: 1.45;
    margin: 0;
}

.clinical-disclaimer {
    background-color: #271C0B;
    color: #FDE68A;
    border: 1px solid #F59E0B;
    border-left: 5px solid #F59E0B;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.925rem;
    font-weight: 600;
    line-height: 1.45;
    margin-bottom: 12px;
}

.clinical-status-ready {
    background-color: #0F2928;
    color: #CCFBF1;
    border: 1px solid #0F766E;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 16px;
    font-variant-numeric: tabular-nums;
}

.clinical-status-missing {
    background-color: #2A1215;
    color: #FECACA;
    border: 1px solid #F87171;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 16px;
}

.clinical-status-missing pre {
    background-color: #1E293B !important;
    color: #E2E8F0 !important;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 8px 10px;
    margin: 8px 0 0 0;
    font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace;
    font-size: 0.84rem;
    overflow-x: auto;
    white-space: pre;
}

.clinical-section-heading h2 {
    color: #F1F5F9;
    font-size: 1.125rem;
    font-weight: 600;
    margin: 0 0 6px 0;
}

.clinical-readout {
    background-color: #111827;
    color: #F1F5F9;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
    font-variant-numeric: tabular-nums;
}

.clinical-probs,
.clinical-json {
    font-variant-numeric: tabular-nums;
}

.clinical-action-btn button {
    min-height: 44px !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    cursor: pointer !important;
}

button:not(:disabled),
summary,
select:not(:disabled),
input[type="checkbox"]:not(:disabled),
label:has(input[type="checkbox"]:not(:disabled)) {
    cursor: pointer;
}

button:disabled,
input:disabled,
select:disabled {
    cursor: not-allowed;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible,
[tabindex]:focus-visible {
    outline: 2px solid #2DD4BF !important;
    outline-offset: 2px !important;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 12px 12px 24px !important;
    }

    .clinical-masthead h1 {
        font-size: 1.35rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        transition: none !important;
        animation: none !important;
    }
}
"""


def build_clinical_theme() -> gr.themes.Soft:
    """Builds a high-contrast dark clinical theme using Gradio Soft tokens."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
    ).set(
        body_background_fill="#0B1120",
        body_background_fill_dark="#0B1120",
        body_text_color="#F1F5F9",
        body_text_color_dark="#F1F5F9",
        body_text_color_subdued="#CBD5E1",
        body_text_color_subdued_dark="#CBD5E1",
        background_fill_primary="#111827",
        background_fill_primary_dark="#111827",
        background_fill_secondary="#1E293B",
        background_fill_secondary_dark="#1E293B",
        block_background_fill="#111827",
        block_background_fill_dark="#111827",
        block_border_color="#334155",
        block_border_color_dark="#334155",
        block_border_width="1px",
        block_border_width_dark="1px",
        block_label_background_fill="#1E293B",
        block_label_background_fill_dark="#1E293B",
        block_label_text_color="#F1F5F9",
        block_label_text_color_dark="#F1F5F9",
        block_title_text_color="#F1F5F9",
        block_title_text_color_dark="#F1F5F9",
        block_info_text_color="#CBD5E1",
        block_info_text_color_dark="#CBD5E1",
        block_shadow="none",
        block_shadow_dark="none",
        input_background_fill="#1E293B",
        input_background_fill_dark="#1E293B",
        input_border_color="#334155",
        input_border_color_dark="#334155",
        input_border_color_focus="#2DD4BF",
        input_border_color_focus_dark="#2DD4BF",
        button_primary_background_fill="#0F766E",
        button_primary_background_fill_dark="#0F766E",
        button_primary_background_fill_hover="#115E59",
        button_primary_background_fill_hover_dark="#115E59",
        button_primary_border_color="#0F766E",
        button_primary_border_color_dark="#0F766E",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill="#1E293B",
        button_secondary_background_fill_dark="#1E293B",
        button_secondary_background_fill_hover="#334155",
        button_secondary_background_fill_hover_dark="#334155",
        button_secondary_border_color="#334155",
        button_secondary_border_color_dark="#334155",
        button_secondary_text_color="#F1F5F9",
        button_secondary_text_color_dark="#F1F5F9",
        color_accent="#0F766E",
        slider_color="#0F766E",
        slider_color_dark="#0F766E",
        checkbox_background_color_selected="#0F766E",
        checkbox_background_color_selected_dark="#0F766E",
        checkbox_border_color="#334155",
        checkbox_border_color_dark="#334155",
    )


CLINICAL_THEME: gr.themes.Soft = build_clinical_theme()

_DEFAULT_MANAGERS_LOCK = threading.Lock()
_DEFAULT_MANAGERS: dict[str, "PredictorManager"] = {}
_PREDICTOR_LOCKS_GUARD = threading.Lock()
_PREDICTOR_LOCKS: weakref.WeakKeyDictionary[CascadePredictor, threading.Lock] = (
    weakref.WeakKeyDictionary()
)


def _get_predictor_lock(predictor: CascadePredictor) -> threading.Lock:
    """Returns a per-predictor lock guarding threshold overrides and Grad-CAM hooks."""
    with _PREDICTOR_LOCKS_GUARD:
        lock = _PREDICTOR_LOCKS.get(predictor)
        if lock is None:
            lock = threading.Lock()
            _PREDICTOR_LOCKS[predictor] = lock
        return lock


def _resolve_device_label(device: torch.device) -> str:
    """Formats the active PyTorch compute device cleanly."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            idx = 0 if device.index is None else int(device.index)
            gpu_name = torch.cuda.get_device_name(idx)
            return f"{gpu_name} ({device})"
        except Exception:
            return str(device)
    return str(device)


def _read_default_thresholds(cascade_config_path: Path) -> tuple[float, float, float]:
    """Reads fallback thresholds from ``cascade.yaml`` when checkpoints are absent."""
    default_s1 = 0.35
    default_s2 = 0.70
    default_temp = 1.0
    if not cascade_config_path.is_file():
        return default_s1, default_s2, default_temp
    try:
        cfg = load_yaml_config(cascade_config_path)
        s1_raw = cfg.get("stage1_threshold")
        s2_raw = cfg.get("stage2_uncertainty_threshold")
        temp_raw = cfg.get("stage2_temperature")
        return (
            float(s1_raw) if s1_raw is not None else default_s1,
            float(s2_raw) if s2_raw is not None else default_s2,
            float(temp_raw) if temp_raw is not None else default_temp,
        )
    except Exception:
        return default_s1, default_s2, default_temp


class PredictorManager:
    """Lazy/cached loader around ``CascadePredictor.from_configs`` with readiness status."""

    def __init__(
        self,
        cascade_config_path: str | Path = "configs/cascade.yaml",
        predictor: CascadePredictor | None = None,
    ) -> None:
        self.cascade_config_path = Path(cascade_config_path)
        self._predictor: CascadePredictor | None = predictor
        self._last_error: str | None = None
        self._lock = threading.RLock()

    @property
    def predictor(self) -> CascadePredictor | None:
        """Returns the cached predictor without triggering disk load."""
        with self._lock:
            return self._predictor

    @property
    def last_error(self) -> str | None:
        """Returns the most recent checkpoint/config load error, if any."""
        with self._lock:
            return self._last_error

    def get_predictor(self, force_reload: bool = False) -> CascadePredictor:
        """Loads and caches ``CascadePredictor`` from disk when needed."""
        with self._lock:
            if self._predictor is not None and not force_reload:
                return self._predictor
            try:
                loaded = CascadePredictor.from_configs(
                    self.cascade_config_path,
                    load_weights=True,
                    allow_missing_checkpoints=False,
                )
            except Exception as exc:
                self._last_error = str(exc)
                raise
            self._predictor = loaded
            self._last_error = None
            return loaded

    def inspect_status(self, attempt_load: bool = True) -> dict[str, Any]:
        """Returns model readiness metadata without raising on missing or invalid checkpoints."""
        with self._lock:
            if self._predictor is None and attempt_load:
                try:
                    self.get_predictor()
                except Exception:
                    pass

            if self._predictor is not None:
                pred = self._predictor
                return {
                    "ready": True,
                    "device_label": _resolve_device_label(pred.device),
                    "stage1_threshold": float(pred.stage1_threshold),
                    "stage2_uncertainty_threshold": float(pred.stage2_uncertainty_threshold),
                    "stage2_temperature": float(pred.stage2_temperature),
                    "error": None,
                    "config_path": str(self.cascade_config_path),
                }

            last_err = self._last_error

        s1_thr, s2_thr, s2_temp = _read_default_thresholds(self.cascade_config_path)
        return {
            "ready": False,
            "device_label": _resolve_device_label(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ),
            "stage1_threshold": s1_thr,
            "stage2_uncertainty_threshold": s2_thr,
            "stage2_temperature": s2_temp,
            "error": last_err or "Model checkpoints have not been loaded.",
            "config_path": str(self.cascade_config_path),
        }


def _get_default_manager(cascade_config_path: str | Path) -> PredictorManager:
    """Returns a cached ``PredictorManager`` keyed by resolved cascade config path."""
    key = str(Path(cascade_config_path).resolve())
    with _DEFAULT_MANAGERS_LOCK:
        manager = _DEFAULT_MANAGERS.get(key)
        if manager is None:
            manager = PredictorManager(cascade_config_path=cascade_config_path)
            _DEFAULT_MANAGERS[key] = manager
        return manager


def _missing_checkpoint_instructions(error_text: str, cascade_config_path: Path) -> str:
    """Formats actionable CLI training guidance when checkpoints are missing."""
    return (
        f"Trained cascade checkpoints are not ready ({error_text}). "
        f"Run `mri-tumor prepare --config {cascade_config_path}`, "
        "`mri-tumor train --config configs/stage1.yaml`, and "
        "`mri-tumor train --config configs/stage2.yaml` first."
    )


def _validate_slider_probability(value: float, label: str) -> float:
    """Validates slider threshold inputs and surfaces user-friendly ``gr.Error``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise gr.Error(f"{label} must be a numeric probability in [0.0, 1.0].")
    val = float(value)
    if not (math.isfinite(val) and 0.0 <= val <= 1.0):
        raise gr.Error(f"{label} must be finite and between 0.0 and 1.0 (got {value!r}).")
    return val


def _resolve_mat_path(mat_file: Any) -> Path | None:
    """Extracts a filesystem path from Gradio ``gr.File`` payloads or direct path inputs."""
    if mat_file is None:
        return None
    if isinstance(mat_file, (str, Path)):
        raw = str(mat_file).strip()
        return Path(raw) if raw else None
    if isinstance(mat_file, (list, tuple)):
        if not mat_file:
            return None
        return _resolve_mat_path(mat_file[0])
    if isinstance(mat_file, dict):
        path_val = mat_file.get("path") or mat_file.get("name")
        raw = str(path_val).strip() if path_val is not None else ""
        return Path(raw) if raw else None
    saw_attr = False
    for attr in ("path", "name"):
        if hasattr(mat_file, attr):
            saw_attr = True
            attr_val = getattr(mat_file, attr)
            if isinstance(attr_val, (str, Path)):
                raw = str(attr_val).strip()
                if raw:
                    return Path(raw)
    if saw_attr:
        return None
    raise gr.Error(f"Unsupported .mat file upload payload: {type(mat_file).__name__}.")


def _format_decision_title(result: CascadeResult) -> str:
    """Formats the primary cascade decision heading."""
    if not result.stage1_positive:
        return "No Tumor"
    if result.abstained:
        return ABSTAIN_LABEL
    return result.predicted_label.replace("_", " ").title()


def outline_tumor_region(
    original_image: np.ndarray,
    heatmap: np.ndarray | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
    tumor_mask: np.ndarray | None = None,
    activation_threshold: float = 0.55,
) -> np.ndarray:
    """Draws a crisp tumor contour + subtle 15% tint on the original MRI slice.

    Uses the ground-truth ``tumor_mask`` (emerald) when available, otherwise maps
    the Grad-CAM ``heatmap`` back into ``crop_box`` and outlines the high-activation
    region (coral-red). Returns clean ``uint8`` RGB when neither is present.
    """
    rgb = _to_rgb_uint8(original_image)
    h_orig, w_orig = rgb.shape[:2]

    if tumor_mask is not None and np.any(tumor_mask > 0):
        mask_arr = (np.asarray(tumor_mask) > 0).astype(np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        if mask_arr.shape[:2] != (h_orig, w_orig):
            mask_arr = cv2.resize(mask_arr, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        fg = mask_arr > 0
        if np.any(fg):
            emerald = np.array([34, 197, 94], dtype=np.float32)
            out = rgb.astype(np.float32)
            out[fg] = 0.85 * out[fg] + 0.15 * emerald
            out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out_u8, contours, -1, (34, 197, 94), thickness=2, lineType=cv2.LINE_AA)
            return out_u8

    if heatmap is not None and np.max(heatmap) > 0:
        hm = np.asarray(heatmap, dtype=np.float32)
        if crop_box is not None:
            x1, y1, x2, y2 = crop_box
            x1 = max(0, min(w_orig, int(x1)))
            y1 = max(0, min(h_orig, int(y1)))
            x2 = max(0, min(w_orig, int(x2)))
            y2 = max(0, min(h_orig, int(y2)))
            bw = x2 - x1
            bh = y2 - y1
            if bw > 0 and bh > 0:
                full_hm = np.zeros((h_orig, w_orig), dtype=np.float32)
                full_hm[y1:y2, x1:x2] = cv2.resize(hm, (bw, bh), interpolation=cv2.INTER_LINEAR)
            else:
                full_hm = cv2.resize(hm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            full_hm = cv2.resize(hm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        region = full_hm >= float(activation_threshold)
        if not np.any(region):
            pos_vals = full_hm[full_hm > 0]
            if pos_vals.size > 0:
                region = full_hm >= float(np.quantile(pos_vals, 0.85))

        if np.any(region):
            coral_red = np.array([239, 68, 68], dtype=np.float32)
            out = rgb.astype(np.float32)
            out[region] = 0.85 * out[region] + 0.15 * coral_red
            out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out_u8, contours, -1, (239, 68, 68), thickness=2, lineType=cv2.LINE_AA)
            return out_u8

    return rgb


def _format_status_markdown(
    result: CascadeResult,
    *,
    source_summary: str,
    outline_source: str,
) -> str:
    """Builds the structured clinical Markdown readout for a single cascade prediction."""
    decision_title = _format_decision_title(result)
    s1_prob = float(result.stage1_tumor_prob)
    s1_thr = float(result.stage1_threshold)

    if not result.stage1_positive:
        routing_summary = (
            f"Negative screening: P(tumor) = **{s1_prob:.4f} ({s1_prob * 100:.1f}%)** "
            f"is below calibrated cutoff **{s1_thr:.2f}**."
        )
        stage2_block = (
            f"- **Stage 2 Subtyping:** Not executed (cleared at Stage 1; "
            f"abstention threshold tau = **{result.stage2_uncertainty_threshold:.2f}**, "
            f"temperature T = **{result.stage2_temperature:.3f}**)"
        )
    else:
        routing_summary = (
            f"Positive screening: P(tumor) = **{s1_prob:.4f} ({s1_prob * 100:.1f}%)** "
            f"meets or exceeds cutoff **{s1_thr:.2f}**; slice routed to Stage 2."
        )
        conf = float(result.stage2_confidence or 0.0)
        tau = float(result.stage2_uncertainty_threshold)
        temp = float(result.stage2_temperature)
        probs_text = ", ".join(
            f"{name.replace('_', ' ').title()}: **{prob:.4f} ({prob * 100:.1f}%)**"
            for name, prob in (result.stage2_probs or {}).items()
        )
        if result.abstained:
            subtype_note = (
                f"**Specialist Referral Triggered:** Calibrated confidence **{conf:.4f} "
                f"({conf * 100:.1f}%)** is below abstention threshold tau = **{tau:.2f}** "
                f"(top raw candidate: **{result.raw_4way_label.replace('_', ' ').title()}**)."
            )
        else:
            subtype_note = (
                f"Confident subtype assignment at **{conf:.4f} ({conf * 100:.1f}%)** "
                f"above abstention threshold tau = **{tau:.2f}**."
            )
        stage2_block = (
            f"- **Stage 2 Subtyping:** {subtype_note} Temperature T = **{temp:.3f}**.\n"
            f"- **Stage 2 Probabilities:** {probs_text}"
        )

    return "\n".join([
        f"### Final Cascade Decision: {decision_title}",
        "",
        routing_summary,
        "",
        f"- **Input Source:** {source_summary}",
        (
            f"- **Stage 1 Screening:** P(tumor) = **{s1_prob:.4f} ({s1_prob * 100:.1f}%)** "
            f"vs threshold **{s1_thr:.2f}** "
            f"({'Positive -> Stage 2' if result.stage1_positive else 'Negative -> Cleared'})"
        ),
        stage2_block,
        f"- **Tumor Outline:** {outline_source}",
    ])


def _build_label_probabilities(result: CascadeResult) -> dict[str, float]:
    """Formats Stage 1 or Stage 2 calibrated probabilities for ``gr.Label``."""
    if not result.stage1_positive or result.stage2_probs is None:
        tumor_prob = float(np.clip(result.stage1_tumor_prob, 0.0, 1.0))
        return {
            "No Tumor": float(1.0 - tumor_prob),
            "Tumor": tumor_prob,
        }
    return {str(name): float(prob) for name, prob in result.stage2_probs.items()}


def _render_disclaimer_html() -> str:
    """Renders the non-dismissible clinical research disclaimer banner."""
    return (
        f'<div class="clinical-disclaimer" role="alert" '
        f'aria-label="Clinical research disclaimer">'
        f"{html.escape(RESEARCH_DISCLAIMER)}"
        f"</div>"
    )


def _render_status_banner_html(status_info: dict[str, Any]) -> str:
    """Renders checkpoint readiness, device, and calibrated defaults or setup instructions."""
    if status_info["ready"]:
        device_label = html.escape(str(status_info["device_label"]))
        s1_thr = float(status_info["stage1_threshold"])
        s2_thr = float(status_info["stage2_uncertainty_threshold"])
        s2_temp = float(status_info["stage2_temperature"])
        return (
            '<div class="clinical-status-ready" role="status" aria-live="polite">'
            f"<strong>Model Status: Ready</strong> | "
            f"<strong>Device:</strong> <strong>{device_label}</strong> | "
            f"<strong>Calibrated Defaults:</strong> Stage 1 threshold = <strong>{s1_thr:.3f}</strong>, "
            f"Stage 2 temperature T = <strong>{s2_temp:.3f}</strong>, "
            f"Stage 2 abstention threshold = <strong>{s2_thr:.3f}</strong>"
            "</div>"
        )

    err_msg = html.escape(str(status_info.get("error") or "Checkpoints missing."))
    cfg_str = html.escape(str(status_info.get("config_path") or "configs/cascade.yaml"))
    cmds = html.escape(
        f"mri-tumor prepare --config {cfg_str}\n"
        "mri-tumor train --config configs/stage1.yaml\n"
        "mri-tumor train --config configs/stage2.yaml"
    )
    return (
        '<div class="clinical-status-missing" role="status" aria-live="polite">'
        f"<strong>Model Status: Checkpoints Unavailable</strong> ({err_msg}). "
        "Train both cascade stages before running inference:"
        f"<pre>{cmds}</pre>"
        "</div>"
    )


def analyze_scan(
    image_np: np.ndarray | None,
    mat_file: Any | None,
    stage1_threshold: float,
    stage2_uncertainty_threshold: float,
    generate_gradcam: bool = True,
    gradcam_on_negative: bool = False,
    predictor: CascadePredictor | None = None,
    cascade_config_path: str | Path = "configs/cascade.yaml",
) -> tuple[np.ndarray, np.ndarray, str, dict[str, float], dict[str, Any]]:
    """Runs two-stage cascade inference on an uploaded 2D slice or Figshare ``.mat`` file."""
    mat_path = _resolve_mat_path(mat_file)
    raw_mask: np.ndarray | None = None
    if mat_path is not None:
        if not mat_path.is_file():
            raise gr.Error(f"Uploaded .mat file not found: {mat_path}")
        try:
            sample = read_figshare_mat(mat_path)
        except Exception as exc:
            raise gr.Error(
                f"Failed to read Figshare .mat file ({mat_path.name}): {exc}"
            ) from exc
        raw_image = sample.image
        raw_mask = sample.tumor_mask
        source_summary = (
            f"Figshare .mat slice **{mat_path.name}** "
            f"(PID: **{sample.patient_id}**, original shape: **{sample.height}x{sample.width}**)"
        )
    elif image_np is not None:
        arr = np.asarray(image_np)
        if arr.size == 0:
            raise gr.Error("Upload a 2D MRI slice (.jpg/.png) or Figshare .mat file first.")
        raw_image = arr
        source_summary = f"Uploaded 2D image slice (shape: **{'x'.join(str(d) for d in arr.shape[:2])}**)"
    else:
        raise gr.Error("Upload a 2D MRI slice (.jpg/.png) or Figshare .mat file first.")

    s1_thr = _validate_slider_probability(stage1_threshold, "Stage 1 screening threshold")
    s2_thr = _validate_slider_probability(
        stage2_uncertainty_threshold, "Stage 2 uncertainty threshold"
    )

    cfg_path = Path(cascade_config_path)
    active_predictor = predictor
    if active_predictor is None:
        try:
            active_predictor = _get_default_manager(cfg_path).get_predictor()
        except FileNotFoundError as exc:
            raise gr.Error(_missing_checkpoint_instructions(str(exc), cfg_path)) from exc
        except Exception as exc:
            raise gr.Error(f"Failed to load cascade predictor: {exc}") from exc

    has_gt_mask = raw_mask is not None and bool(np.any(raw_mask > 0))

    lock = _get_predictor_lock(active_predictor)
    with lock:
        orig_s1_thr = active_predictor.stage1_threshold
        orig_s2_thr = active_predictor.stage2_uncertainty_threshold
        active_predictor.stage1_threshold = s1_thr
        active_predictor.stage2_uncertainty_threshold = s2_thr
        try:
            result = active_predictor.predict_image(
                raw_image,
                return_gradcam=bool(generate_gradcam),
                gradcam_on_negative=bool(gradcam_on_negative),
            )
        except ValueError as exc:
            raise gr.Error(f"Failed to analyze MRI slice: {exc}") from exc
        finally:
            active_predictor.stage1_threshold = orig_s1_thr
            active_predictor.stage2_uncertainty_threshold = orig_s2_thr

    if result.stage1_positive:
        _, _, crop_box = crop_brain_contour(
            raw_image,
            margin_ratio=active_predictor.crop_margin_ratio,
        )
        outlined_img = outline_tumor_region(
            raw_image,
            heatmap=result.gradcam_heatmap,
            crop_box=crop_box,
            tumor_mask=raw_mask,
        )
        if has_gt_mask:
            outline_source = "Ground-Truth Mask (.mat)"
        elif result.gradcam_heatmap is not None and float(np.max(result.gradcam_heatmap)) > 0:
            outline_source = "Model Focus (Grad-CAM Contour)"
        else:
            outline_source = "Disabled"
    else:
        outlined_img = _to_rgb_uint8(raw_image)
        outline_source = "None (No Tumor)"

    if result.gradcam_overlay is not None:
        hotspot_img = result.gradcam_overlay
    elif result.stage1_positive and generate_gradcam and result.preprocessed_image is not None:
        hotspot_img = result.preprocessed_image
    else:
        hotspot_img = _to_rgb_uint8(raw_image)

    status_markdown = _format_status_markdown(
        result,
        source_summary=source_summary,
        outline_source=outline_source,
    )
    stage_probs_dict = _build_label_probabilities(result)
    result_json = result.to_dict(include_arrays=False)

    return (
        outlined_img,
        hotspot_img,
        status_markdown,
        stage_probs_dict,
        result_json,
    )


def build_gradio_app(
    cascade_config_path: str | Path = "configs/cascade.yaml",
    predictor: CascadePredictor | None = None,
) -> gr.Blocks:
    """Constructs the Gradio 6.31.0 ``Blocks`` application without deprecated kwargs."""
    cfg_path = Path(cascade_config_path)
    manager = PredictorManager(cascade_config_path=cfg_path, predictor=predictor)
    status_info = manager.inspect_status(attempt_load=True)
    default_s1 = round(float(status_info["stage1_threshold"]), 2)
    default_s2 = round(float(status_info["stage2_uncertainty_threshold"]), 2)

    with gr.Blocks(title="MRITumorDetection-CNN | Clinical Research Cascade") as demo:
        gr.HTML(
            '<header class="clinical-masthead">'
            "<h1>MRITumorDetection-CNN</h1>"
            "<p>Two-Stage Brain MRI Screening &amp; Subtyping Cascade "
            "(Stage 1 Binary Screening, Stage 2 Temperature-Scaled Subtyping, Grad-CAM Attribution)</p>"
            "</header>"
        )
        gr.HTML(_render_disclaimer_html())
        gr.HTML(_render_status_banner_html(status_info))

        with gr.Row():
            with gr.Column(scale=6, min_width=320):
                image_input = gr.Image(
                    type="numpy",
                    label="Original MRI Slice",
                    sources=["upload", "clipboard"],
                    height=320,
                )
                mat_input = gr.File(
                    file_types=[".mat"],
                    type="filepath",
                    label="Or Upload .mat Slice",
                )
                with gr.Row():
                    run_btn = gr.Button(
                        "Run Screening",
                        variant="primary",
                        elem_classes=["clinical-action-btn"],
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_classes=["clinical-action-btn"],
                    )
                with gr.Accordion("Advanced Settings", open=False):
                    stage1_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=default_s1,
                        label="Stage 1 Screening Threshold",
                        info="Slices with P(tumor) >= threshold advance to Stage 2 subtyping.",
                    )
                    stage2_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=default_s2,
                        label="Stage 2 Uncertainty Threshold (tau)",
                        info="Positive slices below calibrated confidence tau trigger specialist referral.",
                    )
                    gradcam_checkbox = gr.Checkbox(
                        value=True,
                        label="Generate Grad-CAM",
                        info="Outline model focus region on positive slices when no .mat mask is present.",
                    )

            with gr.Column(scale=6, min_width=320):
                with gr.Row():
                    outlined_out = gr.Image(
                        type="numpy",
                        label="Model Focus Outline",
                        interactive=False,
                        height=320,
                    )
                    hotspot_out = gr.Image(
                        type="numpy",
                        label="Grad-CAM Hotspot",
                        interactive=False,
                        height=320,
                    )
                status_out = gr.Markdown(
                    value=EMPTY_STATUS_MARKDOWN,
                    elem_classes=["clinical-readout"],
                )
                probs_out = gr.Label(
                    label="Class Probabilities",
                    num_top_classes=3,
                    elem_classes=["clinical-probs"],
                )
                with gr.Accordion("Raw JSON Details", open=False):
                    json_out = gr.JSON(
                        label="Structured Audit Payload",
                        elem_classes=["clinical-json"],
                    )

        def _on_run(
            image_np: np.ndarray | None,
            mat_file: Any | None,
            s1_thr: float,
            s2_thr: float,
            gen_cam: bool,
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            str,
            dict[str, float],
            dict[str, Any],
        ]:
            try:
                active_predictor = manager.get_predictor()
            except FileNotFoundError as exc:
                raise gr.Error(_missing_checkpoint_instructions(str(exc), cfg_path)) from exc
            except Exception as exc:
                raise gr.Error(f"Failed to load cascade predictor: {exc}") from exc

            return analyze_scan(
                image_np=image_np,
                mat_file=mat_file,
                stage1_threshold=s1_thr,
                stage2_uncertainty_threshold=s2_thr,
                generate_gradcam=gen_cam,
                predictor=active_predictor,
                cascade_config_path=cfg_path,
            )

        def _on_clear() -> tuple[Any, ...]:
            refreshed = manager.inspect_status(attempt_load=False)
            return (
                None,
                None,
                round(float(refreshed["stage1_threshold"]), 2),
                round(float(refreshed["stage2_uncertainty_threshold"]), 2),
                True,
                None,
                None,
                EMPTY_STATUS_MARKDOWN,
                {},
                None,
            )

        run_btn.click(
            fn=_on_run,
            inputs=[
                image_input,
                mat_input,
                stage1_slider,
                stage2_slider,
                gradcam_checkbox,
            ],
            outputs=[
                outlined_out,
                hotspot_out,
                status_out,
                probs_out,
                json_out,
            ],
            api_name="analyze_scan",
            show_progress="full",
        )
        clear_btn.click(
            fn=_on_clear,
            inputs=None,
            outputs=[
                image_input,
                mat_input,
                stage1_slider,
                stage2_slider,
                gradcam_checkbox,
                outlined_out,
                hotspot_out,
                status_out,
                probs_out,
                json_out,
            ],
            queue=False,
        )

    return demo


def launch_ui(
    cascade_config_path: str | Path = "configs/cascade.yaml",
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> tuple[Any, str, str]:
    """Builds and launches the Gradio 6.31.0 web interface."""
    demo = build_gradio_app(cascade_config_path=cascade_config_path)
    return demo.launch(
        server_name=host,
        server_port=int(port),
        share=bool(share),
        inbrowser=bool(inbrowser),
        theme=CLINICAL_THEME,
        css=CUSTOM_CSS,
    )


__all__ = [
    "CLINICAL_THEME",
    "CUSTOM_CSS",
    "EMPTY_STATUS_MARKDOWN",
    "PredictorManager",
    "RESEARCH_DISCLAIMER",
    "analyze_scan",
    "build_clinical_theme",
    "build_gradio_app",
    "launch_ui",
    "outline_tumor_region",
]
