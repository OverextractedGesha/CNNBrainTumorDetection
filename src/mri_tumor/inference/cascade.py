"""Two-stage clinical cascade inference engine and configuration loader."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from mri_tumor.data.figshare import read_figshare_mat
from mri_tumor.data.preprocess import load_rgb_image, preprocess_slice
from mri_tumor.evaluation.metrics import DEFAULT_STAGE2_CLASSES
from mri_tumor.explain.gradcam import (
    GradCAM,
    _get_model_device,
    _get_model_dtype,
    _to_rgb_uint8,
    overlay_gradcam,
)
from mri_tumor.models.factory import build_model_from_config
from mri_tumor.training.calibrate import ABSTAIN_LABEL


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file and verifies it parses as a mapping."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {cfg_path}, got {type(data).__name__}.")
    return data


def resolve_config_path(
    raw_path: str | Path,
    config_path: str | Path,
    *,
    must_exist: bool = False,
    allow_cwd_fallback: bool = True,
) -> Path:
    """Resolves a config-referenced path against project root, config dir, or CWD.

    When ``config_path`` lives inside a ``configs``/``config`` directory, its parent's
    parent (``config_path.resolve().parent.parent``) is treated as the project root.
    """
    candidate = Path(raw_path)
    if candidate.is_absolute():
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Path not found: {candidate}")
        return candidate

    cfg_resolved = Path(config_path).resolve()
    cfg_dir = cfg_resolved.parent
    proj_root = cfg_dir.parent
    base_dir = proj_root if cfg_dir.name.lower() in {"configs", "config"} else cfg_dir

    search_roots: list[Path] = [base_dir, cfg_dir, proj_root]
    if allow_cwd_fallback:
        search_roots.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for root in search_roots:
        resolved_cand = (root / candidate).resolve()
        if resolved_cand in seen:
            continue
        seen.add(resolved_cand)
        if resolved_cand.exists():
            return root / candidate

    if must_exist:
        raise FileNotFoundError(
            f"Path not found: {raw_path!r} (base_dir={base_dir}, cfg_dir={cfg_dir})"
        )
    return base_dir / candidate


def _validate_probability_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a numeric value in [0, 1], got {value!r}.")
    val = float(value)
    if not (math.isfinite(val) and 0.0 <= val <= 1.0):
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}.")
    return val


def _validate_positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    val = float(value)
    if not (math.isfinite(val) and val > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return val


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def allow_cwd_fallback_for_config(config_path: str | Path) -> bool:
    """Prevents isolated projects with their own configs dir from falling back to CWD."""
    cfg_resolved = Path(config_path).resolve()
    cfg_dir = cfg_resolved.parent
    proj_root = cfg_dir.parent if cfg_dir.name.lower() in {"configs", "config"} else cfg_dir
    has_local_configs = (proj_root / "configs").is_dir() or (proj_root / "config").is_dir()
    return not (has_local_configs and proj_root != Path.cwd().resolve())


def _first_not_none(*values: Any, default: Any) -> Any:
    """Returns the first value that is not None, or ``default``."""
    for val in values:
        if val is not None:
            return val
    return default


def _resolve_runtime_device_dtype(
    model: nn.Module, fallback_device: torch.device
) -> tuple[torch.device, torch.dtype]:
    """Resolves active model device (preserving fallback for stateless mocks) and float dtype."""
    has_state = any(True for _ in model.parameters()) or any(True for _ in model.buffers())
    device = _get_model_device(model) if has_state else fallback_device
    return device, _get_model_dtype(model)


@dataclass
class CascadeResult:
    """Structured output for a single-slice Stage 1 -> Stage 2 cascade prediction."""

    stage1_tumor_prob: float
    stage1_threshold: float
    stage1_positive: bool
    predicted_label: str
    raw_4way_label: str
    abstained: bool
    stage2_probs: dict[str, float] | None
    stage2_confidence: float | None
    stage2_temperature: float
    stage2_uncertainty_threshold: float
    preprocessed_image: np.ndarray | None = None
    gradcam_heatmap: np.ndarray | None = None
    gradcam_overlay: np.ndarray | None = None
    gradcam_stage: str | None = None

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Serializes cascade prediction metadata (and optionally image/heatmap arrays)."""
        payload: dict[str, Any] = {
            "stage1_tumor_prob": float(self.stage1_tumor_prob),
            "stage1_threshold": float(self.stage1_threshold),
            "stage1_positive": bool(self.stage1_positive),
            "predicted_label": str(self.predicted_label),
            "raw_4way_label": str(self.raw_4way_label),
            "abstained": bool(self.abstained),
            "stage2_probs": (
                {str(k): float(v) for k, v in self.stage2_probs.items()}
                if self.stage2_probs is not None
                else None
            ),
            "stage2_confidence": (
                float(self.stage2_confidence)
                if self.stage2_confidence is not None
                else None
            ),
            "stage2_temperature": float(self.stage2_temperature),
            "stage2_uncertainty_threshold": float(self.stage2_uncertainty_threshold),
            "gradcam_stage": self.gradcam_stage,
        }
        if include_arrays:
            payload["preprocessed_image"] = self.preprocessed_image
            payload["gradcam_heatmap"] = self.gradcam_heatmap
            payload["gradcam_overlay"] = self.gradcam_overlay
        return payload


class CascadePredictor:
    """End-to-end two-stage brain MRI tumor screening and subtyping predictor."""

    def __init__(
        self,
        stage1_model: nn.Module,
        stage2_model: nn.Module | None = None,
        stage1_threshold: float = 0.35,
        stage2_temperature: float = 1.0,
        stage2_uncertainty_threshold: float = 0.70,
        image_size: int = 224,
        use_clahe: bool = False,
        crop_margin_ratio: float = 0.05,
        stage2_classes: Sequence[str] = DEFAULT_STAGE2_CLASSES,
        device: str | torch.device | None = None,
        stage2_models: Sequence[nn.Module] | None = None,
        stage2_temperatures: Sequence[float] | None = None,
    ) -> None:
        self.stage1_threshold = _validate_probability_threshold(
            stage1_threshold, "stage1_threshold"
        )
        self.stage2_uncertainty_threshold = _validate_probability_threshold(
            stage2_uncertainty_threshold, "stage2_uncertainty_threshold"
        )
        self.stage2_temperature = _validate_positive_finite(
            stage2_temperature, "stage2_temperature"
        )

        if isinstance(image_size, bool) or not isinstance(image_size, (int, np.integer)) or int(image_size) <= 0:
            raise ValueError(f"image_size must be a positive integer, got {image_size!r}.")
        self.image_size = int(image_size)

        if (
            isinstance(crop_margin_ratio, bool)
            or not isinstance(crop_margin_ratio, (int, float, np.integer, np.floating))
            or not (math.isfinite(float(crop_margin_ratio)) and float(crop_margin_ratio) >= 0.0)
        ):
            raise ValueError(
                f"crop_margin_ratio must be finite and non-negative, got {crop_margin_ratio!r}."
            )
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.use_clahe = bool(use_clahe)

        if stage2_classes is None or isinstance(stage2_classes, (str, bytes)):
            raise ValueError(f"stage2_classes must be a sequence of class names, got {stage2_classes!r}.")
        classes = tuple(str(c) for c in stage2_classes)
        if len(classes) < 2:
            raise ValueError(f"stage2_classes must contain at least 2 classes, got {classes}.")
        self.stage2_classes = classes

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.stage1_model = stage1_model.to(self.device).eval()

        if stage2_models is not None and len(stage2_models) > 0:
            self.stage2_models = [m.to(self.device).eval() for m in stage2_models]
            self.stage2_model = self.stage2_models[0]
        else:
            if stage2_model is None:
                raise ValueError(
                    "Either stage2_model or a non-empty stage2_models sequence must be provided."
                )
            self.stage2_model = stage2_model.to(self.device).eval()
            self.stage2_models = [self.stage2_model]

        if stage2_temperatures is not None:
            if len(stage2_temperatures) != len(self.stage2_models):
                raise ValueError(
                    f"stage2_temperatures length ({len(stage2_temperatures)}) must match "
                    f"stage2_models length ({len(self.stage2_models)})."
                )
            self.stage2_temperatures = [
                _validate_positive_finite(t, f"stage2_temperatures[{i}]")
                for i, t in enumerate(stage2_temperatures)
            ]
        else:
            self.stage2_temperatures = [self.stage2_temperature] * len(self.stage2_models)

    @staticmethod
    def _load_stage_model(
        ckpt_path: Path,
        default_cfg: dict[str, Any],
        target_device: torch.device,
        load_weights: bool,
    ) -> nn.Module:
        """Builds a stage model (preferring checkpoint-embedded config) and optionally loads weights."""
        if load_weights and ckpt_path.is_file():
            ckpt_obj = torch.load(ckpt_path, map_location=target_device, weights_only=False)
            build_cfg = copy.deepcopy(
                ckpt_obj["config"]
                if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("config"), dict)
                else default_cfg
            )
            build_cfg.setdefault("model", {})["pretrained"] = False
            model = build_model_from_config(build_cfg)
            state_dict = (
                ckpt_obj["model_state_dict"]
                if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj
                else ckpt_obj
            )
            model.load_state_dict(state_dict)
            return model.to(target_device)

        build_cfg = copy.deepcopy(default_cfg)
        if load_weights and ckpt_path.is_file():
            build_cfg.setdefault("model", {})["pretrained"] = False
        return build_model_from_config(build_cfg)

    @classmethod
    def from_configs(
        cls,
        cascade_config_path: str | Path,
        device: str | torch.device | None = None,
        load_weights: bool = True,
        allow_missing_checkpoints: bool = False,
    ) -> CascadePredictor:
        """Constructs a ``CascadePredictor`` from ``cascade.yaml`` and stage configs."""
        cascade_path = Path(cascade_config_path)
        cascade_cfg = load_yaml_config(cascade_path)
        allow_cwd = allow_cwd_fallback_for_config(cascade_path)

        s1_cfg_path = resolve_config_path(
            cascade_cfg.get("stage1_config") or "configs/stage1.yaml",
            cascade_path,
            must_exist=True,
            allow_cwd_fallback=allow_cwd,
        )
        s2_cfg_path = resolve_config_path(
            cascade_cfg.get("stage2_config") or "configs/stage2.yaml",
            cascade_path,
            must_exist=True,
            allow_cwd_fallback=allow_cwd,
        )
        s1_cfg = load_yaml_config(s1_cfg_path)
        s2_cfg = load_yaml_config(s2_cfg_path)

        s1_ckpt = resolve_config_path(
            cascade_cfg.get("stage1_checkpoint") or "checkpoints/stage1/best_model.pt",
            cascade_path,
            allow_cwd_fallback=allow_cwd,
        )

        raw_s2_ckpts = cascade_cfg.get("stage2_checkpoints")
        has_ensemble_ckpts = isinstance(raw_s2_ckpts, list) and len(raw_s2_ckpts) > 0
        if has_ensemble_ckpts:
            s2_ckpts = [
                resolve_config_path(p, cascade_path, allow_cwd_fallback=allow_cwd)
                for p in raw_s2_ckpts
            ]
        else:
            s2_ckpts = [
                resolve_config_path(
                    cascade_cfg.get("stage2_checkpoint") or "checkpoints/stage2/best_model.pt",
                    cascade_path,
                    allow_cwd_fallback=allow_cwd,
                )
            ]

        s1_exists = s1_ckpt.is_file()
        if load_weights and not allow_missing_checkpoints:
            if not s1_exists:
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1_ckpt}")
            for s2_ckpt in s2_ckpts:
                if not s2_ckpt.is_file():
                    raise FileNotFoundError(f"Stage 2 checkpoint not found: {s2_ckpt}")

        target_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )

        stage1_model = cls._load_stage_model(s1_ckpt, s1_cfg, target_device, load_weights)
        stage2_models = [
            cls._load_stage_model(s2_ckpt, s2_cfg, target_device, load_weights)
            for s2_ckpt in s2_ckpts
        ]

        stage1_threshold = float(
            _first_not_none(cascade_cfg.get("stage1_threshold"), default=0.35)
        )
        default_s2_temperature = float(
            _first_not_none(cascade_cfg.get("stage2_temperature"), default=1.0)
        )
        stage2_uncertainty_threshold = float(
            _first_not_none(cascade_cfg.get("stage2_uncertainty_threshold"), default=0.70)
        )

        if not load_weights or s1_exists:
            s1_cal = _read_json_if_exists(s1_ckpt.parent / "calibration.json")
            if s1_cal.get("threshold") is not None:
                stage1_threshold = float(s1_cal["threshold"])

        stage2_temperatures: list[float] = []
        unc_thresh_from_cal = False
        for s2_ckpt in s2_ckpts:
            temp = default_s2_temperature
            if not load_weights or s2_ckpt.is_file():
                s2_cal = _read_json_if_exists(s2_ckpt.parent / "calibration.json")
                if s2_cal.get("temperature") is not None:
                    temp = float(s2_cal["temperature"])
                if not unc_thresh_from_cal and s2_cal.get("uncertainty_threshold") is not None:
                    stage2_uncertainty_threshold = float(s2_cal["uncertainty_threshold"])
                    unc_thresh_from_cal = True
            stage2_temperatures.append(temp)

        s1_data = s1_cfg.get("data", {}) if isinstance(s1_cfg.get("data"), dict) else {}
        s2_data = s2_cfg.get("data", {}) if isinstance(s2_cfg.get("data"), dict) else {}
        image_size = int(
            _first_not_none(
                cascade_cfg.get("image_size"),
                s1_data.get("image_size"),
                s2_data.get("image_size"),
                default=224,
            )
        )
        use_clahe = bool(
            _first_not_none(
                cascade_cfg.get("use_clahe"),
                s1_data.get("use_clahe"),
                s2_data.get("use_clahe"),
                default=False,
            )
        )
        crop_margin_ratio = float(
            _first_not_none(
                cascade_cfg.get("crop_margin_ratio"),
                s1_data.get("crop_margin_ratio"),
                s2_data.get("crop_margin_ratio"),
                default=0.05,
            )
        )
        stage2_classes = _first_not_none(
            cascade_cfg.get("stage2_classes"), default=DEFAULT_STAGE2_CLASSES
        )

        return cls(
            stage1_model=stage1_model,
            stage2_model=stage2_models[0],
            stage1_threshold=stage1_threshold,
            stage2_temperature=stage2_temperatures[0],
            stage2_uncertainty_threshold=stage2_uncertainty_threshold,
            image_size=image_size,
            use_clahe=use_clahe,
            crop_margin_ratio=crop_margin_ratio,
            stage2_classes=stage2_classes,
            device=target_device,
            stage2_models=stage2_models,
            stage2_temperatures=stage2_temperatures,
        )

    def preprocess_array(self, image: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Preprocesses a raw 2D/3D MRI array into ``(tensor (1, 3, H, W), uint8_rgb)``."""
        if image is None:
            raise ValueError("Input image cannot be None.")
        arr = np.asarray(image)
        if arr.size == 0:
            raise ValueError("Input image must not be empty.")
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            raise ValueError("Input image must contain only finite values.")
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        img_3ch, _ = preprocess_slice(
            arr,
            mask=None,
            target_size=(self.image_size, self.image_size),
            crop_margin_ratio=self.crop_margin_ratio,
            use_clahe=self.use_clahe,
        )
        uint8_rgb = np.clip(np.round(img_3ch * 255.0), 0, 255).astype(np.uint8)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(img_3ch).transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
        )
        return tensor, uint8_rgb

    def _build_negative_result(
        self,
        tumor_prob: float,
        preprocessed_rgb: np.ndarray | None = None,
        heatmap: np.ndarray | None = None,
        overlay: np.ndarray | None = None,
        gradcam_stage: str | None = None,
    ) -> CascadeResult:
        return CascadeResult(
            stage1_tumor_prob=float(tumor_prob),
            stage1_threshold=self.stage1_threshold,
            stage1_positive=False,
            predicted_label="no_tumor",
            raw_4way_label="no_tumor",
            abstained=False,
            stage2_probs=None,
            stage2_confidence=None,
            stage2_temperature=self.stage2_temperature,
            stage2_uncertainty_threshold=self.stage2_uncertainty_threshold,
            preprocessed_image=preprocessed_rgb,
            gradcam_heatmap=heatmap,
            gradcam_overlay=overlay,
            gradcam_stage=gradcam_stage,
        )

    def _build_positive_result(
        self,
        tumor_prob: float,
        probs_row: np.ndarray,
        pred_idx: int,
        confidence: float,
        abstained: bool,
        predicted_label: str,
        preprocessed_rgb: np.ndarray | None = None,
        heatmap: np.ndarray | None = None,
        overlay: np.ndarray | None = None,
        gradcam_stage: str | None = None,
    ) -> CascadeResult:
        stage2_probs = {
            str(name): float(probs_row[i]) for i, name in enumerate(self.stage2_classes)
        }
        return CascadeResult(
            stage1_tumor_prob=float(tumor_prob),
            stage1_threshold=self.stage1_threshold,
            stage1_positive=True,
            predicted_label=str(predicted_label),
            raw_4way_label=str(self.stage2_classes[pred_idx]),
            abstained=bool(abstained),
            stage2_probs=stage2_probs,
            stage2_confidence=float(confidence),
            stage2_temperature=self.stage2_temperature,
            stage2_uncertainty_threshold=self.stage2_uncertainty_threshold,
            preprocessed_image=preprocessed_rgb,
            gradcam_heatmap=heatmap,
            gradcam_overlay=overlay,
            gradcam_stage=gradcam_stage,
        )

    def _compute_gradcam_overlay(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        target_class: int,
        preprocessed_rgb: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates normalized Grad-CAM heatmap and RGB overlay without leaking hooks."""
        overlay_base = (
            preprocessed_rgb
            if preprocessed_rgb is not None
            else _to_rgb_uint8(batch[0].detach().cpu().numpy().transpose(1, 2, 0))
        )
        with GradCAM(model) as cam:
            cam_res = cam.generate(batch, target_class=target_class)
        heatmap = cam_res.heatmap.astype(np.float32)
        overlay = overlay_gradcam(overlay_base, heatmap)
        return heatmap, overlay

    def _run_stage1_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Runs Stage 1 forward pass matching active model device/dtype and checks finiteness."""
        if batch.shape[2] == 0 or batch.shape[3] == 0:
            raise ValueError(f"Input batch must have non-zero spatial dimensions, got {tuple(batch.shape)}.")
        if batch.is_floating_point() and not torch.isfinite(batch).all():
            raise ValueError("Input tensor batch must contain only finite values.")

        self.stage1_model.eval()
        s1_device, s1_dtype = _resolve_runtime_device_dtype(self.stage1_model, self.device)
        s1_input = batch.to(device=s1_device, dtype=s1_dtype)
        with torch.no_grad():
            logits1 = self.stage1_model(s1_input)
        if logits1.ndim != 2 or logits1.shape[0] != batch.shape[0] or logits1.shape[1] < 2:
            raise ValueError(
                f"Expected Stage 1 logits of shape ({batch.shape[0]}, >=2), got {tuple(logits1.shape)}."
            )
        if not torch.isfinite(logits1).all():
            raise ValueError("Stage 1 logits must contain only finite values.")
        return logits1

    def predict_stage2_probs_tensor(self, images: torch.Tensor) -> np.ndarray:
        """Runs calibrated Stage 2 ensemble inference and returns mean ``(B, C)`` float32 probs."""
        if images.ndim == 3:
            batch = images.unsqueeze(0)
        elif images.ndim == 4:
            batch = images
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor batch, got shape {tuple(images.shape)}."
            )

        batch_size = int(batch.shape[0])
        num_classes = len(self.stage2_classes)
        if batch_size == 0:
            return np.empty((0, num_classes), dtype=np.float32)
        if batch.shape[2] == 0 or batch.shape[3] == 0:
            raise ValueError(
                f"Input batch must have non-zero spatial dimensions, got {tuple(batch.shape)}."
            )
        if batch.is_floating_point() and not torch.isfinite(batch).all():
            raise ValueError("Input tensor batch must contain only finite values.")

        probs_sum: torch.Tensor | None = None
        for model, temp in zip(self.stage2_models, self.stage2_temperatures):
            model.eval()
            s2_device, s2_dtype = _resolve_runtime_device_dtype(model, self.device)
            s2_input = batch.to(device=s2_device, dtype=s2_dtype)
            with torch.no_grad():
                logits = model(s2_input)
            if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] != num_classes:
                raise ValueError(
                    f"Expected Stage 2 logits of shape ({batch_size}, {num_classes}), "
                    f"got {tuple(logits.shape)}."
                )
            if not torch.isfinite(logits).all():
                raise ValueError("Stage 2 logits must contain only finite values.")
            model_probs = torch.softmax(logits.float() / float(temp), dim=-1).cpu()
            probs_sum = model_probs if probs_sum is None else (probs_sum + model_probs)

        assert probs_sum is not None
        mean_probs = probs_sum / float(len(self.stage2_models))
        return mean_probs.numpy().astype(np.float32, copy=False)

    def _apply_abstention_to_probs(self, probs: np.ndarray) -> dict[str, Any]:
        """Applies uncertainty abstention to pre-calibrated Stage 2 probability rows."""
        pred_indices = np.argmax(probs, axis=-1)
        confidences = np.max(probs, axis=-1)
        abstain_mask = confidences < float(self.stage2_uncertainty_threshold)
        labels = [
            ABSTAIN_LABEL if bool(abstain) else str(self.stage2_classes[int(idx)])
            for idx, abstain in zip(pred_indices, abstain_mask)
        ]
        return {
            "probs": probs,
            "pred_indices": pred_indices,
            "confidences": confidences,
            "abstain_mask": abstain_mask,
            "labels": labels,
        }

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        return_gradcam: bool = False,
        orig_rgb_for_overlay: np.ndarray | None = None,
        gradcam_on_negative: bool = False,
    ) -> CascadeResult:
        """Runs cascade inference (and optional Grad-CAM) on a ``(3, H, W)`` or ``(1, 3, H, W)`` tensor."""
        if tensor.ndim == 3:
            batch = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[0] == 1:
            batch = tensor
        else:
            raise ValueError(
                f"Expected tensor of shape (3, H, W) or (1, 3, H, W), got {tuple(tensor.shape)}."
            )

        preprocessed_rgb = (
            _to_rgb_uint8(orig_rgb_for_overlay)
            if orig_rgb_for_overlay is not None
            else None
        )

        logits1 = self._run_stage1_logits(batch)
        tumor_prob = float(torch.softmax(logits1.float(), dim=-1)[0, 1].item())

        if tumor_prob < self.stage1_threshold:
            heatmap: np.ndarray | None = None
            overlay: np.ndarray | None = None
            gradcam_stage: str | None = None
            if return_gradcam and gradcam_on_negative:
                heatmap, overlay = self._compute_gradcam_overlay(
                    self.stage1_model, batch, target_class=0, preprocessed_rgb=preprocessed_rgb
                )
                gradcam_stage = "stage1"
            return self._build_negative_result(
                tumor_prob=tumor_prob,
                preprocessed_rgb=preprocessed_rgb,
                heatmap=heatmap,
                overlay=overlay,
                gradcam_stage=gradcam_stage,
            )

        probs2 = self.predict_stage2_probs_tensor(batch)
        cal = self._apply_abstention_to_probs(probs2)
        pred_idx = int(cal["pred_indices"][0])

        heatmap = None
        overlay = None
        gradcam_stage = None
        if return_gradcam:
            heatmap, overlay = self._compute_gradcam_overlay(
                self.stage2_model, batch, target_class=pred_idx, preprocessed_rgb=preprocessed_rgb
            )
            gradcam_stage = "stage2"

        return self._build_positive_result(
            tumor_prob=tumor_prob,
            probs_row=cal["probs"][0],
            pred_idx=pred_idx,
            confidence=float(cal["confidences"][0]),
            abstained=bool(cal["abstain_mask"][0]),
            predicted_label=str(cal["labels"][0]),
            preprocessed_rgb=preprocessed_rgb,
            heatmap=heatmap,
            overlay=overlay,
            gradcam_stage=gradcam_stage,
        )

    def predict_image(
        self,
        image: np.ndarray,
        return_gradcam: bool = False,
        gradcam_on_negative: bool = False,
    ) -> CascadeResult:
        """Preprocesses an MRI NumPy array and returns a ``CascadeResult``."""
        tensor, uint8_rgb = self.preprocess_array(image)
        return self.predict_tensor(
            tensor,
            return_gradcam=return_gradcam,
            orig_rgb_for_overlay=uint8_rgb,
            gradcam_on_negative=gradcam_on_negative,
        )

    def predict_file(
        self,
        filepath: str | Path,
        return_gradcam: bool = False,
        gradcam_on_negative: bool = False,
    ) -> CascadeResult:
        """Loads a ``.mat`` or image file (``.jpg``/``.png``) and runs cascade prediction."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")
        image = (
            read_figshare_mat(path).image
            if path.suffix.lower() == ".mat"
            else load_rgb_image(path)
        )
        return self.predict_image(
            image,
            return_gradcam=return_gradcam,
            gradcam_on_negative=gradcam_on_negative,
        )

    def predict_batch_tensors(self, images: torch.Tensor) -> list[CascadeResult]:
        """Runs batched Stage 1 inference followed by selective batched Stage 2 inference."""
        if images.ndim == 3:
            batch = images.unsqueeze(0)
        elif images.ndim == 4:
            batch = images
        else:
            raise ValueError(
                f"Expected 3D or 4D tensor batch, got shape {tuple(images.shape)}."
            )

        batch_size = int(batch.shape[0])
        if batch_size == 0:
            return []

        logits1 = self._run_stage1_logits(batch)
        tumor_probs = torch.softmax(logits1.float(), dim=-1)[:, 1].cpu().numpy()

        pos_indices = [
            idx for idx, prob in enumerate(tumor_probs) if float(prob) >= self.stage1_threshold
        ]
        stage2_cal: dict[str, Any] | None = None
        if pos_indices:
            probs2 = self.predict_stage2_probs_tensor(batch[pos_indices])
            stage2_cal = self._apply_abstention_to_probs(probs2)

        pos_lookup = {orig_idx: pos_pos for pos_pos, orig_idx in enumerate(pos_indices)}
        results: list[CascadeResult] = []
        for idx in range(batch_size):
            tumor_prob = float(tumor_probs[idx])
            if idx not in pos_lookup:
                results.append(self._build_negative_result(tumor_prob=tumor_prob))
                continue

            assert stage2_cal is not None
            pos_i = pos_lookup[idx]
            results.append(
                self._build_positive_result(
                    tumor_prob=tumor_prob,
                    probs_row=stage2_cal["probs"][pos_i],
                    pred_idx=int(stage2_cal["pred_indices"][pos_i]),
                    confidence=float(stage2_cal["confidences"][pos_i]),
                    abstained=bool(stage2_cal["abstain_mask"][pos_i]),
                    predicted_label=str(stage2_cal["labels"][pos_i]),
                )
            )
        return results
