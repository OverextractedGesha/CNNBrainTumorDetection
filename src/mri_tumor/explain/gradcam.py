"""Grad-CAM visual attribution, overlay rendering, and mask-overlap evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Self, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mri_tumor.models.cnn import resolve_target_conv_layer


@dataclass
class GradCAMResult:
    """Single-sample Grad-CAM attribution output."""

    heatmap: np.ndarray
    target_class: int
    predicted_class: int
    confidence: float
    probs: np.ndarray


def _get_model_device(model: nn.Module) -> torch.device:
    """Resolves the device where ``model`` parameters or buffers reside."""
    for param in model.parameters():
        return param.device
    for buf in model.buffers():
        return buf.device
    return torch.device("cpu")


def _get_model_dtype(model: nn.Module) -> torch.dtype:
    """Resolves the floating-point dtype of ``model`` parameters or buffers."""
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    for buf in model.buffers():
        if buf.is_floating_point():
            return buf.dtype
    return torch.float32


def _resolve_target_classes(
    target_classes: int | Sequence[int] | None,
    pred_classes: torch.Tensor,
    num_classes: int,
) -> list[int]:
    """Validates and normalizes per-sample target class indices for Grad-CAM."""
    batch_size = int(pred_classes.shape[0])
    if target_classes is None:
        return [int(idx) for idx in pred_classes.tolist()]

    if isinstance(target_classes, (int, np.integer)) and not isinstance(target_classes, bool):
        cls_idx = int(target_classes)
        if not (0 <= cls_idx < num_classes):
            raise ValueError(f"target_class {cls_idx} out of range [0, {num_classes - 1}].")
        return [cls_idx] * batch_size

    if isinstance(target_classes, (torch.Tensor, np.ndarray)):
        if target_classes.ndim != 1:
            raise ValueError(
                f"Expected 1D target_classes, got shape {tuple(target_classes.shape)}."
            )
        raw_list = (
            target_classes.detach().cpu().tolist()
            if isinstance(target_classes, torch.Tensor)
            else target_classes.tolist()
        )
    elif isinstance(target_classes, Sequence) and not isinstance(target_classes, (str, bytes)):
        raw_list = list(target_classes)
    else:
        raise ValueError(f"Unsupported target_classes type: {type(target_classes).__name__}.")

    if len(raw_list) != batch_size:
        raise ValueError(
            f"Expected {batch_size} target classes for batch, got {len(raw_list)}."
        )

    resolved: list[int] = []
    for item in raw_list:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"Target class must be an integer, got {item!r}.")
        cls_idx = int(item)
        if not (0 <= cls_idx < num_classes):
            raise ValueError(f"target_class {cls_idx} out of range [0, {num_classes - 1}].")
        resolved.append(cls_idx)
    return resolved


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) using safe tensor hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        if target_layer is None:
            if hasattr(model, "get_target_layer") and callable(model.get_target_layer):
                target_layer = model.get_target_layer()
            else:
                target_layer = resolve_target_conv_layer(model)
        self.target_layer: nn.Module = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._grad_hook_handles: list[Any] = []
        self._forward_hook_handle: Any = self.target_layer.register_forward_hook(self._forward_hook)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach().clone()

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Expected Tensor output from target_layer, got {type(output).__name__}."
            )
        self._activations = output.detach().clone()
        if output.requires_grad:
            self._grad_hook_handles.append(output.register_hook(self._save_grad))

    def _clear_step_state(self) -> None:
        """Clears cached forward/backward tensors and per-pass tensor grad hooks."""
        for handle in self._grad_hook_handles:
            handle.remove()
        self._grad_hook_handles.clear()
        self._activations = None
        self._gradients = None

    def remove_hooks(self) -> None:
        """Removes registered forward and tensor hooks cleanly to prevent leaks."""
        self._clear_step_state()
        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.remove_hooks()

    def generate_batch(
        self,
        input_tensor: torch.Tensor,
        target_classes: int | Sequence[int] | None = None,
    ) -> list[GradCAMResult]:
        """Generates normalized ``(H, W)`` Grad-CAM heatmaps for a 4D batch ``(B, C, H, W)``."""
        if self._forward_hook_handle is None:
            raise RuntimeError("GradCAM hooks have been removed; create a new GradCAM instance.")
        if input_tensor.ndim != 4 or input_tensor.shape[0] == 0 or input_tensor.shape[2] == 0 or input_tensor.shape[3] == 0:
            raise ValueError(
                f"Expected non-empty 4D tensor (B, C, H, W), got shape {tuple(input_tensor.shape)}."
            )
        if input_tensor.is_floating_point() and not torch.isfinite(input_tensor).all():
            raise ValueError("input_tensor must contain only finite values.")

        batch_size, _, height, width = input_tensor.shape
        device = _get_model_device(self.model)
        dtype = _get_model_dtype(self.model)
        was_training = self.model.training

        self._clear_step_state()
        self.model.eval()
        try:
            with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                x_req = input_tensor.detach().to(device=device, dtype=dtype).requires_grad_(True)
                self.model.zero_grad(set_to_none=True)

                logits = self.model(x_req)
                if logits.ndim != 2 or logits.shape[0] != batch_size:
                    raise ValueError(
                        f"Expected 2D logits ({batch_size}, K), got shape {tuple(logits.shape)}."
                    )

                num_classes = int(logits.shape[1])
                probs_t = torch.softmax(logits.detach().float(), dim=-1)
                pred_classes = probs_t.argmax(dim=-1)
                targets_list = _resolve_target_classes(target_classes, pred_classes, num_classes)

                target_idx = torch.tensor(targets_list, dtype=torch.long, device=logits.device)
                batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
                selected_scores = logits[batch_idx, target_idx]
                selected_scores.sum().backward()

            if self._activations is None or self._gradients is None:
                raise RuntimeError(
                    "Grad-CAM failed to capture target_layer activations or gradients. "
                    "Verify target_layer lies on the differentiable forward path."
                )
            if self._activations.ndim != 4 or self._gradients.ndim != 4:
                raise ValueError(
                    f"Expected 4D activations and gradients (B, C, h, w), got "
                    f"{tuple(self._activations.shape)} and {tuple(self._gradients.shape)}."
                )
            if self._activations.shape != self._gradients.shape:
                raise RuntimeError(
                    f"Activation/gradient shape mismatch: {tuple(self._activations.shape)} vs "
                    f"{tuple(self._gradients.shape)}."
                )

            acts = self._activations.float()
            grads = self._gradients.float()
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1))
            cam_up = F.interpolate(
                cam.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            probs_np = probs_t.cpu().numpy().astype(np.float32)
            results: list[GradCAMResult] = []
            for idx in range(batch_size):
                cam_i = cam_up[idx]
                if not torch.isfinite(cam_i).all():
                    cam_i = torch.nan_to_num(cam_i, nan=0.0, posinf=0.0, neginf=0.0)
                max_val = float(cam_i.max().item())
                if max_val > 1e-12:
                    norm_cam = (cam_i / max_val).clamp(0.0, 1.0)
                else:
                    norm_cam = torch.zeros_like(cam_i)

                heatmap = norm_cam.detach().cpu().numpy().astype(np.float32)
                pred_cls = int(pred_classes[idx].item())
                results.append(
                    GradCAMResult(
                        heatmap=heatmap,
                        target_class=targets_list[idx],
                        predicted_class=pred_cls,
                        confidence=float(probs_np[idx, pred_cls]),
                        probs=probs_np[idx],
                    )
                )
            return results
        finally:
            self.model.train(was_training)
            self.model.zero_grad(set_to_none=True)
            self._clear_step_state()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> GradCAMResult:
        """Generates a Grad-CAM result for a single image ``(C, H, W)`` or ``(1, C, H, W)``."""
        if input_tensor.ndim == 3:
            batch = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 4 and input_tensor.shape[0] == 1:
            batch = input_tensor
        else:
            raise ValueError(
                f"Expected 3D (C, H, W) or 4D (1, C, H, W) tensor, got shape {tuple(input_tensor.shape)}."
            )
        return self.generate_batch(batch, target_classes=target_class)[0]


def _prepare_heatmap_and_mask(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates and converts 2D heatmap and binary mask arrays."""
    hm = np.asarray(heatmap, dtype=np.float32)
    msk = np.asarray(mask)
    if hm.size == 0 or msk.size == 0:
        return hm, msk
    if hm.ndim != 2 or msk.ndim != 2:
        raise ValueError(
            f"Expected 2D heatmap and 2D mask, got shapes {hm.shape} and {msk.shape}."
        )
    if hm.shape != msk.shape:
        raise ValueError(
            f"Shape mismatch between heatmap {hm.shape} and mask {msk.shape}."
        )
    return hm, msk


def compute_pointing_game_hit(
    heatmap: np.ndarray,
    mask: np.ndarray,
    tolerance_px: int = 0,
) -> bool:
    """Checks whether the peak Grad-CAM activation falls inside the ground-truth tumor mask."""
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, (int, np.integer)) or int(tolerance_px) < 0:
        raise ValueError(f"tolerance_px must be a non-negative integer, got {tolerance_px!r}.")

    hm, msk = _prepare_heatmap_and_mask(heatmap, mask)
    if hm.size == 0 or msk.size == 0:
        return False
    if not np.all(np.isfinite(hm)) or float(np.max(hm)) <= 0.0:
        return False
    if int(np.sum(msk > 0)) == 0:
        return False

    peak_r, peak_c = np.unravel_index(int(np.argmax(hm)), hm.shape)
    if tolerance_px == 0:
        return bool(msk[peak_r, peak_c] > 0)

    ksize = 2 * int(tolerance_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate((msk > 0).astype(np.uint8), kernel)
    return bool(dilated[peak_r, peak_c] > 0)


def compute_energy_inside_mask(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Computes the fraction of positive Grad-CAM attribution energy inside the tumor mask."""
    hm, msk = _prepare_heatmap_and_mask(heatmap, mask)
    if hm.size == 0 or msk.size == 0 or not np.all(np.isfinite(hm)):
        return 0.0

    pos = np.maximum(hm.astype(np.float64), 0.0)
    total_energy = float(pos.sum())
    mask_bool = msk > 0
    if total_energy <= 1e-12 or not np.any(mask_bool):
        return 0.0
    return float(pos[mask_bool].sum() / total_energy)


def _unpack_masked_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts ``(images, labels, masks)`` from dict or tuple/list DataLoader batches."""
    if isinstance(batch, dict):
        if "image" not in batch or "label" not in batch or "mask" not in batch:
            raise ValueError("Dict batch must contain 'image', 'label', and 'mask' keys.")
        images, labels, masks = batch["image"], batch["label"], batch["mask"]
    elif isinstance(batch, (tuple, list)) and len(batch) >= 3:
        images, labels, masks = batch[0], batch[1], batch[2]
    else:
        raise ValueError(
            "Batch must be a dict with 'image'/'label'/'mask' or a sequence of length >= 3."
        )

    images_t = torch.as_tensor(images)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    masks_t = torch.as_tensor(masks)
    if masks_t.ndim == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t.squeeze(1)
    elif masks_t.ndim == 4 and masks_t.shape[3] == 1:
        masks_t = masks_t.squeeze(3)
    if images_t.ndim != 4:
        raise ValueError(f"Expected 4D images tensor (B, C, H, W), got shape {tuple(images_t.shape)}.")
    if labels_t.ndim != 1:
        raise ValueError(f"Expected 1D labels tensor (B,), got shape {tuple(labels_t.shape)}.")
    if masks_t.ndim != 3:
        raise ValueError(f"Expected 3D masks tensor (B, H, W), got shape {tuple(masks_t.shape)}.")
    if not (images_t.shape[0] == labels_t.shape[0] == masks_t.shape[0]):
        raise ValueError(
            f"Batch size mismatch: images={images_t.shape[0]}, "
            f"labels={labels_t.shape[0]}, masks={masks_t.shape[0]}."
        )

    return images_t, labels_t, masks_t


def evaluate_gradcam_against_masks(
    gradcam: GradCAM,
    loader: DataLoader,
    max_samples: int | None = None,
    target_mode: str = "predicted",
    tolerance_px: int = 0,
) -> dict[str, Any]:
    """Evaluates Grad-CAM pointing game hit rate and energy-inside-mask over a DataLoader."""
    mode = target_mode.strip().lower()
    if mode not in {"predicted", "ground_truth"}:
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}. Expected 'predicted' or 'ground_truth'."
        )
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or int(max_samples) <= 0
    ):
        raise ValueError(f"max_samples must be a positive integer when provided, got {max_samples!r}.")
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, (int, np.integer)) or int(tolerance_px) < 0:
        raise ValueError(f"tolerance_px must be a non-negative integer, got {tolerance_px!r}.")

    hits: list[bool] = []
    energies: list[float] = []

    for batch in loader:
        images, labels, masks = _unpack_masked_batch(batch)
        valid_indices = [
            idx for idx in range(int(masks.shape[0])) if float((masks[idx] > 0).sum().item()) > 0.0
        ]
        if not valid_indices:
            continue

        if max_samples is not None:
            remaining = max_samples - len(hits)
            if remaining <= 0:
                break
            valid_indices = valid_indices[:remaining]

        sel_images = images[valid_indices]
        sel_labels = labels[valid_indices]
        sel_masks = masks[valid_indices].detach().cpu().numpy()

        target_classes = None if mode == "predicted" else [int(v) for v in sel_labels.tolist()]
        cam_results = gradcam.generate_batch(sel_images, target_classes=target_classes)

        for result, mask_np in zip(cam_results, sel_masks):
            hits.append(
                compute_pointing_game_hit(result.heatmap, mask_np, tolerance_px=tolerance_px)
            )
            energies.append(compute_energy_inside_mask(result.heatmap, mask_np))

        if max_samples is not None and len(hits) >= max_samples:
            break

    if not hits:
        return {
            "n_evaluated": 0,
            "pointing_game_hit_rate": 0.0,
            "mean_energy_in_mask": 0.0,
            "median_energy_in_mask": 0.0,
            "std_energy_in_mask": 0.0,
            "hits": [],
            "energies": [],
        }

    return {
        "n_evaluated": len(hits),
        "pointing_game_hit_rate": float(np.mean(hits)),
        "mean_energy_in_mask": float(np.mean(energies)),
        "median_energy_in_mask": float(np.median(energies)),
        "std_energy_in_mask": float(np.std(energies)),
        "hits": hits,
        "energies": energies,
    }


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Converts grayscale or RGB float/uint8 image array into uint8 RGB ``(H, W, 3)``."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or arr.size == 0:
        raise ValueError(f"Expected non-empty 2D or 3D image array, got shape {arr.shape}.")

    if np.issubdtype(arr.dtype, np.floating):
        arr_f = arr.astype(np.float32)
        if not np.all(np.isfinite(arr_f)):
            raise ValueError("image must contain only finite values.")
        min_val = float(np.min(arr_f))
        max_val = float(np.max(arr_f))
        if min_val >= -1e-5 and max_val <= 1.0 + 1e-5:
            scaled = np.clip(arr_f, 0.0, 1.0) * 255.0
        elif min_val >= 0.0 and max_val <= 255.0:
            scaled = arr_f
        elif max_val > min_val:
            scaled = (arr_f - min_val) / (max_val - min_val) * 255.0
        else:
            scaled = np.zeros_like(arr_f)
        img_u8 = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    else:
        img_u8 = np.clip(arr, 0, 255).astype(np.uint8)

    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    if img_u8.shape[2] == 1:
        return cv2.cvtColor(img_u8[:, :, 0], cv2.COLOR_GRAY2RGB)
    if img_u8.shape[2] == 3:
        return img_u8
    raise ValueError(f"Unsupported channel count in image shape {arr.shape}.")


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Renders a ``uint8`` RGB Grad-CAM heatmap overlay onto an input MRI slice."""
    if isinstance(alpha, bool) or not (math.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha!r}.")

    img_rgb = _to_rgb_uint8(image)
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2 or hm.size == 0:
        raise ValueError(f"Expected non-empty 2D heatmap, got shape {hm.shape}.")
    if not np.all(np.isfinite(hm)):
        raise ValueError("heatmap must contain only finite values.")

    height, width = img_rgb.shape[:2]
    if hm.shape != (height, width):
        hm = cv2.resize(hm, (width, height), interpolation=cv2.INTER_LINEAR)

    hm_u8 = np.clip(np.round(np.clip(hm, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    cam_bgr = cv2.applyColorMap(hm_u8, colormap)
    cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)

    blended = (1.0 - float(alpha)) * img_rgb.astype(np.float32) + float(alpha) * cam_rgb.astype(
        np.float32
    )
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)
