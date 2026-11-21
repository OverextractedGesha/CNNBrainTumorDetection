"""Loss functions and class-imbalance weighting utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Computes balanced inverse-frequency class weights normalized to mean 1.0.

    Args:
        labels: 1D collection of integer class indices.
        num_classes: Total number of classes. Inferred as ``max(labels) + 1`` when None.

    Returns:
        1D ``torch.float32`` tensor of shape ``(num_classes,)`` with mean 1.0.
    """
    if isinstance(labels, torch.Tensor):
        arr = labels.detach().cpu().numpy().astype(np.int64).ravel()
    else:
        arr = np.asarray(labels, dtype=np.int64).ravel()

    if arr.size == 0:
        raise ValueError("Cannot compute class weights from empty labels.")
    if np.any(arr < 0):
        raise ValueError("Labels must be non-negative integers.")

    inferred_classes = int(arr.max()) + 1
    n_classes = inferred_classes if num_classes is None else int(num_classes)
    if n_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {n_classes}.")
    if inferred_classes > n_classes:
        raise ValueError(
            f"Found label {inferred_classes - 1} >= num_classes ({n_classes})."
        )

    counts = np.bincount(arr, minlength=n_classes).astype(np.float64)
    raw_weights = np.zeros(n_classes, dtype=np.float64)
    nonzero_mask = counts > 0
    if not np.any(nonzero_mask):
        return torch.ones(n_classes, dtype=torch.float32)

    raw_weights[nonzero_mask] = float(arr.size) / (float(n_classes) * counts[nonzero_mask])
    mean_weight = float(raw_weights.mean())
    if mean_weight > 0.0:
        raw_weights /= mean_weight

    return torch.tensor(raw_weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional class weights (alpha) and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | Sequence[float] | float | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}.")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction {reduction!r}.")

        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        else:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32).clone()
            if alpha_t.ndim > 1:
                raise ValueError(f"alpha must be scalar or 1D, got shape {tuple(alpha_t.shape)}.")
            self.register_buffer("alpha", alpha_t)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes focal loss from raw logits (B, C) and integer targets (B,)."""
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits of shape (B, C), got {tuple(logits.shape)}.")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Expected 1D targets of shape ({logits.shape[0]},), got {tuple(targets.shape)}."
            )

        logits_f = logits.float()
        targets_long = targets.long()
        ce_loss = F.cross_entropy(
            logits_f,
            targets_long,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if self.gamma > 0.0:
            probs = F.softmax(logits_f, dim=-1)
            pt = probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
            loss = ((1.0 - pt).clamp(min=0.0) ** self.gamma) * ce_loss
        else:
            loss = ce_loss

        if self.alpha is not None:
            alpha_buf = self.alpha.to(device=logits.device, dtype=loss.dtype)
            if alpha_buf.ndim == 0:
                loss = alpha_buf * loss
            else:
                if alpha_buf.shape[0] != logits.shape[1]:
                    raise ValueError(
                        f"alpha length ({alpha_buf.shape[0]}) must match classes ({logits.shape[1]})."
                    )
                loss = alpha_buf[targets_long] * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss_fn(
    loss_name: str = "weighted_ce",
    class_weights: torch.Tensor | Sequence[float] | np.ndarray | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory for classification loss functions."""
    name = loss_name.strip().lower()
    weights = (
        torch.as_tensor(class_weights, dtype=torch.float32)
        if class_weights is not None
        else None
    )

    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name in ("weighted_ce", "weighted_cross_entropy"):
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if name in ("focal", "focal_loss"):
        return FocalLoss(
            gamma=focal_gamma,
            alpha=weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    raise ValueError(
        f"Unsupported loss function {loss_name!r}. "
        "Expected 'ce', 'weighted_ce', or 'focal'."
    )
