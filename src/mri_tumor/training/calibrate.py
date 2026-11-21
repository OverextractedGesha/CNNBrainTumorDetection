"""Stage 1 sensitivity threshold search, Stage 2 temperature scaling, and abstention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

ABSTAIN_LABEL: str = "Tumor detected — subtype uncertain (refer to specialist)"


@dataclass
class ThresholdCalibrationResult:
    """Result summary for sensitivity-constrained binary threshold selection."""

    threshold: float
    sensitivity: float
    specificity: float
    precision: float
    target_sensitivity: float
    achieved_target: bool


def _evaluate_binary_threshold(
    probs_pos: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    target_sensitivity: float,
    achieved_target: bool,
) -> ThresholdCalibrationResult:
    """Computes sensitivity, specificity, and precision at a decision threshold."""
    preds = probs_pos >= threshold
    pos_mask = targets == 1
    neg_mask = targets == 0

    tp = int(np.sum(preds & pos_mask))
    fn = int(np.sum((~preds) & pos_mask))
    tn = int(np.sum((~preds) & neg_mask))
    fp = int(np.sum(preds & neg_mask))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return ThresholdCalibrationResult(
        threshold=float(threshold),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        target_sensitivity=float(target_sensitivity),
        achieved_target=achieved_target,
    )


def find_sensitivity_threshold(
    probs_pos: np.ndarray,
    targets: np.ndarray,
    target_sensitivity: float = 0.98,
    thresholds: np.ndarray | Sequence[float] | None = None,
    min_threshold: float = 0.0,
) -> ThresholdCalibrationResult:
    """Finds the highest threshold achieving ``sensitivity >= target_sensitivity``.

    Among candidate thresholds meeting ``target_sensitivity``, selects the one maximizing
    specificity (breaking ties by highest threshold). If no candidate achieves the target,
    falls back to the candidate maximizing sensitivity (then specificity, then threshold)
    with ``achieved_target=False``.
    """
    if not (0.0 < target_sensitivity <= 1.0):
        raise ValueError(
            f"target_sensitivity must be in (0, 1], got {target_sensitivity}."
        )
    if not (math.isfinite(min_threshold) and 0.0 <= min_threshold <= 1.0):
        raise ValueError(f"min_threshold must be in [0, 1], got {min_threshold}.")

    probs_arr = np.asarray(probs_pos, dtype=np.float64).ravel()
    targets_arr = np.asarray(targets, dtype=np.int64).ravel()

    if probs_arr.size == 0 or probs_arr.shape != targets_arr.shape:
        raise ValueError(
            f"probs_pos and targets must be non-empty and match in size; "
            f"got {probs_arr.shape} and {targets_arr.shape}."
        )
    if not np.all(np.isfinite(probs_arr)):
        raise ValueError("probs_pos must contain only finite values.")
    if not np.all(np.isin(targets_arr, (0, 1))):
        raise ValueError("targets must contain only binary values 0 and 1.")
    if np.sum(targets_arr == 1) == 0:
        raise ValueError("targets must contain at least one positive sample (label == 1).")

    if thresholds is not None:
        candidates = np.unique(np.asarray(thresholds, dtype=np.float64).ravel())
        if candidates.size == 0 or not np.all(np.isfinite(candidates)):
            raise ValueError("thresholds must be non-empty and contain only finite values.")
        candidates = candidates[candidates >= float(min_threshold)]
        if candidates.size == 0:
            raise ValueError(
                f"No candidate thresholds >= min_threshold ({min_threshold})."
            )
    else:
        unique_probs = np.unique(probs_arr)
        candidates = unique_probs[unique_probs >= float(min_threshold)]
        if candidates.size == 0:
            candidates = np.array([float(min_threshold)], dtype=np.float64)

    evaluated = [
        _evaluate_binary_threshold(
            probs_pos=probs_arr,
            targets=targets_arr,
            threshold=float(thr),
            target_sensitivity=target_sensitivity,
            achieved_target=True,
        )
        for thr in candidates
    ]

    valid = [res for res in evaluated if res.sensitivity >= target_sensitivity]
    if valid:
        return max(valid, key=lambda r: (r.specificity, r.threshold))

    best_fallback = max(
        evaluated,
        key=lambda r: (r.sensitivity, r.specificity, r.threshold),
    )
    best_fallback.achieved_target = False
    return best_fallback


class TemperatureScaler(nn.Module):
    """Post-hoc logit temperature scaling parameterized by bounded ``exp(log_temperature)``."""

    def __init__(
        self,
        init_temperature: float = 1.0,
        min_temperature: float = 1e-2,
        max_temperature: float = 100.0,
    ) -> None:
        super().__init__()
        if not (math.isfinite(min_temperature) and math.isfinite(max_temperature)):
            raise ValueError("Temperature bounds must be finite.")
        if min_temperature <= 0.0 or min_temperature >= max_temperature:
            raise ValueError(
                f"Expected 0 < min_temperature < max_temperature, "
                f"got ({min_temperature}, {max_temperature})."
            )
        if not (math.isfinite(init_temperature) and init_temperature > 0.0):
            raise ValueError(f"init_temperature must be positive and finite, got {init_temperature}.")

        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self._min_log_temp = math.log(self.min_temperature)
        self._max_log_temp = math.log(self.max_temperature)
        clamped_init = min(max(math.log(init_temperature), self._min_log_temp), self._max_log_temp)
        self.log_temperature = nn.Parameter(torch.tensor(clamped_init, dtype=torch.float32))

    def _bounded_temperature_tensor(self) -> torch.Tensor:
        log_t = self.log_temperature.clamp(min=self._min_log_temp, max=self._max_log_temp)
        return torch.exp(log_t).clamp(min=self.min_temperature, max=self.max_temperature)

    @property
    def temperature(self) -> float:
        """Strictly positive scalar temperature value within configured bounds."""
        return float(
            min(max(self._bounded_temperature_tensor().detach().cpu().item(), self.min_temperature), self.max_temperature)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by positive bounded temperature."""
        temp = self._bounded_temperature_tensor().to(device=logits.device)
        if logits.dtype in (torch.float16, torch.bfloat16):
            return (logits.float() / temp).to(dtype=logits.dtype)
        return logits / temp.to(dtype=logits.dtype)

    def fit(
        self,
        logits: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Optimizes temperature via L-BFGS to minimize negative log-likelihood."""
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if not (math.isfinite(lr) and lr > 0.0):
            raise ValueError(f"lr must be positive and finite, got {lr}.")

        logits_t = torch.as_tensor(logits, dtype=torch.float32).detach()
        if logits_t.ndim != 2 or logits_t.shape[0] == 0 or logits_t.shape[1] == 0:
            raise ValueError(
                f"Expected non-empty 2D logits tensor, got shape {tuple(logits_t.shape)}."
            )
        if not torch.isfinite(logits_t).all():
            raise ValueError("logits must contain only finite values.")

        labels_t = torch.as_tensor(labels, dtype=torch.long).detach().to(logits_t.device)
        if labels_t.ndim != 1 or labels_t.shape[0] != logits_t.shape[0]:
            raise ValueError(
                f"Expected 1D labels of length {logits_t.shape[0]}, got shape {tuple(labels_t.shape)}."
            )
        if not torch.all((labels_t >= 0) & (labels_t < logits_t.shape[1])):
            raise ValueError(
                f"labels must lie in [0, {logits_t.shape[1] - 1}]."
            )

        self.to(logits_t.device)
        prev_log_temp = float(self.log_temperature.detach().item())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(_closure)
        with torch.no_grad():
            if not torch.isfinite(self.log_temperature).all():
                self.log_temperature.fill_(prev_log_temp)
            self.log_temperature.clamp_(min=self._min_log_temp, max=self._max_log_temp)
        return self.temperature


def apply_temperature_and_abstention(
    logits: np.ndarray | torch.Tensor,
    temperature: float = 1.0,
    uncertainty_threshold: float = 0.70,
    class_names: Sequence[str] = ("glioma", "meningioma", "pituitary"),
) -> dict[str, Any]:
    """Applies temperature scaling, softmax probabilities, and confidence abstention.

    Samples with max calibrated confidence strictly below ``uncertainty_threshold`` are
    flagged in ``abstain_mask`` and assigned ``ABSTAIN_LABEL``.
    """
    if not (math.isfinite(temperature) and temperature > 0.0):
        raise ValueError(f"temperature must be positive and finite, got {temperature}.")
    if not (0.0 <= uncertainty_threshold <= 1.0):
        raise ValueError(
            f"uncertainty_threshold must be in [0, 1], got {uncertainty_threshold}."
        )

    logits_t = torch.as_tensor(logits, dtype=torch.float32).detach().cpu()
    if logits_t.ndim == 1:
        logits_t = logits_t.unsqueeze(0)
    if logits_t.ndim != 2:
        raise ValueError(f"Expected 2D logits tensor, got shape {tuple(logits_t.shape)}.")
    if logits_t.shape[1] != len(class_names):
        raise ValueError(
            f"Logit class dimension ({logits_t.shape[1]}) does not match "
            f"len(class_names) ({len(class_names)})."
        )
    if not torch.isfinite(logits_t).all():
        raise ValueError("logits must contain only finite values.")

    with torch.no_grad():
        probs = torch.softmax(logits_t / float(temperature), dim=-1).numpy()

    pred_indices = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    abstain_mask = confidences < float(uncertainty_threshold)
    labels = [
        ABSTAIN_LABEL if bool(abstain) else str(class_names[int(idx)])
        for idx, abstain in zip(pred_indices, abstain_mask)
    ]

    return {
        "probs": probs,
        "pred_indices": pred_indices,
        "confidences": confidences,
        "abstain_mask": abstain_mask,
        "labels": labels,
    }
