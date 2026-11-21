"""Evaluation metrics for Stage 1 screening, Stage 2 subtyping, calibration, and Cascade routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from mri_tumor.training.calibrate import ABSTAIN_LABEL

DEFAULT_STAGE2_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "pituitary")
DEFAULT_CASCADE_CLASSES: tuple[str, ...] = ("no_tumor", "glioma", "meningioma", "pituitary")


@dataclass
class Stage1Metrics:
    """Binary screening metrics for Stage 1 (no_tumor vs. tumor)."""

    n_samples: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    f1: float
    auroc: float | None
    auprc: float | None
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Serializes Stage 1 metrics to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class Stage2Metrics:
    """Multi-class tumor subtyping and calibration metrics for Stage 2."""

    n_samples: int
    class_names: list[str]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_auroc: float | None
    ece: float
    brier_score: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]
    uncertainty_threshold: float | None = None
    abstention_rate: float | None = None
    selective_accuracy: float | None = None
    selective_macro_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes Stage 2 metrics to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class CascadeMetrics:
    """End-to-end Stage 1 + Stage 2 cascade metrics including abstention routing."""

    n_samples: int
    class_names: list[str]
    routing_columns: list[str]
    overall_4way_accuracy: float
    overall_4way_macro_f1: float
    selective_4way_accuracy: float | None
    selective_4way_macro_f1: float | None
    abstention_rate: float
    cascade_tumor_sensitivity: float
    cascade_healthy_specificity: float
    confusion_matrix_4way: list[list[int]]
    routing_matrix_4x5: list[list[int]]
    per_class_4way: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        """Serializes cascade metrics to a JSON-compatible dictionary."""
        return asdict(self)


def _validate_probabilities(probs: np.ndarray, name: str = "probs") -> np.ndarray:
    """Validates finite probabilities in [0, 1] and clips tiny rounding drift."""
    raw = np.asarray(probs)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must contain numeric probabilities, not booleans.")
    arr = raw.astype(np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < -1e-6) or np.any(arr > 1.0 + 1e-6):
        raise ValueError(f"{name} values must lie in [0, 1].")
    return np.clip(arr, 0.0, 1.0)


def _validate_integer_targets(targets: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Validates 1D integer class indices without silently truncating non-integers."""
    raw = np.asarray(targets)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"targets must be a non-empty 1D array, got shape {raw.shape}.")
    if raw.dtype == np.bool_:
        raise ValueError("targets must be integer class indices, not booleans.")
    if not np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.floating) and np.all(np.isfinite(raw)) and np.all(raw == np.floor(raw)):
            arr = raw.astype(np.int64)
        else:
            raise ValueError(f"targets must contain integer class indices, got dtype {raw.dtype}.")
    else:
        arr = raw.astype(np.int64)

    if num_classes is not None:
        if np.any((arr < 0) | (arr >= num_classes)):
            raise ValueError(f"targets must lie in [0, {num_classes - 1}].")
    return arr


def _split_confidences_and_correctness(
    probs: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts 1D binary or 2D multi-class probabilities into (probs_2d, confidences, correct)."""
    probs_arr = _validate_probabilities(probs, name="probs")
    if probs_arr.ndim == 1:
        probs_2d = np.column_stack([1.0 - probs_arr, probs_arr])
    elif probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
        probs_2d = probs_arr
    else:
        raise ValueError(
            f"Expected 1D binary or 2D (N, K>=2) probabilities, got shape {probs_arr.shape}."
        )

    targets_arr = _validate_integer_targets(targets, num_classes=probs_2d.shape[1])
    if targets_arr.shape[0] != probs_2d.shape[0]:
        raise ValueError(
            f"Sample count mismatch: targets has {targets_arr.shape[0]}, probs has {probs_2d.shape[0]}."
        )

    confidences = np.max(probs_2d, axis=1)
    preds = np.argmax(probs_2d, axis=1)
    correct = (preds == targets_arr).astype(np.float64)
    return probs_2d, confidences, correct


def compute_calibration_bins(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Computes per-bin confidence, accuracy, and sample count statistics for reliability evaluation."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}.")

    _, confidences, correct = _split_confidences_and_correctness(probs, targets)
    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)

    bins: list[dict[str, float | int]] = []
    for idx in range(int(n_bins)):
        low = float(bin_edges[idx])
        high = float(bin_edges[idx + 1])
        if idx == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confidences[mask]))
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bins.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
            }
        )
    return bins


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across equal-width confidence bins.

    Supports both 2D multi-class probabilities ``(N, K)`` and 1D binary positive-class
    probabilities ``(N,)``. Empty confidence bins are safely skipped.
    """
    bins = compute_calibration_bins(probs=probs, targets=targets, n_bins=n_bins)
    total = sum(int(b["count"]) for b in bins)
    if total == 0:
        return 0.0
    ece = sum(
        (int(b["count"]) / float(total)) * abs(float(b["accuracy"]) - float(b["confidence"]))
        for b in bins
        if int(b["count"]) > 0
    )
    return float(ece)


def compute_stage1_metrics(
    targets: np.ndarray,
    probs_pos: np.ndarray,
    threshold: float = 0.5,
) -> Stage1Metrics:
    """Computes Stage 1 binary screening metrics at a given positive probability threshold.

    Single-class ``targets`` are handled safely by returning ``None`` for ``auroc`` and ``auprc``.
    """
    if isinstance(threshold, bool) or not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be finite and in [0, 1], got {threshold}.")

    targets_arr = _validate_integer_targets(targets, num_classes=2)
    probs_arr = _validate_probabilities(probs_pos, name="probs_pos")
    if probs_arr.ndim != 1 or probs_arr.shape != targets_arr.shape:
        raise ValueError(
            f"targets and probs_pos must be 1D arrays with matching shapes, "
            f"got {targets_arr.shape} and {probs_arr.shape}."
        )

    preds = (probs_arr >= float(threshold)).astype(np.int64)
    cm = confusion_matrix(targets_arr, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    n_samples = int(targets_arr.shape[0])
    accuracy = float((tp + tn) / n_samples)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    if (tp + fn) > 0 and (tn + fp) > 0:
        balanced_accuracy = 0.5 * (sensitivity + specificity)
    elif (tp + fn) > 0:
        balanced_accuracy = sensitivity
    else:
        balanced_accuracy = specificity

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    if np.unique(targets_arr).size < 2:
        auroc: float | None = None
        auprc: float | None = None
    else:
        auroc = float(roc_auc_score(targets_arr, probs_arr))
        auprc = float(average_precision_score(targets_arr, probs_arr))

    return Stage1Metrics(
        n_samples=n_samples,
        threshold=float(threshold),
        accuracy=accuracy,
        balanced_accuracy=float(balanced_accuracy),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
        f1=f1,
        auroc=auroc,
        auprc=auprc,
        confusion_matrix=[[tn, fp], [fn, tp]],
    )


def _compute_per_class_dict(
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, dict[str, float | int]], np.ndarray, np.ndarray]:
    """Computes per-class precision, recall, F1, and support for fixed label indices."""
    labels = list(range(len(class_names)))
    prec, rec, f1_vals, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        str(name): {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1_vals[idx]),
            "support": int(support[idx]),
        }
        for idx, name in enumerate(class_names)
    }
    return per_class, rec, support


def compute_stage2_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str] = DEFAULT_STAGE2_CLASSES,
    uncertainty_threshold: float | None = None,
    n_bins: int = 10,
) -> Stage2Metrics:
    """Computes Stage 2 subtype classification, calibration, and optional abstention metrics.

    ``brier_score`` uses the canonical multi-class sum-of-squared-errors formulation:
    ``mean_i sum_c (p_{i,c} - y_{i,c})^2``.
    """
    names_list = [str(name) for name in class_names]
    num_classes = len(names_list)
    if num_classes < 2:
        raise ValueError(f"Expected at least 2 classes, got {num_classes}.")

    if uncertainty_threshold is not None and (
        isinstance(uncertainty_threshold, bool)
        or not (math.isfinite(uncertainty_threshold) and 0.0 <= uncertainty_threshold <= 1.0)
    ):
        raise ValueError(
            f"uncertainty_threshold must be finite and in [0, 1], got {uncertainty_threshold}."
        )

    probs_arr = _validate_probabilities(probs, name="probs")
    if probs_arr.ndim != 2 or probs_arr.shape[1] != num_classes:
        raise ValueError(
            f"Expected probs shape (N, {num_classes}), got {probs_arr.shape}."
        )

    targets_arr = _validate_integer_targets(targets, num_classes=num_classes)
    if targets_arr.shape[0] != probs_arr.shape[0]:
        raise ValueError(
            f"Sample count mismatch: targets has {targets_arr.shape[0]}, probs has {probs_arr.shape[0]}."
        )

    n_samples = int(targets_arr.shape[0])
    preds = np.argmax(probs_arr, axis=1)
    accuracy = float(np.mean(preds == targets_arr))

    per_class, rec_arr, support_arr = _compute_per_class_dict(targets_arr, preds, names_list)
    present_true = support_arr > 0
    balanced_accuracy = float(np.mean(rec_arr[present_true])) if np.any(present_true) else 0.0

    macro_f1 = float(f1_score(targets_arr, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets_arr, preds, average="weighted", zero_division=0))

    if np.unique(targets_arr).size == num_classes:
        row_sums = probs_arr.sum(axis=1, keepdims=True)
        probs_norm = np.where(row_sums > 0.0, probs_arr / np.maximum(row_sums, 1e-12), 1.0 / num_classes)
        if num_classes == 2:
            macro_auroc: float | None = float(roc_auc_score(targets_arr, probs_norm[:, 1]))
        else:
            macro_auroc = float(
                roc_auc_score(
                    targets_arr,
                    probs_norm,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(num_classes)),
                )
            )
    else:
        macro_auroc = None

    ece = compute_ece(probs_arr, targets_arr, n_bins=n_bins)
    onehot = np.eye(num_classes, dtype=np.float64)[targets_arr]
    brier_score = float(np.mean(np.sum((probs_arr - onehot) ** 2, axis=1)))
    cm_list = confusion_matrix(targets_arr, preds, labels=list(range(num_classes))).astype(int).tolist()

    abstention_rate: float | None = None
    selective_accuracy: float | None = None
    selective_macro_f1: float | None = None
    if uncertainty_threshold is not None:
        max_conf = np.max(probs_arr, axis=1)
        abstain_mask = max_conf < float(uncertainty_threshold)
        abstention_rate = float(np.mean(abstain_mask))
        kept = ~abstain_mask
        if np.any(kept):
            selective_accuracy = float(np.mean(preds[kept] == targets_arr[kept]))
            selective_macro_f1 = float(
                f1_score(targets_arr[kept], preds[kept], average="macro", zero_division=0)
            )

    return Stage2Metrics(
        n_samples=n_samples,
        class_names=names_list,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_auroc=macro_auroc,
        ece=ece,
        brier_score=brier_score,
        per_class=per_class,
        confusion_matrix=cm_list,
        uncertainty_threshold=(float(uncertainty_threshold) if uncertainty_threshold is not None else None),
        abstention_rate=abstention_rate,
        selective_accuracy=selective_accuracy,
        selective_macro_f1=selective_macro_f1,
    )


def _normalize_label_token(label: str) -> str:
    """Normalizes string label variants for resilient cascade label parsing."""
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def _map_cascade_labels(
    labels: Sequence[int | str],
    class_names: Sequence[str],
    uncertain_label: str,
    allow_uncertain: bool,
) -> np.ndarray:
    """Maps mixed string/int cascade labels into integer codes (0..K-1, or K for uncertain)."""
    n_classes = len(class_names)
    uncertain_idx = n_classes

    str_to_idx: dict[str, int] = {}
    for idx, name in enumerate(class_names):
        str_to_idx[name.strip().lower()] = idx
        str_to_idx[_normalize_label_token(name)] = idx
    if "no_tumor" in str_to_idx:
        str_to_idx.setdefault("notumor", str_to_idx["no_tumor"])

    uncertain_tokens = {
        uncertain_label.strip().lower(),
        _normalize_label_token(uncertain_label),
        "uncertain",
        "abstain",
    }

    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence of labels, not a single string.")

    mapped: list[int] = []
    for raw in labels:
        if hasattr(raw, "ndim") and getattr(raw, "ndim") == 0 and hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bool):
            raise ValueError(f"Boolean label {raw!r} is not a valid cascade label.")
        if isinstance(raw, (int, np.integer)):
            val = int(raw)
            if 0 <= val < n_classes:
                mapped.append(val)
            elif val == -1 and allow_uncertain:
                mapped.append(uncertain_idx)
            else:
                raise ValueError(f"Invalid integer cascade label {val!r}.")
        elif isinstance(raw, str):
            low = raw.strip().lower()
            norm = _normalize_label_token(raw)
            if low in str_to_idx:
                mapped.append(str_to_idx[low])
            elif norm in str_to_idx:
                mapped.append(str_to_idx[norm])
            elif allow_uncertain and (low in uncertain_tokens or norm in uncertain_tokens):
                mapped.append(uncertain_idx)
            else:
                raise ValueError(f"Unrecognized cascade label {raw!r}.")
        else:
            raise ValueError(f"Unsupported label type {type(raw).__name__}: {raw!r}.")

    return np.asarray(mapped, dtype=np.int64)


def compute_cascade_metrics(
    true_labels: Sequence[int | str],
    pred_labels: Sequence[int | str],
    raw_4way_preds: Sequence[int | str] | None = None,
    class_names: Sequence[str] = DEFAULT_CASCADE_CLASSES,
    uncertain_label: str = ABSTAIN_LABEL,
) -> CascadeMetrics:
    """Computes end-to-end 4-way cascade metrics, clinical routing, and abstention statistics.

    Notes on 4-way overall vs. selective metrics:
      - ``selective_4way_accuracy`` and ``selective_4way_macro_f1`` are evaluated strictly on
        resolved (non-uncertain) predictions.
      - When ``raw_4way_preds`` is ``None``, ``overall_4way_accuracy`` and ``overall_4way_macro_f1``
        evaluate all ``N`` samples treating ``uncertain`` predictions as unresolved/incorrect for
        exact 4-way classification, and ``confusion_matrix_4way`` covers resolved predictions.
      - When ``raw_4way_preds`` is provided, ``overall_4way_*``, ``confusion_matrix_4way``, and
        ``per_class_4way`` reflect the un-abstained 4-way predictions over all ``N`` samples.
      - ``cascade_tumor_sensitivity`` treats any true tumor (`class_idx != 0`) routed to either
        a tumor subtype (`1..3`) or ``uncertain`` (`4`) as detected (not missed as ``no_tumor``).
    """
    names_list = [str(name) for name in class_names]
    n_classes = len(names_list)
    if n_classes != 4:
        raise ValueError(f"Expected 4 cascade classes, got {n_classes}.")
    if len(true_labels) == 0:
        raise ValueError("true_labels must not be empty.")
    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Length mismatch: true_labels has {len(true_labels)}, pred_labels has {len(pred_labels)}."
        )

    true_idx = _map_cascade_labels(
        true_labels, class_names=names_list, uncertain_label=uncertain_label, allow_uncertain=False
    )
    routed_idx = _map_cascade_labels(
        pred_labels, class_names=names_list, uncertain_label=uncertain_label, allow_uncertain=True
    )

    n_samples = int(true_idx.shape[0])
    uncertain_idx = n_classes
    routing_columns = names_list + ["uncertain"]

    routing_matrix = np.zeros((n_classes, n_classes + 1), dtype=int)
    for t_val, p_val in zip(true_idx.tolist(), routed_idx.tolist()):
        routing_matrix[t_val, p_val] += 1

    abstain_mask = routed_idx == uncertain_idx
    abstention_rate = float(np.mean(abstain_mask))

    tumor_true_mask = true_idx != 0
    if np.any(tumor_true_mask):
        cascade_tumor_sensitivity = float(np.mean(routed_idx[tumor_true_mask] != 0))
    else:
        cascade_tumor_sensitivity = 0.0

    healthy_true_mask = true_idx == 0
    if np.any(healthy_true_mask):
        cascade_healthy_specificity = float(np.mean(routed_idx[healthy_true_mask] == 0))
    else:
        cascade_healthy_specificity = 0.0

    resolved_mask = ~abstain_mask
    if np.any(resolved_mask):
        sel_true = true_idx[resolved_mask]
        sel_pred = routed_idx[resolved_mask]
        selective_4way_accuracy: float | None = float(np.mean(sel_true == sel_pred))
        selective_4way_macro_f1: float | None = float(
            f1_score(sel_true, sel_pred, average="macro", zero_division=0)
        )
    else:
        selective_4way_accuracy = None
        selective_4way_macro_f1 = None

    if raw_4way_preds is not None:
        if len(raw_4way_preds) != n_samples:
            raise ValueError(
                f"Length mismatch: raw_4way_preds has {len(raw_4way_preds)}, expected {n_samples}."
            )
        eval_preds = _map_cascade_labels(
            raw_4way_preds, class_names=names_list, uncertain_label=uncertain_label, allow_uncertain=False
        )
        cm_4way = confusion_matrix(true_idx, eval_preds, labels=list(range(n_classes)))
    else:
        eval_preds = routed_idx
        if np.any(resolved_mask):
            cm_4way = confusion_matrix(
                true_idx[resolved_mask], routed_idx[resolved_mask], labels=list(range(n_classes))
            )
        else:
            cm_4way = np.zeros((n_classes, n_classes), dtype=int)

    overall_4way_accuracy = float(np.mean(true_idx == eval_preds))
    active_labels = [
        idx for idx in range(n_classes) if np.any(true_idx == idx) or np.any(eval_preds == idx)
    ]
    overall_4way_macro_f1 = float(
        f1_score(true_idx, eval_preds, labels=active_labels, average="macro", zero_division=0)
    )
    per_class_4way, _, _ = _compute_per_class_dict(true_idx, eval_preds, names_list)

    return CascadeMetrics(
        n_samples=n_samples,
        class_names=names_list,
        routing_columns=routing_columns,
        overall_4way_accuracy=overall_4way_accuracy,
        overall_4way_macro_f1=overall_4way_macro_f1,
        selective_4way_accuracy=selective_4way_accuracy,
        selective_4way_macro_f1=selective_4way_macro_f1,
        abstention_rate=abstention_rate,
        cascade_tumor_sensitivity=cascade_tumor_sensitivity,
        cascade_healthy_specificity=cascade_healthy_specificity,
        confusion_matrix_4way=cm_4way.astype(int).tolist(),
        routing_matrix_4x5=routing_matrix.tolist(),
        per_class_4way=per_class_4way,
    )
