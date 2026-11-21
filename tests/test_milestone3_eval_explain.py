"""Unit test suite for Milestone 3 evaluation metrics, reporting, and Grad-CAM explainability."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from mri_tumor.evaluation import (
    CascadeMetrics,
    Stage1Metrics,
    Stage2Metrics,
    compute_calibration_bins,
    compute_cascade_metrics,
    compute_ece,
    compute_stage1_metrics,
    compute_stage2_metrics,
    generate_evaluation_report,
    plot_ablation_comparison,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_gradcam_examples,
    plot_roc_curves,
)
from mri_tumor.explain import (
    GradCAM,
    GradCAMResult,
    compute_energy_inside_mask,
    compute_pointing_game_hit,
    evaluate_gradcam_against_masks,
    overlay_gradcam,
)
from mri_tumor.models import SimpleCNN, TimmClassifier
from mri_tumor.training import ABSTAIN_LABEL


def _strict_json_loads(path: Path) -> dict:
    """Parses JSON while rejecting non-standard NaN/Infinity constants."""

    def _reject_constant(val: str) -> None:
        raise ValueError(f"Invalid non-RFC8259 JSON constant encountered: {val}")

    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read(), parse_constant=_reject_constant)


def test_stage1_metrics_and_single_class_edge_case():
    """Verifies Stage 1 screening metrics, confusion matrix layout, and single-class guards."""
    targets = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probs = np.array([0.1, 0.4, 0.8, 0.3, 0.5, 0.9], dtype=np.float64)

    metrics = compute_stage1_metrics(targets=targets, probs_pos=probs, threshold=0.5)
    assert isinstance(metrics, Stage1Metrics)
    assert metrics.n_samples == 6
    assert metrics.threshold == pytest.approx(0.5)
    # Predictions at >=0.5: [0, 0, 1, 0, 1, 1] -> TN=2, FP=1, FN=1, TP=2
    assert metrics.confusion_matrix == [[2, 1], [1, 2]]
    assert metrics.accuracy == pytest.approx(4.0 / 6.0)
    assert metrics.sensitivity == pytest.approx(2.0 / 3.0)
    assert metrics.specificity == pytest.approx(2.0 / 3.0)
    assert metrics.balanced_accuracy == pytest.approx(2.0 / 3.0)
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.npv == pytest.approx(2.0 / 3.0)
    assert metrics.f1 == pytest.approx(2.0 / 3.0)
    assert metrics.auroc == pytest.approx(float(roc_auc_score(targets, probs)))
    assert metrics.auprc == pytest.approx(float(average_precision_score(targets, probs)))
    assert metrics.to_dict()["confusion_matrix"] == [[2, 1], [1, 2]]

    # Single-class all-negative targets
    neg_only = compute_stage1_metrics(
        targets=np.array([0, 0, 0]),
        probs_pos=np.array([0.1, 0.6, 0.2]),
        threshold=0.5,
    )
    assert neg_only.auroc is None
    assert neg_only.auprc is None
    assert neg_only.specificity == pytest.approx(2.0 / 3.0)
    assert neg_only.sensitivity == pytest.approx(0.0)
    assert neg_only.balanced_accuracy == pytest.approx(2.0 / 3.0)

    # Single-class all-positive targets
    pos_only = compute_stage1_metrics(
        targets=np.array([1, 1, 1]),
        probs_pos=np.array([0.2, 0.8, 0.9]),
        threshold=0.5,
    )
    assert pos_only.auroc is None
    assert pos_only.auprc is None
    assert pos_only.sensitivity == pytest.approx(2.0 / 3.0)
    assert pos_only.specificity == pytest.approx(0.0)
    assert pos_only.balanced_accuracy == pytest.approx(2.0 / 3.0)


def test_stage2_metrics_ece_brier_and_abstention():
    """Verifies Stage 2 multi-class metrics, ECE, Brier score, and selective abstention."""
    probs = np.array(
        [
            [0.90, 0.05, 0.05],  # conf=0.9, pred=0, true=0 (correct)
            [0.80, 0.10, 0.10],  # conf=0.8, pred=0, true=1 (wrong)
            [0.10, 0.70, 0.20],  # conf=0.7, pred=1, true=1 (correct)
            [0.10, 0.20, 0.70],  # conf=0.7, pred=2, true=2 (correct)
        ],
        dtype=np.float64,
    )
    targets = np.array([0, 1, 1, 2], dtype=np.int64)

    bins = compute_calibration_bins(probs=probs, targets=targets, n_bins=10)
    assert len(bins) == 10
    assert sum(int(b["count"]) for b in bins) == 4

    # Hand-calculated ECE across non-empty bins: (0.6,0.7]:2, (0.7,0.8]:1, (0.8,0.9]:1
    expected_ece = (2 / 4) * abs(1.0 - 0.7) + (1 / 4) * abs(0.0 - 0.8) + (1 / 4) * abs(1.0 - 0.9)
    assert compute_ece(probs=probs, targets=targets, n_bins=10) == pytest.approx(expected_ece)

    # Also verify 1D binary probability ECE path
    ece_1d = compute_ece(probs=np.array([0.9, 0.1, 0.8]), targets=np.array([1, 0, 0]), n_bins=10)
    expected_1d = (1 / 3) * abs(0.0 - 0.8) + (2 / 3) * abs(1.0 - 0.9)
    assert ece_1d == pytest.approx(expected_1d)

    # Hand-calculated multi-class Brier score: mean_i sum_c (p_{i,c} - y_{i,c})^2
    onehot = np.eye(3)[targets]
    expected_brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    s2 = compute_stage2_metrics(
        targets=targets,
        probs=probs,
        uncertainty_threshold=0.75,
        n_bins=10,
    )
    assert isinstance(s2, Stage2Metrics)
    assert s2.n_samples == 4
    assert s2.class_names == ["glioma", "meningioma", "pituitary"]
    assert s2.accuracy == pytest.approx(0.75)
    assert s2.ece == pytest.approx(expected_ece)
    assert s2.brier_score == pytest.approx(expected_brier)
    assert s2.macro_auroc is not None and 0.0 <= s2.macro_auroc <= 1.0
    assert s2.uncertainty_threshold == pytest.approx(0.75)
    assert s2.abstention_rate == pytest.approx(0.5)
    assert s2.selective_accuracy == pytest.approx(0.5)
    assert s2.selective_macro_f1 is not None
    assert set(s2.per_class.keys()) == {"glioma", "meningioma", "pituitary"}

    # All samples abstained -> selective metrics must be None
    s2_all_abstain = compute_stage2_metrics(targets=targets, probs=probs, uncertainty_threshold=0.95)
    assert s2_all_abstain.abstention_rate == pytest.approx(1.0)
    assert s2_all_abstain.selective_accuracy is None
    assert s2_all_abstain.selective_macro_f1 is None

    # 2-class Stage2Metrics (verifies macro_auroc works cleanly for num_classes=2)
    probs_2c = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    targets_2c = np.array([0, 1, 0, 1])
    s2_2c = compute_stage2_metrics(targets=targets_2c, probs=probs_2c, class_names=("a", "b"))
    assert s2_2c.macro_auroc == pytest.approx(1.0)

    # Missing class in targets -> macro_auroc is None
    s2_missing = compute_stage2_metrics(targets=np.array([0, 0, 1, 1]), probs=probs)
    assert s2_missing.macro_auroc is None


def test_cascade_metrics_4way_and_routing_matrix():
    """Verifies 4-way cascade metrics, 4x5 clinical routing, and abstention sensitivity semantics."""
    true_labels = [
        "no_tumor",
        "no_tumor",
        "no_tumor",
        "glioma",
        "glioma",
        "glioma",
        "meningioma",
        "meningioma",
        "pituitary",
        "pituitary",
    ]
    # Sample routing:
    # 0: no_tumor -> no_tumor (healthy TN)
    # 1: no_tumor -> glioma (healthy FP)
    # 2: no_tumor -> ABSTAIN_LABEL (healthy abstained -> lowers specificity)
    # 3: glioma -> glioma (TP)
    # 4: glioma -> ABSTAIN_LABEL (abstained tumor -> counted detected!)
    # 5: glioma -> no_tumor (missed tumor -> lowers sensitivity!)
    # 6: meningioma -> meningioma (TP)
    # 7: meningioma -> ABSTAIN_LABEL (abstained tumor -> counted detected!)
    # 8: pituitary -> pituitary (TP)
    # 9: pituitary -> glioma (subtype error, still tumor-detected!)
    pred_labels = [
        "no_tumor",
        "glioma",
        ABSTAIN_LABEL,
        "glioma",
        ABSTAIN_LABEL,
        "no_tumor",
        "meningioma",
        ABSTAIN_LABEL,
        "pituitary",
        "glioma",
    ]
    raw_4way_preds = [
        "no_tumor",
        "glioma",
        "meningioma",
        "glioma",
        "glioma",
        "no_tumor",
        "meningioma",
        "pituitary",
        "pituitary",
        "glioma",
    ]

    cm_with_raw = compute_cascade_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
        raw_4way_preds=raw_4way_preds,
    )
    assert isinstance(cm_with_raw, CascadeMetrics)
    assert cm_with_raw.n_samples == 10
    assert cm_with_raw.class_names == ["no_tumor", "glioma", "meningioma", "pituitary"]
    assert cm_with_raw.routing_columns == [
        "no_tumor",
        "glioma",
        "meningioma",
        "pituitary",
        "uncertain",
    ]
    # 3 abstained out of 10
    assert cm_with_raw.abstention_rate == pytest.approx(0.3)
    # Resolved samples: idx 0(ok), 1(err), 3(ok), 5(err), 6(ok), 8(ok), 9(err) -> 4/7
    assert cm_with_raw.selective_4way_accuracy == pytest.approx(4.0 / 7.0)
    assert cm_with_raw.selective_4way_macro_f1 is not None
    # True tumors: 7 samples (idx 3..9). Only idx 5 routed to no_tumor -> 6/7 detected!
    assert cm_with_raw.cascade_tumor_sensitivity == pytest.approx(6.0 / 7.0)
    # True healthy: 3 samples (idx 0..2). Only idx 0 routed to no_tumor -> 1/3 specificity!
    assert cm_with_raw.cascade_healthy_specificity == pytest.approx(1.0 / 3.0)
    # With raw_4way_preds: idx 0, 3, 4, 6, 8 match -> 5/10
    assert cm_with_raw.overall_4way_accuracy == pytest.approx(0.5)
    assert cm_with_raw.routing_matrix_4x5 == [
        [1, 1, 0, 0, 1],  # no_tumor -> [no_tumor, glioma, meningioma, pituitary, uncertain]
        [1, 1, 0, 0, 1],  # glioma
        [0, 0, 1, 0, 1],  # meningioma
        [0, 1, 0, 1, 0],  # pituitary
    ]
    assert np.sum(cm_with_raw.confusion_matrix_4way) == 10

    # Without raw_4way_preds: abstained predictions count against overall_4way_accuracy
    cm_no_raw = compute_cascade_metrics(true_labels=true_labels, pred_labels=pred_labels)
    assert cm_no_raw.overall_4way_accuracy == pytest.approx(4.0 / 10.0)
    assert np.sum(cm_no_raw.confusion_matrix_4way) == 7

    # All-abstained edge case (also testing integer -1 and string "uncertain")
    cm_all_abstain = compute_cascade_metrics(
        true_labels=[0, 1, 2, 3],
        pred_labels=[-1, "uncertain", ABSTAIN_LABEL, "abstain"],
    )
    assert cm_all_abstain.abstention_rate == pytest.approx(1.0)
    assert cm_all_abstain.selective_4way_accuracy is None
    assert cm_all_abstain.selective_4way_macro_f1 is None
    assert cm_all_abstain.cascade_tumor_sensitivity == pytest.approx(1.0)
    assert cm_all_abstain.cascade_healthy_specificity == pytest.approx(0.0)
    assert cm_all_abstain.confusion_matrix_4way == [[0] * 4 for _ in range(4)]


def test_gradcam_simple_cnn_and_timm_backbones():
    """Verifies Grad-CAM across SimpleCNN and Timm backbones including frozen backbones and hook cleanup."""
    torch.manual_seed(123)
    x_single = torch.randn(3, 64, 64)
    x_batch = torch.randn(2, 3, 64, 64)

    models = [
        SimpleCNN(num_classes=2, dropout=0.1),
        TimmClassifier("efficientnet_b0", num_classes=2, pretrained=False, dropout=0.1),
        TimmClassifier("resnet50", num_classes=3, pretrained=False, dropout=0.1),
    ]

    for model in models:
        target_layer = model.get_target_layer()
        assert len(target_layer._forward_hooks) == 0

        with GradCAM(model) as cam:
            assert len(target_layer._forward_hooks) == 1

            res_single = cam.generate(x_single)
            assert isinstance(res_single, GradCAMResult)
            assert res_single.heatmap.shape == (64, 64)
            assert res_single.heatmap.dtype == np.float32
            assert np.all(np.isfinite(res_single.heatmap))
            assert 0.0 <= float(res_single.heatmap.min()) <= float(res_single.heatmap.max()) <= 1.0
            assert 0.0 <= res_single.confidence <= 1.0
            assert np.isclose(float(res_single.probs.sum()), 1.0, atol=1e-5)

            res_batch = cam.generate_batch(x_batch, target_classes=[0, 1])
            assert len(res_batch) == 2
            assert [r.target_class for r in res_batch] == [0, 1]
            assert all(r.heatmap.shape == (64, 64) for r in res_batch)

        # Context manager exit must remove all hooks cleanly
        assert len(target_layer._forward_hooks) == 0
        assert cam._grad_hook_handles == []
        with pytest.raises(RuntimeError, match="hooks have been removed"):
            cam.generate(x_single)

    # Verify Grad-CAM still succeeds when backbone is frozen and float64 input is passed
    frozen_model = TimmClassifier("efficientnet_b0", num_classes=2, pretrained=False)
    frozen_model.freeze_backbone()
    frozen_model.train()  # Backbone must remain in eval mode after Grad-CAM
    assert frozen_model.backbone.training is False

    cam_frozen = GradCAM(frozen_model)
    try:
        res_f64 = cam_frozen.generate(x_single.double(), target_class=1)
        assert res_f64.heatmap.shape == (64, 64)
        assert np.all(np.isfinite(res_f64.heatmap))
        assert 0.0 <= float(res_f64.heatmap.min()) <= float(res_f64.heatmap.max()) <= 1.0
        assert frozen_model.training is True
        assert frozen_model.backbone.training is False
    finally:
        cam_frozen.remove_hooks()
        cam_frozen.remove_hooks()  # Idempotent second call
    assert len(frozen_model.get_target_layer()._forward_hooks) == 0


def test_pointing_game_and_energy_inside_mask():
    """Verifies pointing game hits, energy-inside-mask, empty/zero guards, and DataLoader evaluation."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1

    # Unique peak strictly inside mask (0.6 + 0.15 = 0.75 inside, 0.25 outside -> energy = 0.75)
    heatmap_hit = np.zeros((16, 16), dtype=np.float32)
    heatmap_hit[5, 5] = 0.60
    heatmap_hit[5, 6] = 0.15
    heatmap_hit[0, 0] = 0.25
    assert compute_pointing_game_hit(heatmap_hit, mask, tolerance_px=0) is True
    assert compute_energy_inside_mask(heatmap_hit, mask) == pytest.approx(0.75)

    # Peak at (4, 9), 2 pixels right of mask[:, 7]
    heatmap_near = np.zeros((16, 16), dtype=np.float32)
    heatmap_near[4, 9] = 1.0
    assert compute_pointing_game_hit(heatmap_near, mask, tolerance_px=0) is False
    assert compute_pointing_game_hit(heatmap_near, mask, tolerance_px=1) is False
    assert compute_pointing_game_hit(heatmap_near, mask, tolerance_px=2) is True

    # Zero-map, empty-mask, zero-size, and NaN guards
    zero_map = np.zeros((16, 16), dtype=np.float32)
    empty_mask = np.zeros((16, 16), dtype=np.uint8)
    nan_map = np.full((16, 16), np.nan, dtype=np.float32)

    assert compute_pointing_game_hit(zero_map, mask) is False
    assert compute_pointing_game_hit(heatmap_hit, empty_mask) is False
    assert compute_pointing_game_hit(np.empty((0, 0)), np.empty((0, 0))) is False
    assert compute_pointing_game_hit(nan_map, mask) is False

    assert compute_energy_inside_mask(zero_map, mask) == 0.0
    assert compute_energy_inside_mask(heatmap_hit, empty_mask) == 0.0
    assert compute_energy_inside_mask(np.empty((0, 0)), np.empty((0, 0))) == 0.0
    assert compute_energy_inside_mask(nan_map, mask) == 0.0

    # Evaluate over DataLoader yielding (images, labels, masks) with some empty masks skipped
    torch.manual_seed(7)
    images = torch.randn(5, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    masks = torch.zeros(5, 32, 32, dtype=torch.float32)
    masks[1, 8:20, 8:20] = 1.0
    masks[3, 10:22, 10:22] = 1.0
    masks[4, 4:14, 4:14] = 1.0
    # Indices 0 and 2 have empty masks -> must be skipped

    loader = DataLoader(TensorDataset(images, labels, masks), batch_size=2, shuffle=False)
    model = SimpleCNN(num_classes=2, dropout=0.0)
    with GradCAM(model) as cam:
        summary = evaluate_gradcam_against_masks(
            cam,
            loader,
            target_mode="ground_truth",
            tolerance_px=2,
        )
        assert summary["n_evaluated"] == 3
        assert len(summary["hits"]) == 3
        assert len(summary["energies"]) == 3
        assert 0.0 <= summary["pointing_game_hit_rate"] <= 1.0
        assert 0.0 <= summary["mean_energy_in_mask"] <= 1.0
        assert summary["median_energy_in_mask"] == pytest.approx(float(np.median(summary["energies"])))
        assert summary["std_energy_in_mask"] == pytest.approx(float(np.std(summary["energies"])))

        # Verify max_samples cap and dict-batch DataLoader support
        capped = evaluate_gradcam_against_masks(cam, loader, max_samples=2, target_mode="predicted")
        assert capped["n_evaluated"] == 2

        dict_batches = [
            {"image": images[:2], "label": labels[:2], "mask": masks[:2].unsqueeze(1)},
            {"image": images[2:], "label": labels[2:], "mask": masks[2:].unsqueeze(-1)},
        ]
        dict_summary = evaluate_gradcam_against_masks(cam, dict_batches, target_mode="predicted")
        assert dict_summary["n_evaluated"] == 3

        # Verify DataLoader with all-empty masks returns zeroed summary
        empty_loader = DataLoader(
            TensorDataset(images[:2], labels[:2], torch.zeros(2, 32, 32)),
            batch_size=2,
        )
        empty_summary = evaluate_gradcam_against_masks(cam, empty_loader)
        assert empty_summary["n_evaluated"] == 0
        assert empty_summary["hits"] == []


def test_overlay_gradcam_and_report_plots_generation(tmp_path: Path):
    """Verifies overlay_gradcam, all evaluation plots, evaluation_report.md, and strict JSON output."""
    # Overlay tests across grayscale, float [0,1], ImageNet-style normalized, and heatmap resize
    gray_u8 = np.full((32, 32), 128, dtype=np.uint8)
    hm_small = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    ov1 = overlay_gradcam(gray_u8, hm_small, alpha=0.45)
    assert ov1.shape == (32, 32, 3)
    assert ov1.dtype == np.uint8

    norm_img = np.random.RandomState(0).randn(32, 32, 3).astype(np.float32)
    ov2 = overlay_gradcam(norm_img, hm_small, alpha=0.5)
    assert ov2.shape == (32, 32, 3)
    assert ov2.dtype == np.uint8

    # Confusion matrix (3x3 normalized) and 4x5 routing matrix
    cm_path = plot_confusion_matrix(
        [[10, 2, 1], [1, 12, 0], [0, 2, 15]],
        class_names=["glioma", "meningioma", "pituitary"],
        output_path=tmp_path / "plots" / "stage2_cm.png",
        title="Stage 2 Normalized CM",
        normalize=True,
    )
    routing_path = plot_confusion_matrix(
        [[18, 1, 0, 0, 1], [0, 12, 1, 0, 2], [0, 1, 11, 1, 1], [0, 0, 1, 13, 1]],
        class_names=["no_tumor", "glioma", "meningioma", "pituitary", "uncertain"],
        output_path=tmp_path / "plots" / "cascade_routing.png",
        title="Cascade 4x5 Routing",
        normalize=False,
    )
    assert cm_path.exists() and cm_path.stat().st_size > 0
    assert routing_path.exists() and routing_path.stat().st_size > 0

    # Binary ROC & Multiclass ROC (including single-class edge case)
    roc_bin = plot_roc_curves(
        targets=np.array([0, 0, 1, 1]),
        probs=np.array([0.1, 0.3, 0.8, 0.9]),
        class_names=["no_tumor", "tumor"],
        output_path=tmp_path / "plots" / "roc_bin.png",
    )
    roc_multi = plot_roc_curves(
        targets=np.array([0, 1, 2, 0, 1, 2]),
        probs=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        class_names=["glioma", "meningioma", "pituitary"],
        output_path=tmp_path / "plots" / "roc_multi.png",
    )
    roc_single = plot_roc_curves(
        targets=np.array([0, 0, 0]),
        probs=np.array([0.1, 0.4, 0.7]),
        class_names=["no_tumor", "tumor"],
        output_path=tmp_path / "plots" / "roc_single.png",
    )
    assert roc_bin.exists() and roc_multi.exists() and roc_single.exists()

    # Calibration curve
    cal_path = plot_calibration_curve(
        targets=np.array([0, 1, 2, 0, 1, 2]),
        probs=np.array(
            [
                [0.85, 0.10, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.15, 0.80],
                [0.70, 0.20, 0.10],
                [0.20, 0.65, 0.15],
                [0.10, 0.20, 0.70],
            ]
        ),
        output_path=tmp_path / "plots" / "calibration.png",
        n_bins=5,
    )
    assert cal_path.exists()

    # Ablation comparison (including None/NaN safety)
    ablation_table = {
        "SimpleCNN": {"accuracy": 0.82, "macro_f1": 0.80},
        "EfficientNet-B0": {"accuracy": 0.93, "macro_f1": 0.92},
    }
    ablation_path = plot_ablation_comparison(
        results={
            "SimpleCNN": {"accuracy": 0.82, "macro_f1": None},
            "EfficientNet-B0": {"accuracy": 0.93, "macro_f1": float("nan")},
        },
        metrics=["accuracy", "macro_f1"],
        output_path=tmp_path / "plots" / "ablation.png",
    )
    assert ablation_path.exists()

    # Grad-CAM examples grid (with CHW tensor, HWC image, CHW/HWC/None masks)
    img_chw = torch.rand(3, 32, 32)
    img_hwc = np.random.RandomState(1).randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    mask_chw = torch.ones(1, 32, 32)
    mask_hwc = np.ones((32, 32, 1), dtype=np.uint8)
    grid_path = plot_gradcam_examples(
        images=[img_chw, img_hwc, img_hwc],
        masks=[mask_chw, mask_hwc, None],
        heatmaps=[hm_small, hm_small, hm_small],
        titles=["Sample 1 (Glioma)", "Sample 2 (HWC Mask)", "Sample 3 (No Mask)"],
        output_path=tmp_path / "plots" / "gradcam_grid.png",
    )
    assert grid_path.exists()

    # Full Markdown + JSON report generation
    s1 = compute_stage1_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    s2 = compute_stage2_metrics(
        np.array([0, 1, 2]),
        np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]),
        uncertainty_threshold=0.7,
    )
    cascade = compute_cascade_metrics(
        ["no_tumor", "glioma", "meningioma", "pituitary"],
        ["no_tumor", "glioma", ABSTAIN_LABEL, "pituitary"],
        raw_4way_preds=["no_tumor", "glioma", "meningioma", "pituitary"],
    )

    report_path = generate_evaluation_report(
        output_dir=tmp_path,
        stage1_metrics=s1,
        stage2_metrics=s2,
        cascade_metrics=cascade,
        gradcam_metrics={
            "n_evaluated": 12,
            "pointing_game_hit_rate": 0.8333,
            "mean_energy_in_mask": float("nan"),  # Must serialize safely as null
            "median_energy_in_mask": np.float32(0.65),
            "std_energy_in_mask": float("inf"),  # Must serialize safely as null
            "scalar_np": np.array(0.5),
        },
        ablation_table=ablation_table,
        plot_paths={
            "Stage 2 Confusion Matrix": cm_path,
            "Cascade 4x5 Routing": routing_path,
            "Grad-CAM Examples": grid_path,
        },
    )
    assert report_path == tmp_path / "evaluation_report.md"
    assert report_path.exists()

    json_path = tmp_path / "metrics_summary.json"
    assert json_path.exists()
    parsed = _strict_json_loads(json_path)
    assert parsed["stage1"]["n_samples"] == 4
    assert parsed["stage2"]["n_samples"] == 3
    assert parsed["cascade"]["n_samples"] == 4
    assert parsed["gradcam"]["mean_energy_in_mask"] is None
    assert parsed["gradcam"]["std_energy_in_mask"] is None
    assert parsed["gradcam"]["scalar_np"] == pytest.approx(0.5)

    md_text = report_path.read_text(encoding="utf-8")
    assert "## Stage 1: Binary Tumor Screening" in md_text
    assert "## Stage 2: Subtype Classification & Calibration" in md_text
    assert "## End-to-End Cascade Evaluation" in md_text
    assert "### 4x5 Clinical Routing Matrix" in md_text
    assert "## Grad-CAM Localization Overlap" in md_text
    assert "## Model Ablation Comparison" in md_text
    assert "![Stage 2 Confusion Matrix](plots/stage2_cm.png)" in md_text
    assert "| Mean Energy Inside Mask | N/A |" in md_text


def test_eval_and_explain_edge_case_validation(tmp_path: Path):
    """Verifies strict validation errors on invalid shapes, thresholds, out-of-range labels, and NaNs."""
    # Metrics validation
    with pytest.raises(ValueError, match="finite"):
        compute_stage1_metrics(np.array([0, 1]), np.array([0.2, np.nan]))
    with pytest.raises(ValueError, match="threshold"):
        compute_stage1_metrics(np.array([0, 1]), np.array([0.2, 0.8]), threshold=1.2)
    with pytest.raises(ValueError, match="threshold"):
        compute_stage1_metrics(np.array([0, 1]), np.array([0.2, 0.8]), threshold=True)
    with pytest.raises(ValueError, match="lie in"):
        compute_stage1_metrics(np.array([0, 2]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="integer class indices"):
        compute_stage1_metrics(np.array([0.5, 1.0]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="booleans"):
        compute_stage1_metrics(np.array([False, True]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="booleans"):
        compute_ece(np.array([False, True]), np.array([0, 1]))
    with pytest.raises(ValueError, match="n_bins"):
        compute_calibration_bins(np.array([0.2, 0.8]), np.array([0, 1]), n_bins=0)
    with pytest.raises(ValueError, match="Expected probs shape"):
        compute_stage2_metrics(np.array([0, 1]), np.array([[0.5, 0.5], [0.4, 0.6]]))
    with pytest.raises(ValueError, match="uncertainty_threshold"):
        compute_stage2_metrics(
            np.array([0, 1, 2]),
            np.eye(3),
            uncertainty_threshold=float("nan"),
        )

    # Cascade validation
    with pytest.raises(ValueError, match="4 cascade classes"):
        compute_cascade_metrics(["a", "b"], ["a", "b"], class_names=("a", "b"))
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_cascade_metrics(["no_tumor"], ["no_tumor", "glioma"])
    with pytest.raises(ValueError, match="Unrecognized cascade label"):
        compute_cascade_metrics([ABSTAIN_LABEL], ["no_tumor"])
    with pytest.raises(ValueError, match="Invalid integer cascade label"):
        compute_cascade_metrics([-1], ["no_tumor"])
    with pytest.raises(ValueError, match="single string"):
        compute_cascade_metrics("no_tumor", "no_tumor")  # type: ignore[arg-type]

    # Grad-CAM validation
    model = SimpleCNN(num_classes=2)
    with GradCAM(model) as cam:
        with pytest.raises(ValueError, match="4D tensor"):
            cam.generate_batch(torch.randn(3, 32, 32))
        with pytest.raises(ValueError, match="finite"):
            cam.generate_batch(torch.full((1, 3, 32, 32), float("nan")))
        with pytest.raises(ValueError, match="out of range"):
            cam.generate_batch(torch.randn(1, 3, 32, 32), target_classes=2)
        with pytest.raises(ValueError, match="Expected 1D target_classes"):
            cam.generate_batch(torch.randn(1, 3, 32, 32), target_classes=torch.tensor(0))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="3D"):
            cam.generate(torch.randn(2, 3, 32, 32))

        loader = DataLoader(
            TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), torch.ones(2, 32, 32)),
            batch_size=2,
        )
        with pytest.raises(ValueError, match="target_mode"):
            evaluate_gradcam_against_masks(cam, loader, target_mode="invalid")
        with pytest.raises(ValueError, match="max_samples"):
            evaluate_gradcam_against_masks(cam, loader, max_samples=0)
        with pytest.raises(ValueError, match="tolerance_px"):
            evaluate_gradcam_against_masks(cam, loader, tolerance_px=-1)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_pointing_game_hit(np.ones((8, 8)), np.ones((8, 10)))
    with pytest.raises(ValueError, match="tolerance_px"):
        compute_pointing_game_hit(np.ones((8, 8)), np.ones((8, 8)), tolerance_px=-1)
    with pytest.raises(ValueError, match="alpha"):
        overlay_gradcam(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8)), alpha=1.5)
    with pytest.raises(ValueError, match="finite"):
        overlay_gradcam(np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8), np.nan))
    with pytest.raises(ValueError, match="finite"):
        overlay_gradcam(np.full((8, 8, 3), np.nan, dtype=np.float32), np.zeros((8, 8)))

    # Report validation
    with pytest.raises(ValueError, match="finite"):
        plot_confusion_matrix([[1, np.nan], [0, 1]], ["a", "b"], tmp_path / "bad_cm.png")
    with pytest.raises(ValueError, match="non-negative"):
        plot_confusion_matrix([[1, -1], [0, 1]], ["a", "b"], tmp_path / "neg_cm.png")
    with pytest.raises(ValueError, match="Column count"):
        plot_confusion_matrix([[1, 0]], ["a", "b", "c"], tmp_path / "col_cm.png")
    with pytest.raises(ValueError, match="at most 2 class_names"):
        plot_roc_curves(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            ["a", "b", "c"],
            tmp_path / "bad_roc.png",
        )
    with pytest.raises(ValueError, match="results must not be empty"):
        plot_ablation_comparison({}, ["acc"], tmp_path / "bad_ab.png")
    with pytest.raises(ValueError, match="images must not be empty"):
        plot_gradcam_examples([], None, [], [], tmp_path / "bad_grid.png")
