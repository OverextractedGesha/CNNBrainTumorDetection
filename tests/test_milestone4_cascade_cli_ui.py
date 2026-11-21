"""Unit and integration test suite for Milestone 4 cascade inference and CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import zipfile

import cv2
import gradio as gr
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from mri_tumor.app import (
    RESEARCH_DISCLAIMER,
    analyze_scan,
    build_gradio_app,
    launch_ui,
    main,
)
from mri_tumor.app.web_ui import CUSTOM_CSS, EMPTY_STATUS_MARKDOWN, PredictorManager
from mri_tumor.data.figshare import STAGE2_CLASS_TO_IDX
from mri_tumor.inference import CascadePredictor, CascadeResult
from mri_tumor.models import SimpleCNN
from mri_tumor.training import ABSTAIN_LABEL


def _strict_json_loads(raw_text: str) -> dict:
    """Parses JSON while rejecting non-standard NaN/Infinity constants."""

    def _reject_constant(val: str) -> None:
        raise ValueError(f"Invalid non-RFC8259 JSON constant encountered: {val}")

    return json.loads(raw_text, parse_constant=_reject_constant)


class _ScriptedConvNet(nn.Module):
    """Minimal differentiable CNN returning deterministic logits near ``bias_logits``."""

    def __init__(self, bias_logits: Sequence[float], input_scale: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.fill_(0.02)
            self.conv.bias.fill_(0.5)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, len(bias_logits))
        with torch.no_grad():
            self.head.weight.fill_(1e-4)
            self.head.bias.copy_(torch.tensor(bias_logits, dtype=torch.float32))
        self.input_scale = float(input_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.relu(self.conv(x))
        pooled = self.pool(feat).flatten(1)
        logits = self.head(pooled)
        if self.input_scale != 0.0:
            mean_val = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            shift = torch.cat([-mean_val * self.input_scale, mean_val * self.input_scale], dim=1)
            logits = logits + shift
        return logits


def _make_distinct_brain_slice(
    idx: int,
    size: int = 64,
    with_tumor: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a deterministic 2D synthetic MRI slice with distinct perceptual hash per index."""
    rng = np.random.RandomState(1000 + idx)
    image = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    cx = size // 2 + ((idx % 3) - 1) * 2
    cy = size // 2 + (((idx // 3) % 3) - 1) * 2
    rx = int(size * 0.30) + (idx % 4)
    ry = int(size * 0.34) - (idx % 3)
    cv2.ellipse(image, (cx, cy), (rx, ry), float((idx * 17) % 45), 0, 360, 150.0, -1)

    # Distinct anatomical bar/ventricle pattern so pHash separates synthetic slices cleanly
    bar_x = max(8, min(size - 12, cx - 8 + (idx * 5) % 16))
    bar_y = max(8, min(size - 12, cy - 8 + (idx * 7) % 16))
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + 6, bar_y + 4), 95.0, -1)

    noise = rng.normal(0.0, 12.0, (size, size)).astype(np.float32)
    brain_region = image > 0
    image[brain_region] = np.clip(image[brain_region] + noise[brain_region], 40.0, 225.0)

    if with_tumor:
        tx = cx + ((idx % 5) - 2) * 3
        ty = cy + (((idx // 2) % 5) - 2) * 3
        cv2.circle(image, (tx, ty), max(4, size // 12), 245.0, -1)
        cv2.circle(mask, (tx, ty), max(4, size // 12), 1, -1)

    return image, mask


def _write_hdf5_mat(
    file_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    label_raw: int,
    pid: str,
) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        cj = f.create_group("cjdata")
        cj.create_dataset("image", data=image.T)
        cj.create_dataset("tumorMask", data=mask.T.astype(np.float64))
        cj.create_dataset("label", data=np.array([[float(label_raw)]]))
        cj.create_dataset("PID", data=np.array([ord(c) for c in pid], dtype=np.uint16))
        cj.create_dataset("tumorBorder", data=np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float64))
    return file_path


def test_cascade_predictor_routing_calibration_and_gradcam(tmp_path: Path):
    """Verifies healthy clearance, confident subtyping, abstention, Grad-CAM, file I/O, batching, and from_configs."""
    img, _ = _make_distinct_brain_slice(1, size=64, with_tumor=True)
    jpg_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(jpg_path), np.clip(img, 0, 255).astype(np.uint8))

    mat_img, mat_mask = _make_distinct_brain_slice(2, size=64, with_tumor=True)
    mat_path = _write_hdf5_mat(tmp_path / "sample.mat", mat_img, mat_mask, label_raw=2, pid="PAT_TEST")

    # 1. Healthy clearance (Stage 1 tumor_prob < threshold)
    healthy_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([4.0, -4.0]),
        stage2_model=_ScriptedConvNet([5.0, 0.0, 0.0]),
        stage1_threshold=0.35,
        stage2_temperature=1.0,
        stage2_uncertainty_threshold=0.70,
        image_size=64,
        device="cpu",
    )
    res_neg = healthy_predictor.predict_image(img, return_gradcam=True, gradcam_on_negative=False)
    assert isinstance(res_neg, CascadeResult)
    assert not res_neg.stage1_positive
    assert res_neg.predicted_label == "no_tumor"
    assert res_neg.raw_4way_label == "no_tumor"
    assert not res_neg.abstained
    assert res_neg.stage2_probs is None
    assert res_neg.stage2_confidence is None
    assert res_neg.gradcam_heatmap is None
    assert res_neg.gradcam_overlay is None
    assert res_neg.gradcam_stage is None
    assert res_neg.preprocessed_image is not None
    assert res_neg.preprocessed_image.shape == (64, 64, 3)
    assert res_neg.preprocessed_image.dtype == np.uint8

    # Optional Stage 1 Grad-CAM when gradcam_on_negative=True
    res_neg_cam = healthy_predictor.predict_image(img, return_gradcam=True, gradcam_on_negative=True)
    assert res_neg_cam.gradcam_stage == "stage1"
    assert res_neg_cam.gradcam_heatmap is not None
    assert res_neg_cam.gradcam_heatmap.shape == (64, 64)
    assert res_neg_cam.gradcam_heatmap.dtype == np.float32
    assert res_neg_cam.gradcam_overlay is not None
    assert res_neg_cam.gradcam_overlay.shape == (64, 64, 3)
    assert res_neg_cam.gradcam_overlay.dtype == np.uint8

    # 2. Confident tumor subtyping (Stage 1 positive + Stage 2 confidence >= threshold)
    confident_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([-4.0, 4.0]),
        stage2_model=_ScriptedConvNet([0.0, 5.0, 0.0]),  # index 1 -> meningioma
        stage1_threshold=0.35,
        stage2_temperature=1.0,
        stage2_uncertainty_threshold=0.70,
        image_size=64,
        device="cpu",
    )
    res_jpg = confident_predictor.predict_file(jpg_path, return_gradcam=True)
    assert res_jpg.stage1_positive
    assert res_jpg.predicted_label == "meningioma"
    assert res_jpg.raw_4way_label == "meningioma"
    assert not res_jpg.abstained
    assert res_jpg.stage2_probs is not None
    assert pytest.approx(sum(res_jpg.stage2_probs.values()), abs=1e-5) == 1.0
    assert res_jpg.stage2_confidence is not None and res_jpg.stage2_confidence > 0.90
    assert res_jpg.gradcam_stage == "stage2"
    assert res_jpg.gradcam_heatmap is not None and res_jpg.gradcam_heatmap.shape == (64, 64)
    assert float(res_jpg.gradcam_heatmap.min()) >= 0.0
    assert float(res_jpg.gradcam_heatmap.max()) <= 1.0
    assert res_jpg.gradcam_overlay is not None and res_jpg.gradcam_overlay.shape == (64, 64, 3)

    # .mat loading + to_dict serialization checks
    res_mat = confident_predictor.predict_file(mat_path, return_gradcam=False)
    assert res_mat.predicted_label == "meningioma"
    d_no_arr = res_mat.to_dict(include_arrays=False)
    assert "preprocessed_image" not in d_no_arr
    assert json.loads(json.dumps(d_no_arr))["predicted_label"] == "meningioma"
    d_with_arr = res_mat.to_dict(include_arrays=True)
    assert isinstance(d_with_arr["preprocessed_image"], np.ndarray)

    # 3. Uncertain tumor abstention via temperature scaling (ABSTAIN_LABEL)
    abstain_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([-4.0, 4.0]),
        stage2_model=_ScriptedConvNet([4.0, 0.0, 0.0]),  # argmax glioma, softened by T=5.0 -> conf ~0.53 < 0.70
        stage1_threshold=0.35,
        stage2_temperature=5.0,
        stage2_uncertainty_threshold=0.70,
        image_size=64,
        device="cpu",
    )
    res_abstain = abstain_predictor.predict_file(mat_path, return_gradcam=True)
    assert res_abstain.stage1_positive
    assert res_abstain.abstained
    assert res_abstain.predicted_label == ABSTAIN_LABEL
    assert res_abstain.raw_4way_label == "glioma"
    assert res_abstain.stage2_confidence is not None and res_abstain.stage2_confidence < 0.70
    assert res_abstain.gradcam_stage == "stage2"
    assert res_abstain.gradcam_heatmap is not None

    # 4. Batched selective forwarding (1 negative + 1 positive in same batch)
    batch_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([5.0, -5.0], input_scale=10.0),
        stage2_model=_ScriptedConvNet([0.0, 0.0, 5.0]),  # pituitary
        stage1_threshold=0.5,
        stage2_temperature=1.0,
        stage2_uncertainty_threshold=0.70,
        image_size=64,
        device="cpu",
    )
    dark_t = torch.zeros(1, 3, 64, 64)
    bright_t = torch.ones(1, 3, 64, 64)
    batch_results = batch_predictor.predict_batch_tensors(torch.cat([dark_t, bright_t], dim=0))
    assert len(batch_results) == 2
    assert not batch_results[0].stage1_positive
    assert batch_results[0].predicted_label == "no_tumor"
    assert batch_results[1].stage1_positive
    assert batch_results[1].predicted_label == "pituitary"

    # 5. from_configs reading calibration.json and forcing pretrained=False offline
    cfg_dir = tmp_path / "configs"
    ckpt1_dir = tmp_path / "checkpoints" / "stage1"
    ckpt2_dir = tmp_path / "checkpoints" / "stage2"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ckpt1_dir.mkdir(parents=True, exist_ok=True)
    ckpt2_dir.mkdir(parents=True, exist_ok=True)

    s1_model = SimpleCNN(num_classes=2)
    s2_model = SimpleCNN(num_classes=3)
    torch.save({"model_state_dict": s1_model.state_dict()}, ckpt1_dir / "best_model.pt")
    torch.save({"model_state_dict": s2_model.state_dict()}, ckpt2_dir / "best_model.pt")
    (ckpt1_dir / "calibration.json").write_text(
        json.dumps({"threshold": 0.27}), encoding="utf-8"
    )
    (ckpt2_dir / "calibration.json").write_text(
        json.dumps({"temperature": 2.4, "uncertainty_threshold": 0.81}), encoding="utf-8"
    )

    s1_yaml = cfg_dir / "stage1.yaml"
    s2_yaml = cfg_dir / "stage2.yaml"
    cas_yaml = cfg_dir / "cascade.yaml"
    s1_yaml.write_text(
        yaml.safe_dump({
            "stage": "stage1",
            "data": {"image_size": 64, "use_clahe": False, "crop_margin_ratio": 0.05},
            "model": {"arch": "simple_cnn", "pretrained": True, "num_classes": 2},
        }),
        encoding="utf-8",
    )
    s2_yaml.write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "data": {"image_size": 64, "use_clahe": False, "crop_margin_ratio": 0.05},
            "model": {"arch": "simple_cnn", "pretrained": True, "num_classes": 3},
        }),
        encoding="utf-8",
    )
    cas_yaml.write_text(
        yaml.safe_dump({
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/stage1/best_model.pt",
            "stage2_checkpoint": "checkpoints/stage2/best_model.pt",
            "image_size": 64,
            "stage1_threshold": 0.35,
            "stage2_temperature": 1.0,
            "stage2_uncertainty_threshold": 0.70,
        }),
        encoding="utf-8",
    )

    loaded = CascadePredictor.from_configs(cas_yaml, device="cpu", load_weights=True)
    assert loaded.stage1_threshold == pytest.approx(0.27)
    assert loaded.stage2_temperature == pytest.approx(2.4)
    assert loaded.stage2_uncertainty_threshold == pytest.approx(0.81)
    assert loaded.image_size == 64

    # Verify missing checkpoints raise unless allow_missing_checkpoints=True
    missing_cas = cfg_dir / "cascade_missing.yaml"
    missing_cas.write_text(
        yaml.safe_dump({
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/missing1/best_model.pt",
            "stage2_checkpoint": "checkpoints/missing2/best_model.pt",
        }),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Stage 1 checkpoint not found"):
        CascadePredictor.from_configs(missing_cas, device="cpu", load_weights=True)
    missing_ok = CascadePredictor.from_configs(
        missing_cas, device="cpu", load_weights=True, allow_missing_checkpoints=True
    )
    assert missing_ok.stage1_threshold == pytest.approx(0.35)

    # Validation guards
    with pytest.raises(ValueError, match="stage1_threshold"):
        CascadePredictor(s1_model, s2_model, stage1_threshold=1.5)
    with pytest.raises(ValueError, match="stage1_threshold"):
        CascadePredictor(s1_model, s2_model, stage1_threshold=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stage2_temperature"):
        CascadePredictor(s1_model, s2_model, stage2_temperature=0.0)


def test_cli_end_to_end_smoke_prepare_train_evaluate_predict(tmp_path: Path, capsys):
    """Exercises CLI prepare -> train (stage1 & stage2) -> evaluate -> predict end-to-end."""
    raw_zips_dir = tmp_path / "archives"
    raw_zips_dir.mkdir(parents=True, exist_ok=True)
    fig_zip = raw_zips_dir / "figshare.zip"
    np_zip = raw_zips_dir / "nickparvar.zip"

    # Build synthetic Figshare .mat archive (4 patients x 3 classes = 12 patients, 12 slices)
    fig_stage_dir = tmp_path / "src_fig"
    sample_mat_src: Path | None = None
    sample_idx = 0
    for label_raw, class_name in [(1, "meningioma"), (2, "glioma"), (3, "pituitary")]:
        for pat_num in range(4):
            img, mask = _make_distinct_brain_slice(sample_idx, size=64, with_tumor=True)
            mat_p = _write_hdf5_mat(
                fig_stage_dir / f"fig_{class_name}_{pat_num}.mat",
                img,
                mask,
                label_raw=label_raw,
                pid=f"PAT_{class_name[:3].upper()}_{pat_num}",
            )
            if sample_mat_src is None:
                sample_mat_src = mat_p
            sample_idx += 1

    with zipfile.ZipFile(fig_zip, "w") as zf:
        for p in sorted(fig_stage_dir.glob("*.mat")):
            zf.write(p, arcname=p.name)

    # Build synthetic NickParvar .jpg archive (6 images per class across 4 classes = 24 images)
    np_stage_dir = tmp_path / "src_np"
    sample_jpg_src: Path | None = None
    for cls_idx, cls_name in enumerate(("notumor", "glioma", "meningioma", "pituitary")):
        folder = "Training" if cls_idx % 2 == 0 else "Testing"
        for k in range(6):
            unique_idx = 100 + cls_idx * 20 + k
            img, _ = _make_distinct_brain_slice(
                unique_idx,
                size=64,
                with_tumor=(cls_name != "notumor"),
            )
            out_jpg = np_stage_dir / folder / cls_name / f"np_{cls_name}_{k}.jpg"
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_jpg), np.clip(img, 0, 255).astype(np.uint8))
            if sample_jpg_src is None and cls_name == "glioma":
                sample_jpg_src = out_jpg

    with zipfile.ZipFile(np_zip, "w") as zf:
        for p in sorted(np_stage_dir.rglob("*.jpg")):
            zf.write(p, arcname=p.relative_to(np_stage_dir).as_posix())

    # Write temporary YAML configs inside tmp_path/configs/
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    s1_yaml = cfg_dir / "stage1.yaml"
    s2_yaml = cfg_dir / "stage2.yaml"
    cas_yaml = cfg_dir / "cascade.yaml"

    s1_yaml.write_text(
        yaml.safe_dump({
            "stage": "stage1",
            "seed": 42,
            "data": {
                "nickparvar_dir": "data/raw/nickparvar",
                "manifest_path": "data/manifests/stage1_manifest.csv",
                "image_size": 64,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "use_cache": True,
                "cache_dir": "data/cache/stage1",
                "val_fraction": 0.25,
                "test_fraction": 0.25,
                "use_clahe": False,
                "crop_margin_ratio": 0.05,
            },
            "model": {
                "arch": "simple_cnn",
                "backbone": "simple_cnn",
                "pretrained": False,
                "num_classes": 2,
                "dropout": 0.1,
            },
            "training": {
                "epochs": 1,
                "warmup_epochs": 0,
                "freeze_backbone_epochs": 0,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "loss": "weighted_ce",
                "label_smoothing": 0.0,
                "early_stopping_patience": 2,
                "checkpoint_dir": "checkpoints/stage1",
            },
            "calibration": {"target_sensitivity": 0.80},
        }),
        encoding="utf-8",
    )

    s2_yaml.write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "seed": 42,
            "data": {
                "figshare_dir": "data/raw/figshare",
                "manifest_path": "data/manifests/stage2_manifest.csv",
                "image_size": 64,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "use_cache": True,
                "cache_dir": "data/cache/stage2",
                "n_folds": 2,
                "fold": 0,
                "holdout_fraction": 0.25,
                "use_clahe": False,
                "crop_margin_ratio": 0.05,
            },
            "model": {
                "arch": "simple_cnn",
                "backbone": "simple_cnn",
                "pretrained": False,
                "num_classes": 3,
                "dropout": 0.1,
            },
            "training": {
                "epochs": 1,
                "warmup_epochs": 0,
                "freeze_backbone_epochs": 0,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "loss": "weighted_ce",
                "label_smoothing": 0.0,
                "early_stopping_patience": 2,
                "checkpoint_dir": "checkpoints/stage2",
            },
            "calibration": {"uncertainty_threshold": 0.70},
        }),
        encoding="utf-8",
    )

    cas_yaml.write_text(
        yaml.safe_dump({
            "seed": 42,
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/stage1/best_model.pt",
            "stage2_checkpoint": "checkpoints/stage2/best_model.pt",
            "cascade_manifest_path": "data/manifests/cascade_test_manifest.csv",
            "output_dir": "outputs/evaluation",
            "image_size": 64,
            "use_clahe": False,
            "crop_margin_ratio": 0.05,
            "stage1_threshold": 0.35,
            "stage2_temperature": 1.0,
            "stage2_uncertainty_threshold": 0.70,
        }),
        encoding="utf-8",
    )

    # 1. Prepare
    rc = main([
        "prepare",
        "--config", str(cas_yaml),
        "--figshare-zip", str(fig_zip),
        "--nickparvar-zip", str(np_zip),
        "--cache",
        "--max-hamming", "6",
    ])
    assert rc == 0

    s1_manifest = tmp_path / "data" / "manifests" / "stage1_manifest.csv"
    s2_manifest = tmp_path / "data" / "manifests" / "stage2_manifest.csv"
    dup_manifest = tmp_path / "data" / "manifests" / "stage1_removed_duplicates.csv"
    cas_manifest = tmp_path / "data" / "manifests" / "cascade_test_manifest.csv"
    for p in (s1_manifest, s2_manifest, dup_manifest, cas_manifest):
        assert p.is_file(), f"Missing manifest {p}"

    s1_df = pd.read_csv(s1_manifest)
    s2_df = pd.read_csv(s2_manifest)
    cas_df = pd.read_csv(cas_manifest)
    assert "cached_path" in s1_df.columns and Path(s1_df["cached_path"].iloc[0]).is_file()
    assert "cached_path" in s2_df.columns and Path(s2_df["cached_path"].iloc[0]).is_file()
    assert len(cas_df) > 0 and "no_tumor" in set(cas_df["class_name"])
    assert set(STAGE2_CLASS_TO_IDX.keys()).issubset(set(cas_df["class_name"]))

    # 2. Train Stage 1 + Stage 2
    assert main([
        "train",
        "--config", str(s1_yaml),
        "--epochs", "1",
        "--batch-size", "2",
        "--num-workers", "0",
    ]) == 0
    s1_ckpt = tmp_path / "checkpoints" / "stage1" / "best_model.pt"
    s1_cal = tmp_path / "checkpoints" / "stage1" / "calibration.json"
    assert s1_ckpt.is_file() and s1_cal.is_file()
    s1_cal_data = _strict_json_loads(s1_cal.read_text(encoding="utf-8"))
    assert 0.0 <= s1_cal_data["threshold"] <= 1.0

    # Verify --skip-if-exists skips cleanly
    assert main(["train", "--config", str(s1_yaml), "--skip-if-exists"]) == 0

    assert main([
        "train",
        "--config", str(s2_yaml),
        "--fold", "0",
        "--epochs", "1",
        "--batch-size", "2",
        "--num-workers", "0",
    ]) == 0
    s2_ckpt = tmp_path / "checkpoints" / "stage2" / "best_model.pt"
    s2_cal = tmp_path / "checkpoints" / "stage2" / "calibration.json"
    assert s2_ckpt.is_file() and s2_cal.is_file()
    s2_cal_data = _strict_json_loads(s2_cal.read_text(encoding="utf-8"))
    assert s2_cal_data["temperature"] > 0.0
    assert s2_cal_data["uncertainty_threshold"] == pytest.approx(0.70)

    # 3. Evaluate
    assert main([
        "evaluate",
        "--config", str(cas_yaml),
        "--gradcam-samples", "4",
        "--batch-size", "2",
        "--num-workers", "0",
    ]) == 0
    report_md = tmp_path / "outputs" / "evaluation" / "evaluation_report.md"
    metrics_json = tmp_path / "outputs" / "evaluation" / "metrics_summary.json"
    assert report_md.is_file()
    assert metrics_json.is_file()

    parsed_summary = _strict_json_loads(metrics_json.read_text(encoding="utf-8"))
    assert "stage1" in parsed_summary
    assert "stage2" in parsed_summary
    assert "cascade" in parsed_summary
    assert "gradcam" in parsed_summary
    assert parsed_summary["gradcam"]["n_evaluated"] > 0
    assert (tmp_path / "outputs" / "evaluation" / "plots" / "gradcam_examples.png").is_file()

    # 4. Predict (--json + --gradcam-out on .jpg, and human-readable on .mat)
    assert sample_jpg_src is not None and sample_mat_src is not None
    capsys.readouterr()
    gradcam_out = tmp_path / "outputs" / "pred_gradcam.png"
    assert main([
        "predict",
        "--config", str(cas_yaml),
        "--image", str(sample_jpg_src),
        "--gradcam-out", str(gradcam_out),
        "--json",
    ]) == 0
    captured = capsys.readouterr()
    pred_json = _strict_json_loads(captured.out)
    assert "predicted_label" in pred_json
    assert "stage1_tumor_prob" in pred_json
    assert pred_json["gradcam_stage"] in {"stage1", "stage2"}
    assert gradcam_out.is_file()
    saved_overlay = cv2.imread(str(gradcam_out))
    assert saved_overlay is not None and saved_overlay.shape == (64, 64, 3)

    assert main([
        "predict",
        "--config", str(cas_yaml),
        "--image", str(sample_mat_src),
    ]) == 0
    human_out = capsys.readouterr().out
    assert "Prediction:" in human_out

    # 5. Verify --resume + --no-cache, --checkpoint-dir override, and lazy ui dispatch
    assert main([
        "train",
        "--config", str(s1_yaml),
        "--epochs", "1",
        "--batch-size", "2",
        "--num-workers", "0",
        "--no-cache",
        "--resume",
    ]) == 0

    custom_ckpt_dir = tmp_path / "checkpoints" / "stage2_cnn_ablation"
    s2_mtime_before = s2_ckpt.stat().st_mtime_ns
    capsys.readouterr()
    assert main([
        "train",
        "--config", str(s2_yaml),
        "--arch", "simple_cnn",
        "--epochs", "1",
        "--batch-size", "2",
        "--num-workers", "0",
        "--checkpoint-dir", str(custom_ckpt_dir),
    ]) == 0
    train_summary_out = capsys.readouterr().out
    assert (custom_ckpt_dir / "best_model.pt").is_file()
    assert (custom_ckpt_dir / "calibration.json").is_file()
    assert s2_ckpt.stat().st_mtime_ns == s2_mtime_before
    assert "best_val_acc=" in train_summary_out
    assert "calibration.json" in train_summary_out

    import sys
    import types

    ui_calls: list[dict] = []
    dummy_web_ui = types.ModuleType("mri_tumor.app.web_ui")

    def _dummy_launch_ui(**kwargs):
        ui_calls.append(kwargs)

    dummy_web_ui.launch_ui = _dummy_launch_ui  # type: ignore[attr-defined]
    prev_mod = sys.modules.get("mri_tumor.app.web_ui")
    sys.modules["mri_tumor.app.web_ui"] = dummy_web_ui
    try:
        assert main(["ui", "--config", str(cas_yaml), "--host", "0.0.0.0", "--port", "7861"]) == 0
    finally:
        if prev_mod is None:
            sys.modules.pop("mri_tumor.app.web_ui", None)
        else:
            sys.modules["mri_tumor.app.web_ui"] = prev_mod
    assert len(ui_calls) == 1
    assert ui_calls[0]["port"] == 7861
    assert ui_calls[0]["host"] == "0.0.0.0"


def test_web_ui_build_and_analyze_scan_jpg_and_mat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercises Gradio UI assembly, missing-checkpoint resilience, and scan analysis on image & .mat inputs."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    s1_yaml = cfg_dir / "stage1.yaml"
    s2_yaml = cfg_dir / "stage2.yaml"
    cas_yaml = cfg_dir / "cascade.yaml"
    s1_yaml.write_text(
        yaml.safe_dump({
            "stage": "stage1",
            "data": {"image_size": 224, "use_clahe": False, "crop_margin_ratio": 0.05},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 2},
        }),
        encoding="utf-8",
    )
    s2_yaml.write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "data": {"image_size": 224, "use_clahe": False, "crop_margin_ratio": 0.05},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 3},
        }),
        encoding="utf-8",
    )
    cas_yaml.write_text(
        yaml.safe_dump({
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/missing_stage1/best_model.pt",
            "stage2_checkpoint": "checkpoints/missing_stage2/best_model.pt",
            "image_size": 224,
            "stage1_threshold": 0.35,
            "stage2_temperature": 1.0,
            "stage2_uncertainty_threshold": 0.70,
        }),
        encoding="utf-8",
    )

    # 1. Missing-checkpoint UI builds cleanly and displays mri-tumor train guidance
    demo_missing = build_gradio_app(cascade_config_path=cas_yaml)
    assert isinstance(demo_missing, gr.Blocks)
    assert len(demo_missing.blocks) > 0
    missing_values = [str(getattr(b, "value", "")) for b in demo_missing.blocks.values()]
    assert any(RESEARCH_DISCLAIMER in val for val in missing_values)
    assert any("mri-tumor train" in val for val in missing_values)

    # 2. Build UI with an active predictor and verify disclaimer, status, and click callbacks
    healthy_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([4.0, -4.0]),
        stage2_model=_ScriptedConvNet([5.0, 0.0, 0.0]),
        stage1_threshold=0.35,
        stage2_temperature=1.0,
        stage2_uncertainty_threshold=0.70,
        image_size=224,
        device="cpu",
    )
    confident_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([-4.0, 4.0]),
        stage2_model=_ScriptedConvNet([0.0, 5.0, 0.0]),
        stage1_threshold=0.35,
        stage2_temperature=1.0,
        stage2_uncertainty_threshold=0.70,
        image_size=224,
        device="cpu",
    )
    abstain_predictor = CascadePredictor(
        stage1_model=_ScriptedConvNet([-4.0, 4.0]),
        stage2_model=_ScriptedConvNet([4.0, 0.0, 0.0]),
        stage1_threshold=0.35,
        stage2_temperature=5.0,
        stage2_uncertainty_threshold=0.70,
        image_size=224,
        device="cpu",
    )

    demo = build_gradio_app(cascade_config_path=cas_yaml, predictor=confident_predictor)
    assert isinstance(demo, gr.Blocks)
    block_values = [str(getattr(b, "value", "")) for b in demo.blocks.values()]
    assert any(RESEARCH_DISCLAIMER in val for val in block_values)
    assert any("Model Status: Ready" in val for val in block_values)
    assert any(EMPTY_STATUS_MARKDOWN in val for val in block_values)

    raw_img, raw_mask = _make_distinct_brain_slice(12, size=128, with_tumor=True)
    rgb_img = np.stack([np.clip(raw_img, 0, 255).astype(np.uint8)] * 3, axis=-1)

    # Verify registered Gradio click callbacks (Run + Clear) execute cleanly
    run_fn = next(fn for fn in demo.fns.values() if getattr(fn, "api_name", None) == "analyze_scan")
    clear_fn = next(fn for fn in demo.fns.values() if getattr(fn, "api_name", None) != "analyze_scan")
    ui_run_res = run_fn.fn(rgb_img, None, 0.35, 0.70, True)
    assert len(ui_run_res) == 5
    assert ui_run_res[0].shape == rgb_img.shape and ui_run_res[0].dtype == np.uint8
    assert ui_run_res[1].shape == (224, 224, 3) and ui_run_res[1].dtype == np.uint8
    ui_clear_res = clear_fn.fn()
    assert ui_clear_res[0] is None and ui_clear_res[1] is None
    assert ui_clear_res[5] is None and ui_clear_res[6] is None
    assert ui_clear_res[7] == EMPTY_STATUS_MARKDOWN
    assert ui_clear_res[8] == {} and ui_clear_res[9] is None

    # 3. Negative numpy image returns clean original slice with no tumor outline
    out_neg, hot_neg, md_neg, probs_neg, json_neg = analyze_scan(
        rgb_img,
        None,
        0.35,
        0.70,
        True,
        predictor=healthy_predictor,
        cascade_config_path=cas_yaml,
    )
    assert out_neg is not None and out_neg.shape == rgb_img.shape and out_neg.dtype == np.uint8
    assert np.array_equal(out_neg, rgb_img)
    assert hot_neg is not None and hot_neg.shape == rgb_img.shape and hot_neg.dtype == np.uint8
    assert np.array_equal(hot_neg, rgb_img)
    assert "No Tumor" in md_neg
    assert "None (No Tumor)" in md_neg
    assert set(probs_neg.keys()) == {"No Tumor", "Tumor"}
    assert sum(probs_neg.values()) == pytest.approx(1.0, abs=1e-5)
    assert json_neg["predicted_label"] == "no_tumor"

    # 4. Positive confident numpy image + dynamic slider override restoration check
    out_pos, hot_pos, md_pos, probs_pos, json_pos = analyze_scan(
        rgb_img,
        None,
        0.35,
        0.70,
        True,
        predictor=confident_predictor,
        cascade_config_path=cas_yaml,
    )
    assert out_pos is not None and out_pos.shape == rgb_img.shape and out_pos.dtype == np.uint8
    assert not np.array_equal(out_pos, rgb_img)
    assert hot_pos is not None and hot_pos.shape == (224, 224, 3) and hot_pos.dtype == np.uint8
    assert "Meningioma" in md_pos
    assert "Model Focus (Grad-CAM Contour)" in md_pos
    assert set(probs_pos.keys()) == {"glioma", "meningioma", "pituitary"}
    assert sum(probs_pos.values()) == pytest.approx(1.0, abs=1e-5)
    assert json_pos["predicted_label"] == "meningioma"
    assert _strict_json_loads(json.dumps(json_pos))["predicted_label"] == "meningioma"

    # High Stage 1 threshold slider override clears the scan cleanly without mutating predictor permanently
    out_override, hot_override, md_override, _, json_override = analyze_scan(
        rgb_img,
        None,
        0.9999,
        0.70,
        False,
        predictor=confident_predictor,
        cascade_config_path=cas_yaml,
    )
    assert np.array_equal(out_override, rgb_img)
    assert np.array_equal(hot_override, rgb_img)
    assert json_override["predicted_label"] == "no_tumor"
    assert "No Tumor" in md_override
    assert confident_predictor.stage1_threshold == pytest.approx(0.35)
    assert confident_predictor.stage2_uncertainty_threshold == pytest.approx(0.70)

    # 5. Abstained tumor case via temperature scaling
    out_abs, hot_abs, md_abs, probs_abs, json_abs = analyze_scan(
        rgb_img,
        None,
        0.35,
        0.70,
        True,
        predictor=abstain_predictor,
        cascade_config_path=cas_yaml,
    )
    assert out_abs is not None and out_abs.shape == rgb_img.shape and out_abs.dtype == np.uint8
    assert not np.array_equal(out_abs, rgb_img)
    assert hot_abs is not None and hot_abs.shape == (224, 224, 3) and hot_abs.dtype == np.uint8
    assert ABSTAIN_LABEL in md_abs
    assert json_abs["abstained"] is True
    assert json_abs["predicted_label"] == ABSTAIN_LABEL
    assert set(probs_abs.keys()) == {"glioma", "meningioma", "pituitary"}

    # 6. Figshare .mat file input (path string and Gradio file-like .name object)
    mat_path = _write_hdf5_mat(
        tmp_path / "ui_sample.mat",
        raw_img,
        raw_mask,
        label_raw=1,
        pid="PAT_UI_MAT",
    )
    for mat_payload in (str(mat_path), SimpleNamespace(name=str(mat_path))):
        out_mat, hot_mat, md_mat, probs_mat, json_mat = analyze_scan(
            None,
            mat_payload,
            0.35,
            0.70,
            True,
            predictor=confident_predictor,
            cascade_config_path=cas_yaml,
        )
        assert out_mat is not None and out_mat.shape == (128, 128, 3) and out_mat.dtype == np.uint8
        assert hot_mat is not None and hot_mat.shape == (224, 224, 3) and hot_mat.dtype == np.uint8
        assert "PAT_UI_MAT" in md_mat
        assert "Ground-Truth Mask (.mat)" in md_mat
        assert json_mat["predicted_label"] == "meningioma"
        assert set(probs_mat.keys()) == {"glioma", "meningioma", "pituitary"}

    # 7. Error handling: missing inputs, invalid .mat file, and missing checkpoints
    with pytest.raises((gr.Error, ValueError), match="Upload a 2D MRI slice"):
        analyze_scan(None, None, 0.35, 0.70, True, False, predictor=confident_predictor)

    bad_mat = tmp_path / "corrupt.mat"
    bad_mat.write_text("not a mat file", encoding="utf-8")
    with pytest.raises((gr.Error, ValueError), match="Failed to read Figshare .mat"):
        analyze_scan(None, str(bad_mat), 0.35, 0.70, True, False, predictor=confident_predictor)

    with pytest.raises((gr.Error, ValueError), match="mri-tumor train"):
        analyze_scan(rgb_img, None, 0.35, 0.70, True, False, predictor=None, cascade_config_path=cas_yaml)

    # 8. PredictorManager caching & launch_ui Gradio 6.31 launch argument forwarding
    mgr = PredictorManager(cascade_config_path=cas_yaml)
    missing_status = mgr.inspect_status(attempt_load=True)
    assert missing_status["ready"] is False
    assert "Stage 1 checkpoint not found" in str(missing_status["error"])

    launch_calls: list[dict] = []

    def _fake_launch(self, **kwargs):
        launch_calls.append(kwargs)
        return (None, "http://127.0.0.1:7865", "")

    monkeypatch.setattr(gr.Blocks, "launch", _fake_launch)
    launch_ret = launch_ui(
        cascade_config_path=cas_yaml,
        host="127.0.0.1",
        port=7865,
        share=False,
        inbrowser=False,
    )
    assert launch_ret[1] == "http://127.0.0.1:7865"
    assert len(launch_calls) == 1
    assert launch_calls[0]["server_name"] == "127.0.0.1"
    assert launch_calls[0]["server_port"] == 7865
    assert launch_calls[0]["css"] == CUSTOM_CSS
    assert isinstance(launch_calls[0]["theme"], gr.themes.Soft)


def test_milestone4_edge_cases_fp16_cuda_nan_isolation_and_gradio_payloads(tmp_path: Path):
    """Covers NaN/dtype/isolation guards, gradcam_samples=0 skipping, and Gradio FileData payloads."""
    from mri_tumor.app.cli import _collect_probs, evaluate_command
    from mri_tumor.app.web_ui import _resolve_mat_path

    # 1. CascadePredictor validation & NaN rejection (must not misroute NaN to no_tumor/stage2)
    class _NanLogitsNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 2), float("nan"), device=x.device, dtype=torch.float32)

    nan_pred = CascadePredictor(
        stage1_model=_NanLogitsNet(),
        stage2_model=_ScriptedConvNet([1.0, 0.0, 0.0]),
        image_size=64,
        device="cpu",
    )
    with pytest.raises(ValueError, match="finite"):
        nan_pred.predict_tensor(torch.full((1, 3, 64, 64), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        nan_pred.predict_tensor(torch.zeros(1, 3, 64, 64))
    with pytest.raises(ValueError, match="finite"):
        nan_pred.predict_batch_tensors(torch.zeros(2, 3, 64, 64))
    with pytest.raises(ValueError, match="empty"):
        nan_pred.preprocess_array(np.empty((0, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        nan_pred.preprocess_array(np.full((32, 32), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="stage2_classes"):
        CascadePredictor(_ScriptedConvNet([1.0, 0.0]), _ScriptedConvNet([1.0, 0.0, 0.0]), stage2_classes="glioma")
    with pytest.raises(ValueError, match="crop_margin_ratio"):
        CascadePredictor(
            _ScriptedConvNet([1.0, 0.0]),
            _ScriptedConvNet([1.0, 0.0, 0.0]),
            crop_margin_ratio=None,  # type: ignore[arg-type]
        )

    # 2. Half-precision / CUDA parity across predict_tensor, predict_batch_tensors, GradCAM, _collect_probs
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1_fp16 = _ScriptedConvNet([-4.0, 4.0]).to(device=test_device).half()
    s2_fp16 = _ScriptedConvNet([0.0, 5.0, 0.0]).to(device=test_device).half()
    fp16_pred = CascadePredictor(
        stage1_model=s1_fp16,
        stage2_model=s2_fp16,
        image_size=64,
        device=test_device,
    )
    img, mask = _make_distinct_brain_slice(9, size=64, with_tumor=True)
    res_fp16 = fp16_pred.predict_image(img, return_gradcam=True)
    assert res_fp16.predicted_label == "meningioma"
    assert res_fp16.gradcam_stage == "stage2"
    assert res_fp16.gradcam_overlay is not None and res_fp16.gradcam_overlay.shape == (64, 64, 3)

    batch_fp16 = fp16_pred.predict_batch_tensors(torch.rand(2, 3, 64, 64))
    assert len(batch_fp16) == 2 and all(r.predicted_label == "meningioma" for r in batch_fp16)

    from torch.utils.data import DataLoader, TensorDataset

    half_loader = DataLoader(TensorDataset(torch.rand(3, 3, 64, 64), torch.tensor([0, 1, 2])), batch_size=2)
    probs_fp16, targets_fp16 = _collect_probs(s2_fp16, half_loader, test_device, temperature=1.5)
    assert probs_fp16.shape == (3, 3) and np.all(np.isfinite(probs_fp16))
    assert targets_fp16.tolist() == [0, 1, 2]

    # 3. Isolated config resolution, orphaned calibration skip, and null YAML fallback
    iso_cfg_dir = tmp_path / "isolated" / "configs"
    iso_ckpt1_dir = tmp_path / "isolated" / "checkpoints" / "stage1"
    iso_ckpt2_dir = tmp_path / "isolated" / "checkpoints" / "stage2"
    iso_cfg_dir.mkdir(parents=True, exist_ok=True)
    iso_ckpt1_dir.mkdir(parents=True, exist_ok=True)
    iso_ckpt2_dir.mkdir(parents=True, exist_ok=True)
    (iso_ckpt1_dir / "calibration.json").write_text(json.dumps({"threshold": 0.11}), encoding="utf-8")

    iso_s1 = iso_cfg_dir / "stage1.yaml"
    iso_s2 = iso_cfg_dir / "stage2.yaml"
    iso_cas = iso_cfg_dir / "cascade.yaml"
    iso_cas.write_text(
        yaml.safe_dump({
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/stage1/best_model.pt",
            "stage2_checkpoint": "checkpoints/stage2/best_model.pt",
            "image_size": None,
            "stage1_threshold": None,
        }),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Path not found"):
        CascadePredictor.from_configs(iso_cas, device="cpu", load_weights=False)

    iso_s1.write_text(
        yaml.safe_dump({
            "stage": "stage1",
            "data": {"image_size": 64},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 2},
        }),
        encoding="utf-8",
    )
    iso_s2.write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "data": {"image_size": 64},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 3},
        }),
        encoding="utf-8",
    )
    # Missing checkpoint with allow_missing_checkpoints=True must ignore orphaned calibration.json
    pred_missing = CascadePredictor.from_configs(
        iso_cas, device="cpu", load_weights=True, allow_missing_checkpoints=True
    )
    assert pred_missing.stage1_threshold == pytest.approx(0.35)
    assert pred_missing.image_size == 64

    # 4. evaluate_command gradcam_samples=0 skips Grad-CAM instead of scanning full dataset
    torch.save({"model_state_dict": SimpleCNN(num_classes=2).state_dict()}, iso_ckpt1_dir / "best_model.pt")
    torch.save({"model_state_dict": SimpleCNN(num_classes=3).state_dict()}, iso_ckpt2_dir / "best_model.pt")
    mat_file = _write_hdf5_mat(
        tmp_path / "isolated" / "holdout.mat", img, mask, label_raw=2, pid="PAT_ISO"
    )
    s2_manifest_p = tmp_path / "isolated" / "s2.csv"
    pd.DataFrame([
        {
            "sample_id": "h1",
            "filepath": str(mat_file),
            "patient_id": "PAT_ISO",
            "class_name": "glioma",
            "class_idx": 0,
            "is_holdout": True,
            "split": "test",
        }
    ]).to_csv(s2_manifest_p, index=False)
    iso_s2.write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "data": {"image_size": 64, "manifest_path": str(s2_manifest_p)},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 3},
        }),
        encoding="utf-8",
    )
    out_eval_0 = tmp_path / "isolated" / "eval_zero"
    evaluate_command(iso_cas, output_dir=out_eval_0, gradcam_samples=0, batch_size=1)
    summary_0 = _strict_json_loads((out_eval_0 / "metrics_summary.json").read_text(encoding="utf-8"))
    assert "stage2" in summary_0
    assert "gradcam" not in summary_0

    with pytest.raises(ValueError, match="gradcam_samples"):
        evaluate_command(iso_cas, gradcam_samples=-1)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_command(iso_cas, batch_size=0)

    # 5. Gradio FileData / list / empty payload handling & malformed config resilience
    assert _resolve_mat_path(SimpleNamespace(path=str(mat_file), orig_name="holdout.mat")) == mat_file
    assert _resolve_mat_path({"path": str(mat_file), "name": "ignored"}) == mat_file
    assert _resolve_mat_path([str(mat_file)]) == mat_file
    assert _resolve_mat_path([]) is None
    assert _resolve_mat_path(SimpleNamespace(name="   ")) is None

    bad_cas = tmp_path / "bad_cascade.yaml"
    bad_cas.write_text(":::invalid_yaml: [", encoding="utf-8")
    bad_mgr = PredictorManager(cascade_config_path=bad_cas)
    bad_status = bad_mgr.inspect_status(attempt_load=True)
    assert bad_status["ready"] is False
    assert bad_status["error"]
    demo_bad = build_gradio_app(cascade_config_path=bad_cas)
    assert isinstance(demo_bad, gr.Blocks)

    with pytest.raises(gr.Error, match="Failed to analyze MRI slice"):
        analyze_scan(
            np.full((32, 32, 3), np.nan, dtype=np.float32),
            None,
            0.35,
            0.70,
            False,
            False,
            predictor=pred_missing,
        )


def test_evaluate_command_ablation_checkpoints(tmp_path: Path):
    """Verifies evaluate_command with ablation_checkpoints generates ablation plot and JSON entry."""
    from mri_tumor.app.cli import evaluate_command

    cfg_dir = tmp_path / "configs"
    ckpt1_dir = tmp_path / "checkpoints" / "stage1"
    ckpt2_dir = tmp_path / "checkpoints" / "stage2"
    ab_dir = tmp_path / "checkpoints" / "stage2_ablation"
    for d in (cfg_dir, ckpt1_dir, ckpt2_dir, ab_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.save({"model_state_dict": SimpleCNN(num_classes=2).state_dict()}, ckpt1_dir / "best_model.pt")
    torch.save({"model_state_dict": SimpleCNN(num_classes=3).state_dict()}, ckpt2_dir / "best_model.pt")
    ab_ckpt = ab_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": SimpleCNN(num_classes=3).state_dict(),
            "config": {
                "stage": "stage2",
                "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 3},
            },
        },
        ab_ckpt,
    )
    (ab_dir / "calibration.json").write_text(json.dumps({"temperature": 1.2}), encoding="utf-8")

    img, mask = _make_distinct_brain_slice(11, size=64, with_tumor=True)
    mat_file = _write_hdf5_mat(tmp_path / "holdout.mat", img, mask, label_raw=2, pid="PAT_AB")
    s2_manifest_p = tmp_path / "s2.csv"
    pd.DataFrame([
        {
            "sample_id": "h1",
            "filepath": str(mat_file),
            "patient_id": "PAT_AB",
            "class_name": "glioma",
            "class_idx": 0,
            "is_holdout": True,
            "split": "test",
        }
    ]).to_csv(s2_manifest_p, index=False)

    (cfg_dir / "stage1.yaml").write_text(
        yaml.safe_dump({
            "stage": "stage1",
            "data": {"image_size": 64},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 2},
        }),
        encoding="utf-8",
    )
    (cfg_dir / "stage2.yaml").write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "data": {"image_size": 64, "manifest_path": str(s2_manifest_p)},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 3},
        }),
        encoding="utf-8",
    )
    cas_yaml = cfg_dir / "cascade.yaml"
    cas_yaml.write_text(
        yaml.safe_dump({
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/stage1/best_model.pt",
            "stage2_checkpoint": "checkpoints/stage2/best_model.pt",
        }),
        encoding="utf-8",
    )

    out_eval = tmp_path / "eval_ablation"
    evaluate_command(
        cas_yaml,
        output_dir=out_eval,
        gradcam_samples=1,
        batch_size=1,
        ablation_checkpoints=[str(ab_ckpt)],
    )

    assert (out_eval / "plots" / "ablation_comparison.png").is_file()
    summary = _strict_json_loads((out_eval / "metrics_summary.json").read_text(encoding="utf-8"))
    assert "ablation" in summary
    assert len(summary["ablation"]) == 2
    assert "Stage2 (simple_cnn)" in summary["ablation"]
    assert "Stage2 (simple_cnn) [stage2_ablation]" in summary["ablation"]


def test_cascade_predictor_stage2_ensemble_averaging_and_from_configs(tmp_path: Path):
    """Verifies 2-model Stage 2 ensemble calibrated averaging and stage2_checkpoints loading."""
    s1 = _ScriptedConvNet([-4.0, 4.0])
    m1 = _ScriptedConvNet([3.0, 0.0, 0.0])
    m2 = _ScriptedConvNet([0.0, 3.0, 0.0])
    t1, t2 = 1.0, 2.0

    ens = CascadePredictor(
        stage1_model=s1,
        stage2_model=m1,
        stage2_models=[m1, m2],
        stage2_temperatures=[t1, t2],
        stage1_threshold=0.35,
        stage2_uncertainty_threshold=0.70,
        image_size=64,
        device="cpu",
    )
    assert len(ens.stage2_models) == 2
    assert ens.stage2_temperatures == pytest.approx([1.0, 2.0])

    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        p1 = torch.softmax(m1(x).float() / t1, dim=-1).numpy()
        p2 = torch.softmax(m2(x).float() / t2, dim=-1).numpy()
    expected = ((p1 + p2) / 2.0).astype(np.float32)

    got = ens.predict_stage2_probs_tensor(x)
    assert got.shape == (2, 3)
    assert got.dtype == np.float32
    assert np.allclose(got, expected, atol=1e-6)

    # Ensemble averages conflicting confident models -> max prob < 0.70 -> abstains
    single_res = ens.predict_tensor(x[0])
    assert single_res.stage1_positive
    assert single_res.abstained
    assert single_res.predicted_label == ABSTAIN_LABEL
    assert single_res.stage2_confidence == pytest.approx(float(np.max(expected[0])), abs=1e-5)

    batch_res = ens.predict_batch_tensors(x)
    assert len(batch_res) == 2
    assert all(r.abstained and r.predicted_label == ABSTAIN_LABEL for r in batch_res)

    # Load stage2_checkpoints list via from_configs (overriding mismatched stage2.yaml arch via ckpt config)
    cfg_dir = tmp_path / "configs"
    ckpt1_dir = tmp_path / "checkpoints" / "stage1"
    fold0_dir = tmp_path / "checkpoints" / "stage2_f0"
    fold1_dir = tmp_path / "checkpoints" / "stage2_f1"
    for d in (cfg_dir, ckpt1_dir, fold0_dir, fold1_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.save({"model_state_dict": SimpleCNN(num_classes=2).state_dict()}, ckpt1_dir / "best_model.pt")
    (ckpt1_dir / "calibration.json").write_text(json.dumps({"threshold": 0.30}), encoding="utf-8")

    scnn0 = SimpleCNN(num_classes=3)
    scnn1 = SimpleCNN(num_classes=3)
    ckpt_model_cfg = {"stage": "stage2", "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 3}}
    torch.save({"model_state_dict": scnn0.state_dict(), "config": ckpt_model_cfg}, fold0_dir / "best_model.pt")
    torch.save({"model_state_dict": scnn1.state_dict(), "config": ckpt_model_cfg}, fold1_dir / "best_model.pt")
    (fold0_dir / "calibration.json").write_text(json.dumps({"temperature": 1.2, "uncertainty_threshold": 0.72}), encoding="utf-8")
    (fold1_dir / "calibration.json").write_text(json.dumps({"temperature": 2.5, "uncertainty_threshold": 0.72}), encoding="utf-8")

    (cfg_dir / "stage1.yaml").write_text(
        yaml.safe_dump({
            "stage": "stage1",
            "data": {"image_size": 64},
            "model": {"arch": "simple_cnn", "pretrained": False, "num_classes": 2},
        }),
        encoding="utf-8",
    )
    (cfg_dir / "stage2.yaml").write_text(
        yaml.safe_dump({
            "stage": "stage2",
            "data": {"image_size": 64},
            "model": {"arch": "cnn", "pretrained": False, "num_classes": 3},
        }),
        encoding="utf-8",
    )
    cas_yaml = cfg_dir / "cascade.yaml"
    cas_yaml.write_text(
        yaml.safe_dump({
            "stage1_config": "configs/stage1.yaml",
            "stage2_config": "configs/stage2.yaml",
            "stage1_checkpoint": "checkpoints/stage1/best_model.pt",
            "stage2_checkpoint": "checkpoints/stage2_f0/best_model.pt",
            "stage2_checkpoints": [
                "checkpoints/stage2_f0/best_model.pt",
                "checkpoints/stage2_f1/best_model.pt",
            ],
            "image_size": 64,
        }),
        encoding="utf-8",
    )

    loaded = CascadePredictor.from_configs(cas_yaml, device="cpu", load_weights=True)
    assert len(loaded.stage2_models) == 2
    assert loaded.stage2_temperatures == pytest.approx([1.2, 2.5])
    assert loaded.stage2_uncertainty_threshold == pytest.approx(0.72)

    scnn0.eval()
    scnn1.eval()
    with torch.no_grad():
        lp0 = torch.softmax(scnn0(x).float() / 1.2, dim=-1).numpy()
        lp1 = torch.softmax(scnn1(x).float() / 2.5, dim=-1).numpy()
    expected_loaded = ((lp0 + lp1) / 2.0).astype(np.float32)
    got_loaded = loaded.predict_stage2_probs_tensor(x)
    assert np.allclose(got_loaded, expected_loaded, atol=1e-6)



