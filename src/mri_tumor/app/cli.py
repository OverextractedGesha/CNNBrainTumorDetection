"""Command-line interface for data preparation, training, evaluation, prediction, and UI."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mri_tumor.data import (
    SliceDataset,
    build_cascade_test_manifest,
    build_figshare_manifest,
    build_nickparvar_manifest,
    cache_preprocessed_manifest,
    create_stage1_splits,
    create_stage2_splits,
    extract_figshare_archive,
    extract_nickparvar_archive,
    get_eval_transforms,
    get_train_transforms,
    resolve_dataloader_kwargs,
    verify_no_patient_leakage,
)
from mri_tumor.data.preprocess import _save_png
from mri_tumor.evaluation import (
    compute_cascade_metrics,
    compute_stage1_metrics,
    compute_stage2_metrics,
    generate_evaluation_report,
    plot_ablation_comparison,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_gradcam_examples,
    plot_roc_curves,
)
from mri_tumor.explain import GradCAM, evaluate_gradcam_against_masks
from mri_tumor.explain.gradcam import _get_model_dtype
from mri_tumor.inference.cascade import (
    CascadePredictor,
    CascadeResult,
    allow_cwd_fallback_for_config,
    load_yaml_config,
    resolve_config_path,
)
from mri_tumor.models import (
    SimpleCNN,
    TimmClassifier,
    build_model_from_config,
)
from mri_tumor.training import (
    TemperatureScaler,
    Trainer,
    build_loss_fn,
    compute_class_weights,
    find_sensitivity_threshold,
    load_checkpoint,
    set_seed,
)


def _release_device_memory(*objs: object) -> None:
    """Releases model/trainer references and clears CUDA cache when available."""
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Parses boolean or string-encoded boolean columns cleanly."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})


def _load_pipeline_configs(
    config_path: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    """Loads cascade, stage1, and stage2 YAML configs from either cascade or stage config."""
    cfg_path = Path(config_path)
    cfg = load_yaml_config(cfg_path)
    allow_cwd = allow_cwd_fallback_for_config(cfg_path)

    if "stage1_config" in cfg or "stage2_config" in cfg:
        s1_path = resolve_config_path(
            cfg.get("stage1_config") or "configs/stage1.yaml",
            cfg_path,
            must_exist=True,
            allow_cwd_fallback=allow_cwd,
        )
        s2_path = resolve_config_path(
            cfg.get("stage2_config") or "configs/stage2.yaml",
            cfg_path,
            must_exist=True,
            allow_cwd_fallback=allow_cwd,
        )
        return (
            cfg_path,
            cfg,
            s1_path,
            load_yaml_config(s1_path),
            s2_path,
            load_yaml_config(s2_path),
        )

    stage = str(cfg.get("stage", "")).lower()
    sibling_cascade = cfg_path.parent / "cascade.yaml"
    if sibling_cascade.is_file() and sibling_cascade.resolve() != cfg_path.resolve():
        return _load_pipeline_configs(sibling_cascade)

    if stage == "stage1":
        s2_path = cfg_path.parent / "stage2.yaml"
        s2_cfg = load_yaml_config(s2_path) if s2_path.is_file() else {"data": {}}
        return cfg_path, {}, cfg_path, cfg, s2_path, s2_cfg

    if stage == "stage2":
        s1_path = cfg_path.parent / "stage1.yaml"
        s1_cfg = load_yaml_config(s1_path) if s1_path.is_file() else {"data": {}}
        return cfg_path, {}, s1_path, s1_cfg, cfg_path, cfg

    raise ValueError(f"Unrecognized config format in {cfg_path}.")


def _dir_has_mat_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        p.is_file() and p.name.lower() not in {"cvind.mat", "cvind"}
        for p in directory.rglob("*.mat")
    )


def _dir_has_image_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    exts = {".jpg", ".jpeg", ".png"}
    return any(p.is_file() and p.suffix.lower() in exts for p in directory.rglob("*"))


def _propagate_cache_columns(
    cascade_df: pd.DataFrame,
    stage1_df: pd.DataFrame,
    stage2_df: pd.DataFrame,
) -> pd.DataFrame:
    """Copies ``cached_path`` and ``cached_mask_path`` from stage manifests into cascade test manifest."""
    if cascade_df.empty:
        return cascade_df

    cache_lookup: dict[str, tuple[str, str]] = {}
    for df in (stage2_df, stage1_df):
        if "cached_path" not in df.columns:
            continue
        for _, row in df.iterrows():
            fp = str(Path(str(row["filepath"])).resolve())
            c_path = "" if pd.isna(row.get("cached_path")) else str(row.get("cached_path"))
            m_path = (
                ""
                if "cached_mask_path" not in df.columns or pd.isna(row.get("cached_mask_path"))
                else str(row.get("cached_mask_path"))
            )
            if c_path:
                cache_lookup[fp] = (c_path, m_path)

    if not cache_lookup:
        return cascade_df

    out = cascade_df.copy()
    cached_paths: list[str] = []
    cached_masks: list[str] = []
    for _, row in out.iterrows():
        fp = str(Path(str(row["filepath"])).resolve())
        c_path, m_path = cache_lookup.get(fp, ("", ""))
        cached_paths.append(c_path)
        cached_masks.append(m_path)

    out["cached_path"] = cached_paths
    out["cached_mask_path"] = cached_masks
    return out


def prepare_command(
    config_path: str | Path,
    figshare_zip: str | Path | None = None,
    nickparvar_zip: str | Path | None = None,
    cache: bool | None = None,
    max_hamming: int = 6,
) -> dict[str, Path]:
    """Extracts datasets if needed, builds deduplicated leak-free splits, and writes manifests."""
    cascade_path, cascade_cfg, s1_path, s1_cfg, s2_path, s2_cfg = _load_pipeline_configs(
        config_path
    )
    allow_cwd = allow_cwd_fallback_for_config(cascade_path)
    s1_data = s1_cfg.get("data", {}) if isinstance(s1_cfg.get("data"), dict) else {}
    s2_data = s2_cfg.get("data", {}) if isinstance(s2_cfg.get("data"), dict) else {}

    figshare_dir = resolve_config_path(
        s2_data.get("figshare_dir", "data/raw/figshare"),
        s2_path,
        allow_cwd_fallback=allow_cwd,
    )
    nickparvar_dir = resolve_config_path(
        s1_data.get("nickparvar_dir", "data/raw/nickparvar"),
        s1_path,
        allow_cwd_fallback=allow_cwd,
    )

    if figshare_zip is not None:
        extract_figshare_archive(figshare_zip, figshare_dir)
    elif not _dir_has_mat_files(figshare_dir):
        auto_fig = resolve_config_path(
            "Dataset/Figshare Dataset.zip",
            cascade_path,
            must_exist=True,
            allow_cwd_fallback=allow_cwd,
        )
        extract_figshare_archive(auto_fig, figshare_dir)

    if nickparvar_zip is not None:
        extract_nickparvar_archive(nickparvar_zip, nickparvar_dir)
    elif not _dir_has_image_files(nickparvar_dir):
        auto_np = resolve_config_path(
            "Dataset/NickParvar Dataset.zip",
            cascade_path,
            must_exist=True,
            allow_cwd_fallback=allow_cwd,
        )
        extract_nickparvar_archive(auto_np, nickparvar_dir)

    default_seed = int(cascade_cfg.get("seed", 42))
    s2_margin = float(
        s2_data.get("crop_margin_ratio", cascade_cfg.get("crop_margin_ratio", 0.05))
    )
    s1_margin = float(
        s1_data.get("crop_margin_ratio", cascade_cfg.get("crop_margin_ratio", 0.05))
    )

    figshare_df = build_figshare_manifest(figshare_dir, crop_margin_ratio=s2_margin)
    if figshare_df.empty:
        raise ValueError(f"No valid Figshare .mat slices found in {figshare_dir}.")
    stage2_df = create_stage2_splits(
        figshare_df,
        n_folds=int(s2_data.get("n_folds", 5)),
        holdout_fraction=float(s2_data.get("holdout_fraction", 0.15)),
        seed=int(s2_cfg.get("seed", default_seed)),
    )
    verify_no_patient_leakage(stage2_df)

    nickparvar_df = build_nickparvar_manifest(nickparvar_dir, crop_margin_ratio=s1_margin)
    if nickparvar_df.empty:
        raise ValueError(f"No valid NickParvar image slices found in {nickparvar_dir}.")
    stage1_df, removed_df = create_stage1_splits(
        nickparvar_df,
        stage2_df,
        val_fraction=float(s1_data.get("val_fraction", 0.15)),
        test_fraction=float(s1_data.get("test_fraction", 0.15)),
        max_hamming_dist=int(max_hamming),
        seed=int(s1_cfg.get("seed", default_seed)),
    )
    cascade_df = build_cascade_test_manifest(stage1_df, stage2_df)

    cache_s1 = bool(s1_data.get("use_cache", False)) if cache is None else bool(cache)
    cache_s2 = bool(s2_data.get("use_cache", False)) if cache is None else bool(cache)

    if cache_s1:
        s1_cache_dir = resolve_config_path(
            s1_data.get("cache_dir", "data/cache/stage1"),
            s1_path,
            allow_cwd_fallback=allow_cwd,
        )
        stage1_df = cache_preprocessed_manifest(
            stage1_df,
            cache_dir=s1_cache_dir,
            image_size=int(s1_data.get("image_size", 224)),
            crop_margin_ratio=s1_margin,
            use_clahe=bool(s1_data.get("use_clahe", False)),
        )

    if cache_s2:
        s2_cache_dir = resolve_config_path(
            s2_data.get("cache_dir", "data/cache/stage2"),
            s2_path,
            allow_cwd_fallback=allow_cwd,
        )
        stage2_df = cache_preprocessed_manifest(
            stage2_df,
            cache_dir=s2_cache_dir,
            image_size=int(s2_data.get("image_size", 224)),
            crop_margin_ratio=s2_margin,
            use_clahe=bool(s2_data.get("use_clahe", False)),
        )

    if cache_s1 or cache_s2:
        cascade_df = _propagate_cache_columns(cascade_df, stage1_df, stage2_df)

    s1_manifest_path = resolve_config_path(
        s1_data.get("manifest_path", "data/manifests/stage1_manifest.csv"),
        s1_path,
        allow_cwd_fallback=allow_cwd,
    )
    s2_manifest_path = resolve_config_path(
        s2_data.get("manifest_path", "data/manifests/stage2_manifest.csv"),
        s2_path,
        allow_cwd_fallback=allow_cwd,
    )
    removed_path = s1_manifest_path.parent / "stage1_removed_duplicates.csv"
    cascade_manifest_path = resolve_config_path(
        cascade_cfg.get("cascade_manifest_path", "data/manifests/cascade_test_manifest.csv"),
        cascade_path,
        allow_cwd_fallback=allow_cwd,
    )

    for out_path, df in (
        (s1_manifest_path, stage1_df),
        (s2_manifest_path, stage2_df),
        (removed_path, removed_df),
        (cascade_manifest_path, cascade_df),
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    print(
        f"Prepared manifests: stage1={len(stage1_df)}, stage2={len(stage2_df)}, "
        f"removed_dups={len(removed_df)}, cascade_test={len(cascade_df)}"
    )
    return {
        "stage1_manifest": s1_manifest_path,
        "stage2_manifest": s2_manifest_path,
        "removed_duplicates": removed_path,
        "cascade_manifest": cascade_manifest_path,
    }


def _drop_cache_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("cached_path", "cached_mask_path") if c in df.columns]
    return df.drop(columns=cols) if cols else df


def train_command(
    config_path: str | Path,
    fold: int | None = None,
    arch: str | None = None,
    backbone: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    lr: float | None = None,
    no_pretrained: bool = False,
    no_cache: bool = False,
    resume: bool = False,
    skip_if_exists: bool = False,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Trains Stage 1 or Stage 2 from YAML config and writes best checkpoint + calibration."""
    cfg_path = Path(config_path)
    cfg = copy.deepcopy(load_yaml_config(cfg_path))
    allow_cwd = allow_cwd_fallback_for_config(cfg_path)

    stage = str(cfg.get("stage", "stage1")).strip().lower()
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported stage {stage!r} in {cfg_path}.")

    data_cfg = cfg.setdefault("data", {})
    model_cfg = cfg.setdefault("model", {})
    train_cfg = cfg.setdefault("training", {})
    cal_cfg = cfg.setdefault("calibration", {})

    if fold is not None:
        data_cfg["fold"] = int(fold)
    if arch is not None:
        model_cfg["arch"] = str(arch)
    if backbone is not None:
        model_cfg["backbone"] = str(backbone)
    if epochs is not None:
        train_cfg["epochs"] = int(epochs)
    if batch_size is not None:
        data_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        data_cfg["num_workers"] = int(num_workers)
    if lr is not None:
        train_cfg["lr"] = float(lr)
    if no_pretrained:
        model_cfg["pretrained"] = False

    resolved_checkpoint_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else resolve_config_path(
            train_cfg.get("checkpoint_dir", f"checkpoints/{stage}"),
            cfg_path,
            allow_cwd_fallback=allow_cwd,
        )
    )
    train_cfg["checkpoint_dir"] = str(resolved_checkpoint_dir)
    best_ckpt_path = resolved_checkpoint_dir / "best_model.pt"
    cal_json_path = resolved_checkpoint_dir / "calibration.json"

    if skip_if_exists and best_ckpt_path.is_file() and cal_json_path.is_file():
        print(f"Skipping {stage}: found existing checkpoint and calibration in {resolved_checkpoint_dir}")
        return {
            "skipped": True,
            "checkpoint": str(best_ckpt_path),
            "calibration": str(cal_json_path),
        }

    set_seed(int(cfg.get("seed", 42)))
    manifest_path = resolve_config_path(
        data_cfg["manifest_path"],
        cfg_path,
        must_exist=True,
        allow_cwd_fallback=allow_cwd,
    )
    df = pd.read_csv(manifest_path)
    if no_cache or not bool(data_cfg.get("use_cache", True)):
        df = _drop_cache_columns(df)

    if stage == "stage2":
        target_fold = int(data_cfg.get("fold", 0))
        is_holdout = _parse_bool_series(df["is_holdout"])
        fold_col = pd.to_numeric(df["fold"], errors="coerce").fillna(-1).astype(int)
        df = df.copy()
        df.loc[(~is_holdout) & (fold_col == target_fold), "split"] = "val"
        df.loc[(~is_holdout) & (fold_col != target_fold), "split"] = "train"
        verify_no_patient_leakage(df)
        train_df = df[(~is_holdout) & (fold_col != target_fold)].reset_index(drop=True)
        val_df = df[(~is_holdout) & (fold_col == target_fold)].reset_index(drop=True)
    else:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Empty split for {stage}: train={len(train_df)}, val={len(val_df)} in {manifest_path}"
        )

    image_size = int(data_cfg.get("image_size", 224))
    use_clahe = bool(data_cfg.get("use_clahe", False))
    crop_margin_ratio = float(data_cfg.get("crop_margin_ratio", 0.05))
    eff_batch_size = int(data_cfg.get("batch_size", 32))
    dl_kwargs = resolve_dataloader_kwargs(
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        is_notebook=False,
    )

    train_ds = SliceDataset(
        train_df,
        stage=stage,
        image_size=image_size,
        transform=get_train_transforms(image_size),
        use_clahe=use_clahe,
        crop_margin_ratio=crop_margin_ratio,
    )
    val_ds = SliceDataset(
        val_df,
        stage=stage,
        image_size=image_size,
        transform=get_eval_transforms(image_size),
        use_clahe=use_clahe,
        crop_margin_ratio=crop_margin_ratio,
    )
    train_loader = DataLoader(train_ds, batch_size=eff_batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=eff_batch_size, shuffle=False, **dl_kwargs)

    label_col = "binary_label" if stage == "stage1" else "class_idx"
    train_labels = train_df[label_col].astype(int).to_numpy()
    num_classes = int(model_cfg.get("num_classes", 2 if stage == "stage1" else 3))
    loss_name = str(train_cfg.get("loss", "weighted_ce"))
    class_weights = (
        compute_class_weights(train_labels, num_classes=num_classes)
        if loss_name.strip().lower() in {"weighted_ce", "weighted_cross_entropy", "focal", "focal_loss"}
        else None
    )
    criterion = build_loss_fn(
        loss_name=loss_name,
        class_weights=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.05)),
        focal_gamma=float(train_cfg.get("focal_gamma", 2.0)),
    )

    if resume and best_ckpt_path.is_file():
        model_cfg["pretrained"] = False

    model: torch.nn.Module | None = None
    trainer: Trainer | None = None
    try:
        model = build_model_from_config(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if resume and best_ckpt_path.is_file():
            load_checkpoint(best_ckpt_path, model, device=device)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            device=device,
            epochs=int(train_cfg.get("epochs", 15)),
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            warmup_epochs=int(train_cfg.get("warmup_epochs", 2)),
            freeze_backbone_epochs=int(train_cfg.get("freeze_backbone_epochs", 2)),
            early_stopping_patience=int(train_cfg.get("early_stopping_patience", 5)),
            checkpoint_dir=resolved_checkpoint_dir,
            config=cfg,
        )
        history = trainer.fit(train_loader, val_loader)

        val_logits, val_targets = trainer.predict_logits(val_loader)
        if stage == "stage1":
            probs_pos = torch.softmax(torch.from_numpy(val_logits).float(), dim=-1)[:, 1].numpy()
            target_sens = float(cal_cfg.get("target_sensitivity", 0.98))
            if np.sum(val_targets == 1) > 0:
                cal_res = find_sensitivity_threshold(
                    probs_pos=probs_pos,
                    targets=val_targets,
                    target_sensitivity=target_sens,
                )
                cal_payload: dict[str, Any] = {
                    "stage": "stage1",
                    "threshold": float(cal_res.threshold),
                    "sensitivity": float(cal_res.sensitivity),
                    "specificity": float(cal_res.specificity),
                    "precision": float(cal_res.precision),
                    "target_sensitivity": float(cal_res.target_sensitivity),
                    "achieved_target": bool(cal_res.achieved_target),
                }
            else:
                cal_payload = {
                    "stage": "stage1",
                    "threshold": float(cal_cfg.get("threshold", 0.35)),
                    "target_sensitivity": target_sens,
                    "achieved_target": False,
                }
        else:
            scaler = TemperatureScaler()
            fitted_temp = scaler.fit(val_logits, val_targets)
            cal_payload = {
                "stage": "stage2",
                "temperature": float(fitted_temp),
                "uncertainty_threshold": float(cal_cfg.get("uncertainty_threshold", 0.70)),
                "fold": int(data_cfg.get("fold", 0)),
            }

        resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(cal_json_path, "w", encoding="utf-8") as f:
            json.dump(cal_payload, f, indent=2)

        best_val_acc = (
            float(history.val_acc[history.best_epoch])
            if 0 <= history.best_epoch < len(history.val_acc)
            else 0.0
        )
        device_label = (
            f"{device} ({torch.cuda.get_device_name(device)})"
            if device.type == "cuda"
            else str(device)
        )
        result_info = {
            "skipped": False,
            "stage": stage,
            "device": str(device),
            "best_epoch": history.best_epoch,
            "best_val_loss": float(history.best_val_loss),
            "best_val_acc": best_val_acc,
            "checkpoint": str(best_ckpt_path),
            "calibration": str(cal_json_path),
            "calibration_values": cal_payload,
        }
        print(
            f"Finished training {stage} on {device_label}: "
            f"best_epoch={history.best_epoch + 1}/{trainer.epochs}, "
            f"best_val_loss={history.best_val_loss:.4f}, "
            f"best_val_acc={best_val_acc:.4f}\n"
            f"  checkpoint:  {best_ckpt_path}\n"
            f"  calibration: {cal_json_path} -> {json.dumps(cal_payload)}",
            flush=True,
        )
        return result_info
    finally:
        del trainer, model, train_loader, val_loader
        _release_device_memory()


def _collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs inference over ``loader`` and returns ``(softmax_probs, targets)``."""
    probs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    model.eval()
    dtype = _get_model_dtype(model)
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device=device, dtype=dtype))
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits during evaluation.")
            probs = torch.softmax(logits.float() / float(temperature), dim=-1).cpu().numpy()
            probs_list.append(probs)
            targets_list.append(targets.numpy())
    if not probs_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(probs_list), np.concatenate(targets_list)


def _collect_gradcam_grid_samples(
    cam: GradCAM,
    loader: DataLoader,
    class_names: Sequence[str],
    max_examples: int = 6,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[np.ndarray], list[str]]:
    vis_imgs: list[torch.Tensor] = []
    vis_masks: list[torch.Tensor] = []
    vis_hms: list[np.ndarray] = []
    vis_titles: list[str] = []

    for images, labels, masks in loader:
        valid = [i for i in range(int(masks.shape[0])) if float((masks[i] > 0).sum().item()) > 0.0]
        if not valid:
            continue
        needed = max_examples - len(vis_imgs)
        sel = valid[:needed]
        cam_results = cam.generate_batch(images[sel])
        for idx, res in zip(sel, cam_results):
            true_lbl = int(labels[idx].item())
            true_name = class_names[true_lbl] if 0 <= true_lbl < len(class_names) else str(true_lbl)
            pred_name = (
                class_names[res.predicted_class]
                if 0 <= res.predicted_class < len(class_names)
                else str(res.predicted_class)
            )
            vis_imgs.append(images[idx])
            vis_masks.append(masks[idx])
            vis_hms.append(res.heatmap)
            vis_titles.append(f"True: {true_name} | Pred: {pred_name} ({res.confidence:.2f})")
        if len(vis_imgs) >= max_examples:
            break

    return vis_imgs, vis_masks, vis_hms, vis_titles


def evaluate_command(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    gradcam_samples: int | None = 24,
    batch_size: int = 32,
    num_workers: int = 0,
    ablation_checkpoints: Sequence[str | Path] | None = None,
) -> Path:
    """Evaluates Stage 1, Stage 2, Cascade routing, and Grad-CAM mask overlap, writing a report."""
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    if gradcam_samples is not None and (
        isinstance(gradcam_samples, bool) or int(gradcam_samples) < 0
    ):
        raise ValueError(f"gradcam_samples must be non-negative, got {gradcam_samples!r}.")

    cascade_path, cascade_cfg, s1_path, s1_cfg, s2_path, s2_cfg = _load_pipeline_configs(
        config_path
    )
    allow_cwd = allow_cwd_fallback_for_config(cascade_path)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_config_path(
            cascade_cfg.get("output_dir") or "outputs/evaluation",
            cascade_path,
            allow_cwd_fallback=allow_cwd,
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dl_kwargs = resolve_dataloader_kwargs(
        num_workers=int(num_workers),
        pin_memory=True,
        is_notebook=False,
    )
    plot_paths: dict[str, str | Path] = {}
    predictor: CascadePredictor | None = CascadePredictor.from_configs(cascade_path)
    try:
        assert predictor is not None
        eval_tf = get_eval_transforms(predictor.image_size)

        def _make_eval_loader(
            manifest_df: pd.DataFrame, stage: str, return_mask: bool = False
        ) -> DataLoader:
            ds = SliceDataset(
                manifest_df,
                stage=stage,
                image_size=predictor.image_size,
                transform=eval_tf,
                return_mask=return_mask,
                use_clahe=predictor.use_clahe,
                crop_margin_ratio=predictor.crop_margin_ratio,
            )
            return DataLoader(ds, batch_size=int(batch_size), shuffle=False, **dl_kwargs)

        stage1_metrics = None
        s1_data = s1_cfg.get("data", {}) if isinstance(s1_cfg.get("data"), dict) else {}
        s1_manifest_raw = s1_data.get("manifest_path")
        if s1_manifest_raw:
            s1_manifest_path = resolve_config_path(
                s1_manifest_raw, s1_path, allow_cwd_fallback=allow_cwd
            )
            if s1_manifest_path.is_file():
                s1_df = pd.read_csv(s1_manifest_path)
                if not bool(s1_data.get("use_cache", True)):
                    s1_df = _drop_cache_columns(s1_df)
                s1_test = s1_df[s1_df["split"] == "test"].reset_index(drop=True)
                if not s1_test.empty:
                    s1_loader = _make_eval_loader(s1_test, stage="stage1")
                    s1_probs_2d, s1_targets = _collect_probs(
                        predictor.stage1_model, s1_loader, predictor.device
                    )
                    s1_probs = s1_probs_2d[:, 1]
                    stage1_metrics = compute_stage1_metrics(
                        targets=s1_targets,
                        probs_pos=s1_probs,
                        threshold=predictor.stage1_threshold,
                    )
                    plot_paths["Stage 1 Confusion Matrix"] = plot_confusion_matrix(
                        stage1_metrics.confusion_matrix,
                        ["no_tumor", "tumor"],
                        plots_dir / "stage1_confusion_matrix.png",
                        title="Stage 1 Screening Confusion Matrix",
                    )
                    plot_paths["Stage 1 ROC Curve"] = plot_roc_curves(
                        s1_targets,
                        s1_probs,
                        ["no_tumor", "tumor"],
                        plots_dir / "stage1_roc_curve.png",
                        title="Stage 1 Screening ROC Curve",
                    )

        stage2_metrics = None
        gradcam_metrics: dict[str, Any] | None = None
        ablation_table: dict[str, dict[str, float]] | None = None
        s2_data = s2_cfg.get("data", {}) if isinstance(s2_cfg.get("data"), dict) else {}
        s2_manifest_raw = s2_data.get("manifest_path")
        if s2_manifest_raw:
            s2_manifest_path = resolve_config_path(
                s2_manifest_raw, s2_path, allow_cwd_fallback=allow_cwd
            )
            if s2_manifest_path.is_file():
                s2_df = pd.read_csv(s2_manifest_path)
                if not bool(s2_data.get("use_cache", True)):
                    s2_df = _drop_cache_columns(s2_df)
                holdout_mask = (
                    _parse_bool_series(s2_df["is_holdout"])
                    if "is_holdout" in s2_df.columns
                    else (s2_df["split"] == "test")
                )
                s2_holdout = s2_df[holdout_mask].reset_index(drop=True)
                if not s2_holdout.empty:
                    s2_loader = _make_eval_loader(s2_holdout, stage="stage2")
                    s2_probs_list: list[np.ndarray] = []
                    s2_targets_list: list[np.ndarray] = []
                    for images, targets in s2_loader:
                        s2_probs_list.append(predictor.predict_stage2_probs_tensor(images))
                        s2_targets_list.append(targets.numpy())
                    s2_probs = (
                        np.concatenate(s2_probs_list, axis=0)
                        if s2_probs_list
                        else np.empty((0, len(predictor.stage2_classes)), dtype=np.float32)
                    )
                    s2_targets = (
                        np.concatenate(s2_targets_list, axis=0)
                        if s2_targets_list
                        else np.empty((0,), dtype=np.int64)
                    )
                    stage2_metrics = compute_stage2_metrics(
                        targets=s2_targets,
                        probs=s2_probs,
                        class_names=predictor.stage2_classes,
                        uncertainty_threshold=predictor.stage2_uncertainty_threshold,
                    )
                    plot_paths["Stage 2 Confusion Matrix"] = plot_confusion_matrix(
                        stage2_metrics.confusion_matrix,
                        predictor.stage2_classes,
                        plots_dir / "stage2_confusion_matrix.png",
                        title="Stage 2 Subtyping Confusion Matrix",
                    )
                    plot_paths["Stage 2 ROC Curves"] = plot_roc_curves(
                        s2_targets,
                        s2_probs,
                        predictor.stage2_classes,
                        plots_dir / "stage2_roc_curves.png",
                        title="Stage 2 Subtyping ROC Curves",
                    )
                    plot_paths["Stage 2 Calibration Curve"] = plot_calibration_curve(
                        s2_targets,
                        s2_probs,
                        plots_dir / "stage2_calibration_curve.png",
                        title="Stage 2 Reliability Diagram",
                    )

                    s2_masked_loader: DataLoader | None = None
                    max_gc: int | None = None
                    if gradcam_samples is None or int(gradcam_samples) > 0:
                        s2_masked_loader = _make_eval_loader(
                            s2_holdout, stage="stage2", return_mask=True
                        )
                        max_gc = int(gradcam_samples) if gradcam_samples is not None else None
                        vis_limit = min(6, max_gc) if max_gc is not None else 6
                        with GradCAM(predictor.stage2_model) as cam:
                            gradcam_metrics = evaluate_gradcam_against_masks(
                                cam,
                                s2_masked_loader,
                                max_samples=max_gc,
                                target_mode="predicted",
                            )
                            vis_imgs, vis_masks, vis_hms, vis_titles = _collect_gradcam_grid_samples(
                                cam,
                                s2_masked_loader,
                                class_names=predictor.stage2_classes,
                                max_examples=vis_limit,
                            )
                        if vis_imgs:
                            plot_paths["Grad-CAM Examples"] = plot_gradcam_examples(
                                images=vis_imgs,
                                masks=vis_masks,
                                heatmaps=vis_hms,
                                titles=vis_titles,
                                output_path=plots_dir / "gradcam_examples.png",
                            )

                    if ablation_checkpoints:
                        ablation_table = {}
                        if isinstance(predictor.stage2_model, TimmClassifier):
                            s2_arch = "cnn"
                        elif isinstance(predictor.stage2_model, SimpleCNN):
                            s2_arch = "simple_cnn"
                        else:
                            s2_arch = str((s2_cfg.get("model") or {}).get("arch", "stage2"))
                        primary_row: dict[str, float] = {
                            "accuracy": float(stage2_metrics.accuracy),
                            "balanced_accuracy": float(stage2_metrics.balanced_accuracy),
                            "macro_f1": float(stage2_metrics.macro_f1),
                            "ece": float(stage2_metrics.ece),
                        }
                        if gradcam_metrics is not None:
                            primary_row["pointing_game_hit_rate"] = float(
                                gradcam_metrics["pointing_game_hit_rate"]
                            )
                            primary_row["mean_energy_in_mask"] = float(
                                gradcam_metrics["mean_energy_in_mask"]
                            )
                        primary_label = (
                            f"Stage2 ({s2_arch} {len(predictor.stage2_models)}-fold)"
                            if len(predictor.stage2_models) > 1
                            else f"Stage2 ({s2_arch})"
                        )
                        ablation_table[primary_label] = primary_row

                        for ckpt_raw in ablation_checkpoints:
                            ckpt_path = resolve_config_path(
                                ckpt_raw,
                                cascade_path,
                                must_exist=True,
                                allow_cwd_fallback=allow_cwd,
                            )
                            ab_model: torch.nn.Module | None = None
                            ckpt_obj: Any = None
                            try:
                                ckpt_obj = torch.load(
                                    ckpt_path,
                                    map_location=predictor.device,
                                    weights_only=False,
                                )
                                ckpt_cfg = copy.deepcopy(
                                    ckpt_obj["config"]
                                    if isinstance(ckpt_obj, dict)
                                    and isinstance(ckpt_obj.get("config"), dict)
                                    else s2_cfg
                                )
                                ckpt_cfg.setdefault("model", {})["pretrained"] = False
                                ab_model = build_model_from_config(ckpt_cfg).to(predictor.device)
                                state_dict = (
                                    ckpt_obj["model_state_dict"]
                                    if isinstance(ckpt_obj, dict)
                                    and "model_state_dict" in ckpt_obj
                                    else ckpt_obj
                                )
                                ab_model.load_state_dict(state_dict)
                                ab_model.eval()

                                cal_file = ckpt_path.parent / "calibration.json"
                                ab_temp = 1.0
                                ab_unc_thresh = predictor.stage2_uncertainty_threshold
                                if cal_file.is_file():
                                    with open(cal_file, "r", encoding="utf-8") as cf:
                                        cal_data = json.load(cf)
                                    if isinstance(cal_data, dict):
                                        if cal_data.get("temperature") is not None:
                                            ab_temp = float(cal_data["temperature"])
                                        if cal_data.get("uncertainty_threshold") is not None:
                                            ab_unc_thresh = float(cal_data["uncertainty_threshold"])

                                ab_probs, ab_targets = _collect_probs(
                                    ab_model,
                                    s2_loader,
                                    predictor.device,
                                    temperature=ab_temp,
                                )
                                ab_s2_metrics = compute_stage2_metrics(
                                    targets=ab_targets,
                                    probs=ab_probs,
                                    class_names=predictor.stage2_classes,
                                    uncertainty_threshold=ab_unc_thresh,
                                )

                                ab_gc_metrics: dict[str, Any] | None = None
                                if s2_masked_loader is not None:
                                    with GradCAM(ab_model) as ab_cam:
                                        ab_gc_metrics = evaluate_gradcam_against_masks(
                                            ab_cam,
                                            s2_masked_loader,
                                            max_samples=max_gc,
                                            target_mode="predicted",
                                        )

                                ab_arch = str(
                                    (ckpt_cfg.get("model") or {}).get(
                                        "arch", ckpt_path.parent.name
                                    )
                                )
                                ab_label = f"Stage2 ({ab_arch})"
                                if ab_label in ablation_table:
                                    ab_label = f"Stage2 ({ab_arch}) [{ckpt_path.parent.name}]"
                                if ab_label in ablation_table:
                                    ab_label = f"{ab_label} #{len(ablation_table)}"

                                ab_row: dict[str, float] = {
                                    "accuracy": float(ab_s2_metrics.accuracy),
                                    "balanced_accuracy": float(ab_s2_metrics.balanced_accuracy),
                                    "macro_f1": float(ab_s2_metrics.macro_f1),
                                    "ece": float(ab_s2_metrics.ece),
                                }
                                if ab_gc_metrics is not None:
                                    ab_row["pointing_game_hit_rate"] = float(
                                        ab_gc_metrics["pointing_game_hit_rate"]
                                    )
                                    ab_row["mean_energy_in_mask"] = float(
                                        ab_gc_metrics["mean_energy_in_mask"]
                                    )
                                ablation_table[ab_label] = ab_row
                            finally:
                                del ab_model, ckpt_obj
                                _release_device_memory()

                        plot_paths["Model Ablation Comparison"] = plot_ablation_comparison(
                            ablation_table,
                            [
                                "accuracy",
                                "macro_f1",
                                "pointing_game_hit_rate",
                                "mean_energy_in_mask",
                            ],
                            plots_dir / "ablation_comparison.png",
                        )

        cascade_metrics = None
        cas_manifest_raw = cascade_cfg.get("cascade_manifest_path")
        if cas_manifest_raw:
            cas_manifest_path = resolve_config_path(
                cas_manifest_raw, cascade_path, allow_cwd_fallback=allow_cwd
            )
            if cas_manifest_path.is_file():
                cas_df = pd.read_csv(cas_manifest_path)
                if not bool(s1_data.get("use_cache", True)) or not bool(s2_data.get("use_cache", True)):
                    cas_df = _drop_cache_columns(cas_df)
                if not cas_df.empty:
                    cas_loader = _make_eval_loader(cas_df, stage="cascade")
                    cas_results: list[CascadeResult] = []
                    for images, _ in cas_loader:
                        cas_results.extend(predictor.predict_batch_tensors(images))

                    true_labels = cas_df["class_name"].astype(str).tolist()
                    pred_labels = [r.predicted_label for r in cas_results]
                    raw_4way = [r.raw_4way_label for r in cas_results]
                    cascade_metrics = compute_cascade_metrics(
                        true_labels=true_labels,
                        pred_labels=pred_labels,
                        raw_4way_preds=raw_4way,
                    )
                    plot_paths["Cascade 4-Way Confusion Matrix"] = plot_confusion_matrix(
                        cascade_metrics.confusion_matrix_4way,
                        cascade_metrics.class_names,
                        plots_dir / "cascade_confusion_matrix_4way.png",
                        title="Cascade 4-Way Confusion Matrix",
                    )
                    plot_paths["Cascade 4x5 Routing Matrix"] = plot_confusion_matrix(
                        cascade_metrics.routing_matrix_4x5,
                        cascade_metrics.routing_columns,
                        plots_dir / "cascade_routing_matrix_4x5.png",
                        title="Cascade 4x5 Clinical Routing Matrix",
                    )

        report_path = generate_evaluation_report(
            output_dir=out_dir,
            stage1_metrics=stage1_metrics,
            stage2_metrics=stage2_metrics,
            cascade_metrics=cascade_metrics,
            gradcam_metrics=gradcam_metrics,
            ablation_table=ablation_table,
            plot_paths=plot_paths,
        )
        print(f"Evaluation report written to {report_path}")
        return report_path
    finally:
        del predictor
        _release_device_memory()


def predict_command(
    config_path: str | Path,
    image_path: str | Path,
    gradcam_out: str | Path | None = None,
    json_output: bool = False,
) -> CascadeResult:
    """Runs single-image cascade inference and optional Grad-CAM overlay export."""
    predictor: CascadePredictor | None = CascadePredictor.from_configs(config_path)
    try:
        assert predictor is not None
        want_gradcam = gradcam_out is not None
        result = predictor.predict_file(
            image_path,
            return_gradcam=want_gradcam,
            gradcam_on_negative=want_gradcam,
        )

        if gradcam_out is not None and result.gradcam_overlay is not None:
            out_path = Path(gradcam_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _save_png(out_path, result.gradcam_overlay)

        if json_output:
            print(json.dumps(result.to_dict(include_arrays=False), indent=2))
        else:
            print(f"Prediction: {result.predicted_label}")
            print(
                f"Stage 1 P(tumor): {result.stage1_tumor_prob:.4f} "
                f"(threshold={result.stage1_threshold:.4f}, positive={result.stage1_positive})"
            )
            if result.stage2_probs is not None:
                probs_fmt = ", ".join(f"{k}={v:.4f}" for k, v in result.stage2_probs.items())
                print(
                    f"Stage 2 confidence: {result.stage2_confidence:.4f} "
                    f"(threshold={result.stage2_uncertainty_threshold:.4f}, T={result.stage2_temperature:.3f})"
                )
                print(f"Stage 2 probs: {probs_fmt}")
            if gradcam_out is not None:
                print(f"Grad-CAM overlay ({result.gradcam_stage}) saved to: {gradcam_out}")

        return result
    finally:
        del predictor
        _release_device_memory()


def ui_command(
    config_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launches the interactive Gradio web application."""
    from mri_tumor.app.web_ui import launch_ui

    launch_ui(
        cascade_config_path=config_path,
        host=host,
        port=int(port),
        share=bool(share),
    )


def build_parser() -> argparse.ArgumentParser:
    """Constructs the ``mri-tumor`` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mri-tumor",
        description="MRITumorDetection-CNN two-stage cascade CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prep = subparsers.add_parser("prepare", help="Extract datasets and build leak-free splits")
    p_prep.add_argument("--config", default="configs/cascade.yaml", help="Path to cascade config")
    p_prep.add_argument("--figshare-zip", default=None, help="Path to Figshare dataset zip")
    p_prep.add_argument("--nickparvar-zip", default=None, help="Path to NickParvar dataset zip")
    p_prep.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable preprocessed PNG caching",
    )
    p_prep.add_argument("--max-hamming", type=int, default=6, help="pHash dedup Hamming threshold")

    p_train = subparsers.add_parser("train", help="Train and calibrate Stage 1 or Stage 2 model")
    p_train.add_argument("--config", required=True, help="Path to stage1.yaml or stage2.yaml")
    p_train.add_argument("--fold", type=int, default=None, help="Override CV fold for Stage 2")
    p_train.add_argument("--arch", type=str, default=None, help="Override model architecture")
    p_train.add_argument("--backbone", type=str, default=None, help="Override model backbone")
    p_train.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    p_train.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p_train.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    p_train.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p_train.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )
    p_train.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights")
    p_train.add_argument("--no-cache", action="store_true", help="Ignore cached PNG manifests")
    p_train.add_argument("--resume", action="store_true", help="Resume from existing best_model.pt")
    p_train.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip training if best_model.pt and calibration.json already exist",
    )

    p_eval = subparsers.add_parser("evaluate", help="Evaluate cascade metrics and Grad-CAM overlap")
    p_eval.add_argument("--config", default="configs/cascade.yaml", help="Path to cascade.yaml")
    p_eval.add_argument("--output-dir", default=None, help="Override evaluation output directory")
    p_eval.add_argument("--gradcam-samples", type=int, default=24, help="Max masked Grad-CAM slices")
    p_eval.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    p_eval.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p_eval.add_argument(
        "--ablation-checkpoints",
        nargs="*",
        default=None,
        help="Optional Stage 2 checkpoint paths for ablation comparison",
    )

    p_pred = subparsers.add_parser("predict", help="Run cascade inference on a single MRI slice")
    p_pred.add_argument("--config", default="configs/cascade.yaml", help="Path to cascade.yaml")
    p_pred.add_argument("--image", required=True, help="Path to input .jpg/.png/.mat slice")
    p_pred.add_argument("--gradcam-out", default=None, help="Output PNG path for Grad-CAM overlay")
    p_pred.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON output")

    p_ui = subparsers.add_parser("ui", help="Launch interactive Gradio web UI")
    p_ui.add_argument("--config", default="configs/cascade.yaml", help="Path to cascade.yaml")
    p_ui.add_argument("--host", default="127.0.0.1", help="Server host address")
    p_ui.add_argument("--port", type=int, default=7860, help="Server port")
    p_ui.add_argument("--share", action="store_true", help="Create public Gradio share link")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point registered as ``mri-tumor``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        prepare_command(
            config_path=args.config,
            figshare_zip=args.figshare_zip,
            nickparvar_zip=args.nickparvar_zip,
            cache=args.cache,
            max_hamming=args.max_hamming,
        )
    elif args.command == "train":
        train_command(
            config_path=args.config,
            fold=args.fold,
            arch=args.arch,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            no_pretrained=args.no_pretrained,
            no_cache=args.no_cache,
            resume=args.resume,
            skip_if_exists=args.skip_if_exists,
        )
    elif args.command == "evaluate":
        evaluate_command(
            config_path=args.config,
            output_dir=args.output_dir,
            gradcam_samples=args.gradcam_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            ablation_checkpoints=args.ablation_checkpoints,
        )
    elif args.command == "predict":
        predict_command(
            config_path=args.config,
            image_path=args.image,
            gradcam_out=args.gradcam_out,
            json_output=args.json_output,
        )
    elif args.command == "ui":
        ui_command(
            config_path=args.config,
            host=args.host,
            port=args.port,
            share=args.share,
        )
    else:
        parser.error(f"Unknown command {args.command!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
