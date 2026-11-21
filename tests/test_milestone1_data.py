"""Unit test suite for Milestone 1 data pipeline modules."""

from __future__ import annotations

from pathlib import Path
import cv2
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.io
import torch

from mri_tumor.data.datasets import SliceDataset, resolve_dataloader_kwargs
from mri_tumor.data.dedup import (
    BKTree,
    compute_phash,
    deduplicate_and_filter_leakage,
    hamming_distance,
)
from mri_tumor.data.figshare import (
    FIGSHARE_LABEL_MAP,
    STAGE2_CLASS_TO_IDX,
    build_figshare_manifest,
    extract_figshare_archive,
    read_figshare_mat,
)
from mri_tumor.data.nickparvar import (
    build_nickparvar_manifest,
    extract_nickparvar_archive,
)
from mri_tumor.data.preprocess import (
    apply_clahe,
    cache_preprocessed_manifest,
    crop_brain_contour,
    normalize_intensity,
    preprocess_slice,
    to_3channel,
)
from mri_tumor.data.splits import (
    build_cascade_test_manifest,
    create_stage1_splits,
    create_stage2_splits,
    verify_no_patient_leakage,
)
from mri_tumor.data.transforms import get_eval_transforms, get_train_transforms


def _create_synthetic_brain_slice(
    height: int = 256,
    width: int = 256,
    brain_radius_y: int = 80,
    brain_radius_x: int = 70,
    tumor_radius: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a synthetic brain MRI slice with black background and bright brain ellipse."""
    image = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    center = (width // 2, height // 2)

    # Draw brain ellipse (intensity ~150-200)
    cv2.ellipse(image, center, (brain_radius_x, brain_radius_y), 0, 0, 360, 180.0, -1)
    # Add internal texture
    noise = np.random.RandomState(42).normal(0, 10, (height, width)).astype(np.float32)
    brain_mask = (image > 0)
    image[brain_mask] = np.clip(image[brain_mask] + noise[brain_mask], 50.0, 255.0)

    # Draw tumor region (higher intensity ~240)
    tumor_center = (center[0] - 20, center[1] - 15)
    cv2.circle(image, tumor_center, tumor_radius, 240.0, -1)
    cv2.circle(mask, tumor_center, tumor_radius, 1, -1)

    return image, mask


def _write_hdf5_mat(
    file_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    label: int = 2,
    pid: str = "PAT_001",
) -> Path:
    """Writes a MATLAB v7.3 HDF5 .mat file compatible with Figshare format."""
    with h5py.File(file_path, "w") as f:
        cj = f.create_group("cjdata")
        # In MATLAB v7.3 HDF5, 2D arrays are transposed
        cj.create_dataset("image", data=image.T)
        cj.create_dataset("tumorMask", data=mask.T.astype(np.float64))
        cj.create_dataset("label", data=np.array([[float(label)]]))
        pid_ascii = np.array([ord(c) for c in pid], dtype=np.uint16)
        cj.create_dataset("PID", data=pid_ascii)
        cj.create_dataset("tumorBorder", data=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64))
    return file_path


def test_crop_brain_contour_removes_black_border_and_crops_mask():
    """Verifies black border removal and synchronized mask cropping."""
    image, mask = _create_synthetic_brain_slice(height=256, width=256)

    cropped_img, cropped_mask, bounds = crop_brain_contour(image, mask=mask, margin_ratio=0.05)

    x1, y1, x2, y2 = bounds
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 256 and y2 <= 256
    assert cropped_img.shape[0] < 256, "Height should be cropped"
    assert cropped_img.shape[1] < 256, "Width should be cropped"
    assert cropped_mask is not None
    assert cropped_mask.shape == cropped_img.shape

    # Ensure tumor is still present inside cropped mask
    assert cropped_mask.sum() > 0
    assert cropped_mask.sum() == mask.sum()


def test_normalize_intensity_percentile_and_constant_image():
    """Verifies [0, 1] bounds, outlier clipping, and zero-division safety on flat images."""
    # Test flat image (zero division safety)
    flat_img = np.ones((64, 64), dtype=np.float32) * 42.0
    norm_flat = normalize_intensity(flat_img)
    assert not np.isnan(norm_flat).any()
    assert (norm_flat == 0.0).all()

    # Test outlier clipping
    img = np.linspace(0, 100, 10000, dtype=np.float32).reshape((100, 100))
    # Inject extreme outliers
    img[0, 0] = -99999.0
    img[-1, -1] = 99999.0

    norm_img = normalize_intensity(img, low_pct=1.0, high_pct=99.0)
    assert norm_img.min() >= 0.0
    assert norm_img.max() <= 1.0
    assert not np.isnan(norm_img).any()


def test_to_3channel_and_clahe():
    """Tests 3-channel expansion and CLAHE enhancement."""
    gray = np.random.uniform(0.0, 1.0, (64, 64)).astype(np.float32)
    ch3 = to_3channel(gray)
    assert ch3.shape == (64, 64, 3)
    assert np.allclose(ch3[:, :, 0], ch3[:, :, 1])

    clahe_img = apply_clahe(gray)
    assert clahe_img.shape == (64, 64)
    assert clahe_img.min() >= 0.0 and clahe_img.max() <= 1.0


def test_read_figshare_mat_and_manifest(tmp_path: Path):
    """Writes synthetic .mat v7.3 files and tests read_figshare_mat and build_figshare_manifest."""
    img, mask = _create_synthetic_brain_slice()

    # Create synthetic samples for each class
    _write_hdf5_mat(tmp_path / "1.mat", img, mask, label=1, pid="PAT_MEN")
    _write_hdf5_mat(tmp_path / "2.mat", img, mask, label=2, pid="PAT_GLI")
    _write_hdf5_mat(tmp_path / "3.mat", img, mask, label=3, pid="PAT_PIT")
    # Auxiliary file that should be ignored
    (tmp_path / "cvind.mat").write_bytes(b"dummy")

    s1 = read_figshare_mat(tmp_path / "1.mat")
    assert s1.patient_id == "PAT_MEN"
    assert s1.class_name == "meningioma"
    assert s1.class_idx == STAGE2_CLASS_TO_IDX["meningioma"]
    assert s1.image.shape == (256, 256)
    assert s1.tumor_mask.shape == (256, 256)

    s2 = read_figshare_mat(tmp_path / "2.mat")
    assert s2.patient_id == "PAT_GLI"
    assert s2.class_name == "glioma"
    assert s2.class_idx == STAGE2_CLASS_TO_IDX["glioma"]

    s3 = read_figshare_mat(tmp_path / "3.mat")
    assert s3.patient_id == "PAT_PIT"
    assert s3.class_name == "pituitary"
    assert s3.class_idx == STAGE2_CLASS_TO_IDX["pituitary"]

    manifest_csv = tmp_path / "manifest.csv"
    df = build_figshare_manifest(tmp_path, output_csv=manifest_csv, compute_hashes=True)

    assert len(df) == 3
    assert set(df["class_name"]) == {"meningioma", "glioma", "pituitary"}
    assert "phash" in df.columns
    assert manifest_csv.exists()


def test_phash_invariance_and_bktree_dedup(tmp_path: Path):
    """Tests perceptual hash invariance to slight distortions and BKTree deduplication."""
    img, _ = _create_synthetic_brain_slice()

    # Slightly perturbed copy (re-encoded as JPG or slight noise)
    noise = np.random.RandomState(123).normal(0, 1.5, img.shape).astype(np.float32)
    perturbed_img = np.clip(img + noise, 0, 255)

    # Completely different image (e.g. inverted checkerboard)
    diff_img = np.zeros_like(img)
    diff_img[::8, :] = 255.0

    h_orig = compute_phash(img, crop_first=False)
    h_pert = compute_phash(perturbed_img, crop_first=False)
    h_diff = compute_phash(diff_img, crop_first=False)

    dist_near = hamming_distance(h_orig, h_pert)
    dist_far = hamming_distance(h_orig, h_diff)

    assert dist_near <= 6, f"Expected small hamming dist, got {dist_near}"
    assert dist_far > 10, f"Expected large hamming dist for different image, got {dist_far}"

    # BKTree tests
    tree = BKTree()
    tree.add("orig", h_orig)
    matches = tree.search(h_pert, max_dist=6)
    assert len(matches) == 1
    assert matches[0][0] == "orig"

    # Leakage and deduplication test
    figshare_holdout = [h_orig]
    nickparvar_df = pd.DataFrame([
        {"sample_id": "np_leak", "phash": h_pert},       # Matches holdout (dist <= 6)
        {"sample_id": "np_unique_1", "phash": h_diff},   # Clean independent sample
        {"sample_id": "np_dup", "phash": h_diff},        # Exact duplicate of np_unique_1
    ])

    clean_df, removed_df = deduplicate_and_filter_leakage(
        nickparvar_df, figshare_holdout_phashes=figshare_holdout, max_hamming_dist=6
    )

    assert len(clean_df) == 1
    assert clean_df.iloc[0]["sample_id"] == "np_unique_1"

    assert len(removed_df) == 2
    leakage_row = removed_df[removed_df["sample_id"] == "np_leak"].iloc[0]
    assert leakage_row["drop_reason"] == "figshare_holdout_leakage"
    dup_row = removed_df[removed_df["sample_id"] == "np_dup"].iloc[0]
    assert dup_row["drop_reason"] == "internal_near_duplicate"


def test_stage2_patient_grouped_splits_no_leakage():
    """Creates synthetic patient-grouped DataFrame and verifies 0% patient leakage across folds."""
    records = []
    # 30 patients, 5 slices each = 150 rows
    for p_idx in range(30):
        pid = f"PAT_{p_idx:03d}"
        class_idx = p_idx % 3  # balanced classes: 0, 1, 2
        for s_idx in range(5):
            records.append({
                "sample_id": f"{pid}_{s_idx}",
                "patient_id": pid,
                "class_idx": class_idx,
                "class_name": FIGSHARE_LABEL_MAP[class_idx + 1 if (class_idx + 1) in FIGSHARE_LABEL_MAP else 1],
                "phash": p_idx * 1000 + s_idx,
            })

    df = pd.DataFrame(records)

    split_df = create_stage2_splits(df, n_folds=5, holdout_fraction=0.15, seed=42)

    # Must pass strict verification
    assert verify_no_patient_leakage(split_df) is True

    # Verify holdout patients do not exist in training or val
    holdout_pts = set(split_df[split_df["is_holdout"]]["patient_id"])
    cv_pts = set(split_df[~split_df["is_holdout"]]["patient_id"])
    assert len(holdout_pts.intersection(cv_pts)) == 0

    # Test that artificial leakage raises ValueError
    corrupted_df = split_df.copy()
    corrupted_df.loc[corrupted_df["is_holdout"], "patient_id"] = "PAT_000"
    corrupted_df.loc[~corrupted_df["is_holdout"], "patient_id"] = "PAT_000"
    with pytest.raises(ValueError, match="leakage"):
        verify_no_patient_leakage(corrupted_df)


def test_stage1_and_cascade_splits_end_to_end(tmp_path: Path):
    """Tests create_stage1_splits and build_cascade_test_manifest end-to-end."""
    # Synthetic Stage 2 split df
    s2_df = pd.DataFrame([
        {
            "sample_id": "fig_001",
            "filepath": str(tmp_path / "fig_001.mat"),
            "patient_id": "P01",
            "class_idx": 0,
            "class_name": "glioma",
            "is_holdout": True,
            "split": "test",
            "phash": 123456789,
        },
        {
            "sample_id": "fig_002",
            "filepath": str(tmp_path / "fig_002.mat"),
            "patient_id": "P02",
            "class_idx": 1,
            "class_name": "meningioma",
            "is_holdout": False,
            "split": "train",
            "phash": 987654321,
        },
    ])

    # Synthetic Nickparvar df (4 classes, balanced)
    np_records = []
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    for i in range(40):
        c = classes[i % 4]
        np_records.append({
            "sample_id": f"np_{i}",
            "filepath": str(tmp_path / f"np_{i}.jpg"),
            "sub_class": c,
            "binary_label": 0 if c == "notumor" else 1,
            "binary_class_name": "no_tumor" if c == "notumor" else "tumor",
            "phash": (i + 1) * 1000000,
        })
    np_df = pd.DataFrame(np_records)

    clean_s1, removed_s1 = create_stage1_splits(
        np_df, s2_df, val_fraction=0.15, test_fraction=0.15, max_hamming_dist=6, seed=42
    )

    assert "split" in clean_s1.columns
    assert set(clean_s1["split"].unique()) == {"train", "val", "test"}

    # Cascade manifest
    cascade_df = build_cascade_test_manifest(clean_s1, s2_df)
    assert "source_dataset" in cascade_df.columns
    assert "subtype_label" in cascade_df.columns

    # Verify Figshare tumor holdout has binary_label=1 and subtype_label in {0, 1, 2}
    fig_cascade = cascade_df[cascade_df["source_dataset"] == "figshare"]
    assert (fig_cascade["binary_label"] == 1).all()
    assert (fig_cascade["subtype_label"] >= 0).all()

    # Verify NickParvar control has binary_label=0 and subtype_label=-1
    np_cascade = cascade_df[cascade_df["source_dataset"] == "nickparvar"]
    assert (np_cascade["binary_label"] == 0).all()
    assert (np_cascade["subtype_label"] == -1).all()


def test_slice_dataset_shapes_and_masks(tmp_path: Path):
    """Verifies SliceDataset returns (3, 224, 224) float32 tensors and binary masks."""
    img, mask = _create_synthetic_brain_slice(height=128, width=128)
    mat_file = _write_hdf5_mat(tmp_path / "sample.mat", img, mask, label=2, pid="P1")

    # Also write a standard jpg image
    jpg_file = tmp_path / "sample.jpg"
    cv2.imwrite(str(jpg_file), img.astype(np.uint8))

    df = pd.DataFrame([
        {
            "sample_id": "mat_1",
            "filepath": str(mat_file),
            "patient_id": "P1",
            "class_idx": 0,
            "binary_label": 1,
        },
        {
            "sample_id": "jpg_1",
            "filepath": str(jpg_file),
            "patient_id": "P2",
            "class_idx": -1,
            "binary_label": 0,
        },
    ])

    # Test tuple return with mask
    ds = SliceDataset(df, stage="stage1", image_size=224, return_mask=True)
    assert len(ds) == 2

    img_t, lbl_t, mask_t = ds[0]
    assert img_t.shape == (3, 224, 224)
    assert img_t.dtype == torch.float32
    assert img_t.min() >= 0.0 and img_t.max() <= 1.0
    assert lbl_t.item() == 1
    assert mask_t.shape == (224, 224)
    assert mask_t.dtype == torch.float32

    # Test second sample (jpg without mask)
    img_t2, lbl_t2, mask_t2 = ds[1]
    assert img_t2.shape == (3, 224, 224)
    assert lbl_t2.item() == 0
    assert mask_t2.shape == (224, 224)

    # Test dict return
    ds_dict = SliceDataset(df, stage="stage2", image_size=224, return_dict=True)
    item0 = ds_dict[0]
    assert isinstance(item0, dict)
    assert "image" in item0 and "label" in item0 and "sample_id" in item0
    assert item0["image"].shape == (3, 224, 224)


def test_transforms_train_and_eval():
    """Verifies train and eval augmentation pipelines on 3-channel images and masks."""
    img = np.random.uniform(0.0, 1.0, (256, 256, 3)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:150, 100:150] = 1

    train_tf = get_train_transforms(image_size=224)
    res_train = train_tf(image=img, mask=mask)
    assert res_train["image"].shape == (224, 224, 3)
    assert res_train["mask"].shape == (224, 224)

    eval_tf = get_eval_transforms(image_size=224)
    res_eval = eval_tf(image=img, mask=mask)
    assert res_eval["image"].shape == (224, 224, 3)
    assert res_eval["mask"].shape == (224, 224)


def test_extract_archives_and_build_nickparvar_manifest(tmp_path: Path):
    """Tests archive extraction and NickParvar manifest indexing."""
    import zipfile

    # Create dummy NickParvar zip structure
    np_zip = tmp_path / "nickparvar.zip"
    img_data = np.zeros((64, 64, 3), dtype=np.uint8)
    _, encoded_jpg = cv2.imencode(".jpg", img_data)

    with zipfile.ZipFile(np_zip, "w") as zf:
        zf.writestr("Training/glioma_tumor/tr_glioma_1.jpg", encoded_jpg.tobytes())
        zf.writestr("Testing/no_tumor/te_notumor_1.jpg", encoded_jpg.tobytes())

    np_extracted = tmp_path / "np_extracted"
    extract_nickparvar_archive(np_zip, np_extracted)
    assert (np_extracted / "Training" / "glioma_tumor" / "tr_glioma_1.jpg").exists()

    np_manifest = build_nickparvar_manifest(np_extracted, output_csv=tmp_path / "np_manifest.csv")
    assert len(np_manifest) == 2
    assert "binary_label" in np_manifest.columns
    assert set(np_manifest["sub_class"]) == {"glioma", "notumor"}
    assert (tmp_path / "np_manifest.csv").exists()

    # Create dummy Figshare nested zip
    inner_zip_bytes = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_zip_bytes, "w") as izf:
        izf.writestr("test_slice.mat", b"dummy_mat_bytes")

    outer_zip = tmp_path / "figshare_outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as ozf:
        ozf.write(inner_zip_bytes, arcname="brainTumorDataPublic_1.zip")

    fig_extracted = tmp_path / "fig_extracted"
    extract_figshare_archive(outer_zip, fig_extracted)
    assert (fig_extracted / "test_slice.mat").exists()


def test_read_figshare_mat_v5_fallback(tmp_path: Path):
    """Verifies fallback to scipy.io.loadmat for MATLAB v5 .mat files."""
    img, mask = _create_synthetic_brain_slice()
    v5_file = tmp_path / "legacy_v5.mat"
    mat_dict = {
        "cjdata": {
            "label": np.array([[2]]),
            "PID": "PAT_V5_001",
            "image": img,
            "tumorMask": mask,
            "tumorBorder": np.array([10.0, 20.0]),
        }
    }
    scipy.io.savemat(str(v5_file), mat_dict)

    sample = read_figshare_mat(v5_file)
    assert sample.patient_id == "PAT_V5_001"
    assert sample.class_name == "glioma"
    assert sample.class_idx == STAGE2_CLASS_TO_IDX["glioma"]
    assert sample.image.shape == (256, 256)
    assert sample.tumor_mask.shape == (256, 256)


def test_hamming_distance_and_bktree_string_types():
    """Verifies that hamming_distance and BKTree handle hex and decimal strings identically to ints."""
    h_int = 0b101010
    h_dec_str = str(h_int)
    h_hex_str = hex(h_int)
    h_u64 = np.uint64(h_int)

    # All should yield distance 0 against each other
    assert hamming_distance(h_int, h_dec_str) == 0
    assert hamming_distance(h_int, h_hex_str) == 0
    assert hamming_distance(h_int, h_u64) == 0

    tree = BKTree()
    tree.add("item_dec", h_dec_str)
    tree.add("item_hex", h_hex_str)
    tree.add("item_u64", h_u64)

    results = tree.search(h_int, max_dist=0)
    assert len(results) == 3


def test_splits_validation_errors():
    """Verifies that invalid split fractions trigger ValueError guards."""
    dummy_df = pd.DataFrame([{"patient_id": "P1", "class_idx": 0, "phash": 1}])
    with pytest.raises(ValueError, match="holdout_fraction"):
        create_stage2_splits(dummy_df, holdout_fraction=1.5)

    with pytest.raises(ValueError, match="n_folds"):
        create_stage2_splits(dummy_df, n_folds=1)

    with pytest.raises(ValueError, match="test_fraction"):
        create_stage1_splits(dummy_df, dummy_df, test_fraction=1.2)

    with pytest.raises(ValueError, match="Sum of val_fraction"):
        create_stage1_splits(dummy_df, dummy_df, val_fraction=0.6, test_fraction=0.5)


def test_preprocess_empty_and_edge_arrays():
    """Verifies zero-size / flat array safety in preprocessing."""
    empty_img = np.zeros((0, 0), dtype=np.float32)
    cropped, mask, bounds = crop_brain_contour(empty_img)
    assert bounds == (0, 0, 0, 0)

    norm_empty = normalize_intensity(empty_img)
    assert norm_empty.size == 0


def test_slice_dataset_nan_safe(tmp_path: Path):
    """Verifies that SliceDataset gracefully handles NaNs in optional metadata columns."""
    img, _ = _create_synthetic_brain_slice(height=64, width=64)
    jpg_file = tmp_path / "nan_test.jpg"
    cv2.imwrite(str(jpg_file), img.astype(np.uint8))

    df = pd.DataFrame([
        {
            "sample_id": "nan_sample",
            "filepath": str(jpg_file),
            "binary_label": np.nan,
            "class_idx": None,
        }
    ])

    ds = SliceDataset(df, stage="stage1", image_size=64)
    img_t, lbl_t = ds[0]
    assert lbl_t.item() == 0  # default for stage1 non-mat is 0


def test_slice_cache_and_dataloader_kwargs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verifies PNG slice/mask caching, resume skip, SliceDataset parity, and Windows DataLoader kwargs."""
    img, mask = _create_synthetic_brain_slice(height=160, width=160)
    mat_file = _write_hdf5_mat(tmp_path / "glioma_1.mat", img, mask, label=2, pid="P10")
    jpg_file = tmp_path / "ctrl_1.jpg"
    cv2.imwrite(str(jpg_file), img.astype(np.uint8))

    df = pd.DataFrame([
        {
            "sample_id": "mat/10",
            "filepath": str(mat_file),
            "patient_id": "P10",
            "class_idx": 0,
            "binary_label": 1,
        },
        {
            "sample_id": "jpg_10",
            "filepath": str(jpg_file),
            "patient_id": "P20",
            "class_idx": -1,
            "binary_label": 0,
        },
    ])

    cache_dir = tmp_path / "cache"
    cached_df = cache_preprocessed_manifest(
        df,
        cache_dir=cache_dir,
        image_size=128,
        crop_margin_ratio=0.05,
        use_clahe=False,
        overwrite=False,
    )

    assert "cached_path" in cached_df.columns
    assert "cached_mask_path" in cached_df.columns
    assert Path(cached_df.loc[0, "cached_path"]).is_file()
    assert Path(cached_df.loc[0, "cached_mask_path"]).is_file()
    assert Path(cached_df.loc[1, "cached_path"]).is_file()
    assert cached_df.loc[1, "cached_mask_path"] == ""

    # Compare raw vs cached Stage 2 (.mat with mask)
    ds_raw_s2 = SliceDataset(df.iloc[[0]], stage="stage2", image_size=128, return_mask=True)
    ds_cache_s2 = SliceDataset(cached_df.iloc[[0]], stage="stage2", image_size=128, return_mask=True)
    raw_img_s2, raw_lbl_s2, raw_mask_s2 = ds_raw_s2[0]
    cache_img_s2, cache_lbl_s2, cache_mask_s2 = ds_cache_s2[0]
    assert cache_lbl_s2.item() == raw_lbl_s2.item() == 0
    assert torch.allclose(cache_img_s2, raw_img_s2, atol=1.0 / 255.0 + 1e-5)
    assert torch.equal(cache_mask_s2, raw_mask_s2)
    assert cache_mask_s2.sum() > 0

    # Compare raw vs cached Stage 1 (.jpg without mask)
    ds_raw_s1 = SliceDataset(df.iloc[[1]], stage="stage1", image_size=128, return_mask=True)
    ds_cache_s1 = SliceDataset(cached_df.iloc[[1]], stage="stage1", image_size=128, return_mask=True)
    raw_img_s1, raw_lbl_s1, raw_mask_s1 = ds_raw_s1[0]
    cache_img_s1, cache_lbl_s1, cache_mask_s1 = ds_cache_s1[0]
    assert cache_lbl_s1.item() == raw_lbl_s1.item() == 0
    assert torch.allclose(cache_img_s1, raw_img_s1, atol=1.0 / 255.0 + 1e-5)
    assert torch.equal(cache_mask_s1, raw_mask_s1)

    # Missing cached_path falls back seamlessly to raw file preprocessing
    fallback_df = cached_df.iloc[[0]].copy()
    fallback_df["cached_path"] = str(tmp_path / "does_not_exist.png")
    fb_img, fb_lbl, fb_mask = SliceDataset(fallback_df, stage="stage2", image_size=128, return_mask=True)[0]
    assert fb_lbl.item() == 0
    assert torch.allclose(fb_img, raw_img_s2)
    assert torch.equal(fb_mask, raw_mask_s2)

    # Missing cached_mask_path on .mat with return_mask=True falls back instead of zeroing mask
    missing_mask_df = cached_df.iloc[[0]].copy()
    missing_mask_df["cached_mask_path"] = str(tmp_path / "missing_mask.png")
    mm_img, mm_lbl, mm_mask = SliceDataset(
        missing_mask_df, stage="stage2", image_size=128, return_mask=True
    )[0]
    assert mm_lbl.item() == 0
    assert torch.allclose(mm_img, raw_img_s2)
    assert torch.equal(mm_mask, raw_mask_s2)
    assert mm_mask.sum() > 0

    # Confirm cached path truly skips read_figshare_mat and preprocess_slice
    def _fail(*args, **kwargs):
        raise AssertionError("Cached SliceDataset read should bypass raw loading and preprocessing")

    with monkeypatch.context() as m:
        m.setattr("mri_tumor.data.datasets.read_figshare_mat", _fail)
        m.setattr("mri_tumor.data.datasets.preprocess_slice", _fail)
        fast_img, fast_lbl, fast_mask = ds_cache_s2[0]
        assert fast_lbl.item() == 0
        assert torch.allclose(fast_img, cache_img_s2)
        assert torch.equal(fast_mask, cache_mask_s2)

    # Verify overwrite=False skips recomputation, while overwrite=True recomputes
    import mri_tumor.data.preprocess as preprocess_mod

    call_count = 0
    orig_preprocess = preprocess_mod.preprocess_slice

    def _spy_preprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_preprocess(*args, **kwargs)

    monkeypatch.setattr(preprocess_mod, "preprocess_slice", _spy_preprocess)
    cache_preprocessed_manifest(df, cache_dir=cache_dir, image_size=128, overwrite=False)
    assert call_count == 0
    cache_preprocessed_manifest(df, cache_dir=cache_dir, image_size=128, overwrite=True)
    assert call_count == 2

    # Empty manifest preserves columns cleanly
    empty_cached = cache_preprocessed_manifest(df.iloc[0:0], cache_dir=cache_dir, image_size=128)
    assert empty_cached.empty
    assert {"cached_path", "cached_mask_path"}.issubset(empty_cached.columns)

    # Windows notebook/script DataLoader kwargs safety
    monkeypatch.setattr("mri_tumor.data.datasets.os.name", "nt")
    nb_kw = resolve_dataloader_kwargs(num_workers=4, pin_memory=True, is_notebook=True)
    assert nb_kw["num_workers"] == 0
    assert nb_kw["persistent_workers"] is False
    assert nb_kw["pin_memory"] is bool(torch.cuda.is_available())

    win_kw = resolve_dataloader_kwargs(num_workers=8, pin_memory=False, is_notebook=False)
    assert win_kw["num_workers"] == 2
    assert win_kw["persistent_workers"] is True
    assert win_kw["pin_memory"] is False

    monkeypatch.setattr("mri_tumor.data.datasets.os.name", "posix")
    posix_kw = resolve_dataloader_kwargs(num_workers=4, pin_memory=True, is_notebook=True)
    assert posix_kw["num_workers"] == 4
    assert posix_kw["persistent_workers"] is True


