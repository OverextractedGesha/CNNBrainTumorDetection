"""PyTorch Dataset classes for brain MRI slices and tumor masks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from mri_tumor.data.figshare import read_figshare_mat
from mri_tumor.data.preprocess import load_rgb_image, preprocess_slice


def _safe_int(val: Any, default: int) -> int:
    """Safely parses integer values handling None, NaN, and string formats."""
    if val is None or pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _resolve_existing_file(val: Any) -> Path | None:
    """Returns a Path if ``val`` points to an existing regular file, else None."""
    if val is None or pd.isna(val):
        return None
    raw = str(val).strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def resolve_dataloader_kwargs(
    num_workers: int = 2,
    pin_memory: bool = True,
    is_notebook: bool = False,
) -> dict[str, Any]:
    """Resolves Windows- and notebook-safe PyTorch DataLoader keyword arguments.

    On Windows (``os.name == "nt"``), forces ``num_workers=0`` when
    ``is_notebook=True`` to avoid multiprocessing ``spawn`` hangs, and caps
    script workers at 2 to bound RAM overhead.
    """
    effective_workers = max(0, int(num_workers))
    if os.name == "nt":
        effective_workers = 0 if is_notebook else min(effective_workers, 2)

    return {
        "num_workers": effective_workers,
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
        "persistent_workers": bool(effective_workers > 0),
    }


class SliceDataset(Dataset):
    """PyTorch Dataset for 2D MRI slices supporting Figshare (.mat) and Nickparvar (.jpg/.png).

    Args:
        manifest_df: DataFrame with sample metadata and filepaths.
        stage: Pipeline stage - 'stage1' (binary: 0=no_tumor, 1=tumor),
               'stage2' (subtype: 0=glioma, 1=meningioma, 2=pituitary),
               or 'cascade' (evaluation mode).
        image_size: Target slice dimension (default 224 -> 224x224).
        transform: Optional Albumentations Compose pipeline.
        return_mask: If True, returns tumor segmentation mask (float tensor, H, W).
        use_clahe: If True, applies CLAHE contrast enhancement.
        crop_margin_ratio: Bounding box expansion ratio for brain contour cropping.
        return_dict: If True, returns sample dictionary instead of tuple.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        stage: str = "stage1",
        image_size: int = 224,
        transform: A.Compose | None = None,
        return_mask: bool = False,
        use_clahe: bool = False,
        crop_margin_ratio: float = 0.05,
        return_dict: bool = False,
    ):
        super().__init__()
        self.df = manifest_df.reset_index(drop=True)
        self.stage = stage.lower()
        self.image_size = image_size
        self.transform = transform
        self.return_mask = return_mask
        self.use_clahe = use_clahe
        self.crop_margin_ratio = crop_margin_ratio
        self.return_dict = return_dict

    def __len__(self) -> int:
        return len(self.df)

    def get_metadata(self, idx: int) -> dict[str, Any]:
        """Returns metadata row dictionary for sample idx."""
        return self.df.iloc[idx].to_dict()

    def _resolve_label(
        self,
        row: pd.Series,
        filepath: Path,
        mat_class_idx: int | None = None,
    ) -> int:
        is_mat = filepath.suffix.lower() == ".mat"
        if is_mat:
            if self.stage == "stage1":
                return _safe_int(row.get("binary_label"), 1)
            col = "subtype_label" if self.stage == "cascade" else "class_idx"
            val = row.get(col)
            if val is not None and not pd.isna(val):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            if mat_class_idx is not None:
                return mat_class_idx
            if filepath.is_file():
                return read_figshare_mat(filepath).class_idx
            return -1

        if self.stage == "stage1":
            return _safe_int(row.get("binary_label"), 0)
        if self.stage == "stage2":
            return _safe_int(row.get("class_idx"), -1)
        if self.stage == "cascade":
            return _safe_int(row.get("subtype_label"), -1)
        return _safe_int(row.get("binary_label"), 0)

    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]
        filepath = Path(row["filepath"])
        sample_id = str(row.get("sample_id", filepath.stem))
        patient_id = str(row.get("patient_id", sample_id))

        cached_img_path = _resolve_existing_file(row.get("cached_path"))
        is_mat = filepath.suffix.lower() == ".mat"
        cached_mask_path = (
            _resolve_existing_file(row.get("cached_mask_path"))
            if (self.return_mask and is_mat)
            else None
        )
        use_cached_fast_path = cached_img_path is not None and not (
            self.return_mask and is_mat and cached_mask_path is None and filepath.is_file()
        )
        if use_cached_fast_path:
            assert cached_img_path is not None
            # Fast path: load pre-cropped cached PNG directly and skip Otsu preprocessing
            img_rgb = load_rgb_image(cached_img_path)
            if img_rgb.shape[:2] != (self.image_size, self.image_size):
                img_rgb = cv2.resize(
                    img_rgb,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            img_3ch = img_rgb.astype(np.float32) / 255.0

            mask_2d: np.ndarray | None = None
            if self.return_mask and cached_mask_path is not None:
                mask_raw = cv2.imread(str(cached_mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_raw is None:
                    with Image.open(cached_mask_path) as pil_mask:
                        mask_raw = np.array(pil_mask.convert("L"))
                if mask_raw.shape[:2] != (self.image_size, self.image_size):
                    mask_raw = cv2.resize(
                        mask_raw,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_2d = (mask_raw > 127).astype(np.float32)

            label = self._resolve_label(row, filepath, mat_class_idx=None)
        else:
            # 1. Load image and mask based on file type
            mat_class_idx: int | None = None
            if filepath.suffix.lower() == ".mat":
                sample = read_figshare_mat(filepath)
                raw_img = sample.image
                raw_mask = sample.tumor_mask
                mat_class_idx = sample.class_idx
            else:
                # Standard image file (jpg / png)
                raw_img = load_rgb_image(filepath)
                raw_mask = None

            label = self._resolve_label(row, filepath, mat_class_idx=mat_class_idx)

            # 2. Preprocess slice: crop -> normalize -> CLAHE -> resize -> 3-channel
            img_3ch, mask_2d = preprocess_slice(
                image=raw_img,
                mask=raw_mask,
                target_size=(self.image_size, self.image_size),
                crop_margin_ratio=self.crop_margin_ratio,
                use_clahe=self.use_clahe,
            )

        # 3. Apply albumentations transforms if provided
        if self.transform is not None:
            if mask_2d is not None:
                augmented = self.transform(image=img_3ch, mask=mask_2d)
                img_3ch = augmented["image"]
                mask_2d = augmented["mask"]
            else:
                augmented = self.transform(image=img_3ch)
                img_3ch = augmented["image"]

        # 4. Convert to PyTorch tensors
        # img_3ch: (H, W, 3) float32 -> (3, H, W) float32
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_3ch).transpose(2, 0, 1)).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_mask:
            if mask_2d is not None:
                mask_tensor = (torch.from_numpy(np.ascontiguousarray(mask_2d)).float() > 0.5).float()
            else:
                mask_tensor = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)

        if self.return_dict:
            item = {
                "image": img_tensor,
                "label": label_tensor,
                "sample_id": sample_id,
                "patient_id": patient_id,
                "filepath": str(filepath),
            }
            if self.return_mask:
                item["mask"] = mask_tensor
            return item

        if self.return_mask:
            return img_tensor, label_tensor, mask_tensor
        return img_tensor, label_tensor
