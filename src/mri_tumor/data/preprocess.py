"""Preprocessing pipeline for brain MRI slices and masks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def crop_brain_contour(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    margin_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]:
    """Crops background border around brain tissue using Otsu thresholding and contour detection.

    Args:
        image: 2D or 3D image array (grayscale or RGB, uint8/uint16/float).
        mask: Optional binary segmentation mask corresponding to image.
        margin_ratio: Fractional bounding box expansion margin (default 0.05 = 5%).

    Returns:
        tuple of (cropped_image, cropped_mask_or_None, (x1, y1, x2, y2)).
    """
    if image.ndim == 2:
        h, w = image.shape
        img_2d = image.copy()
    elif image.ndim == 3:
        h, w = image.shape[:2]
        if image.shape[2] == 1:
            img_2d = image[:, :, 0]
        else:
            img_2d = np.mean(image, axis=2)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

    # Fallback bounds: full image
    full_bounds = (0, 0, w, h)

    if h == 0 or w == 0:
        cropped_mask = mask.copy() if mask is not None else None
        return image.copy(), cropped_mask, full_bounds

    # Normalize to uint8 [0, 255] for OpenCV thresholding
    img_min = float(img_2d.min())
    img_max = float(img_2d.max())
    if img_max <= img_min:
        cropped_mask = mask.copy() if mask is not None else None
        return image.copy(), cropped_mask, full_bounds

    u8 = np.clip(
        ((img_2d.astype(np.float32) - img_min) / (img_max - img_min + 1e-8)) * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # Blur + Otsu thresholding
    blurred = cv2.GaussianBlur(u8, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological opening and closing to clean noise specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cropped_mask = mask.copy() if mask is not None else None
        return image.copy(), cropped_mask, full_bounds

    # Pick largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = float(h * w)

    # If largest contour is less than 1% of total image area, fallback to full image
    if contour_area < 0.01 * total_area:
        cropped_mask = mask.copy() if mask is not None else None
        return image.copy(), cropped_mask, full_bounds

    # Bounding box with margin expansion
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    dx = int(round(bw * margin_ratio))
    dy = int(round(bh * margin_ratio))

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w, x + bw + dx)
    y2 = min(h, y + bh + dy)

    # Safety check on coordinates
    if x2 <= x1 or y2 <= y1:
        cropped_mask = mask.copy() if mask is not None else None
        return image.copy(), cropped_mask, full_bounds

    cropped_image = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2].copy() if mask is not None else None

    return cropped_image, cropped_mask, (x1, y1, x2, y2)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Preserves input dtype/scale (e.g. float32 [0, 1] returns float32 [0, 1]).
    """
    is_float = np.issubdtype(image.dtype, np.floating)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        if is_float:
            u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            eq = clahe.apply(u8)
            return (eq.astype(np.float32) / 255.0)
        else:
            u8 = np.clip(image, 0, 255).astype(np.uint8)
            return clahe.apply(u8)
    elif image.ndim == 3:
        channels = [
            apply_clahe(image[:, :, c], clip_limit=clip_limit, tile_grid_size=tile_grid_size)
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")


def normalize_intensity(
    image: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    """Robust min-max percentile intensity normalization to [0.0, 1.0].

    Safely handles flat / constant images by returning zeros.
    """
    if image.size == 0:
        return image.copy().astype(np.float32)

    img_f32 = image.astype(np.float32)
    p_low, p_high = np.percentile(img_f32, [low_pct, high_pct])

    if p_high <= p_low:
        return np.zeros_like(img_f32, dtype=np.float32)

    clipped = np.clip(img_f32, p_low, p_high)
    norm = (clipped - p_low) / (p_high - p_low)
    return norm.astype(np.float32)


def to_3channel(image: np.ndarray) -> np.ndarray:
    """Expands 2D (H, W) or single-channel (H, W, 1) image to 3-channel (H, W, 3)."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        elif image.shape[2] == 3:
            return image
        else:
            raise ValueError(f"Expected 1 or 3 channels, got shape {image.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")


def preprocess_slice(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    target_size: tuple[int, int] = (224, 224),
    crop_margin_ratio: float = 0.05,
    use_clahe: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Preprocesses a single MRI slice and optional tumor mask.

    Steps:
      1. crop_brain_contour (synchronized crop of image and mask)
      2. normalize_intensity to [0, 1] float32
      3. optional CLAHE enhancement
      4. resize image (INTER_LINEAR) and mask (INTER_NEAREST) to target_size
      5. expand image to 3-channel float32 (H, W, 3)

    Returns:
      (image_3ch_f32, resized_mask_or_None)
    """
    cropped_img, cropped_mask, _ = crop_brain_contour(
        image, mask=mask, margin_ratio=crop_margin_ratio
    )

    norm_img = normalize_intensity(cropped_img)
    if use_clahe:
        norm_img = apply_clahe(norm_img)

    resized_img = cv2.resize(norm_img, target_size, interpolation=cv2.INTER_LINEAR)

    if cropped_mask is not None:
        resized_mask = cv2.resize(
            cropped_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST
        )
    else:
        resized_mask = None

    image_3ch = to_3channel(resized_img).astype(np.float32)
    return image_3ch, resized_mask


def load_rgb_image(filepath: str | Path) -> np.ndarray:
    """Loads an image file (JPG/PNG) as an RGB NumPy array."""
    path = Path(filepath)
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with Image.open(path) as pil_img:
        return np.array(pil_img.convert("RGB"))


def _save_png(path: Path, array_rgb_or_gray: np.ndarray) -> None:
    """Writes an RGB (H, W, 3) or grayscale (H, W) uint8 array as PNG."""
    cv_img = (
        cv2.cvtColor(array_rgb_or_gray, cv2.COLOR_RGB2BGR)
        if array_rgb_or_gray.ndim == 3
        else array_rgb_or_gray
    )
    if not cv2.imwrite(str(path), cv_img):
        Image.fromarray(array_rgb_or_gray).save(path)


def _sanitize_sample_id(raw_id: Any, fallback: str) -> str:
    """Returns a filesystem-safe identifier derived from sample metadata."""
    if raw_id is None or pd.isna(raw_id) or not str(raw_id).strip():
        base = fallback
    else:
        base = str(raw_id).strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "sample"


def cache_preprocessed_manifest(
    manifest_df: pd.DataFrame,
    cache_dir: str | Path,
    image_size: int = 224,
    crop_margin_ratio: float = 0.05,
    use_clahe: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Preprocesses and caches MRI slices (and optional tumor masks) as PNG files.

    Iterates rows of ``manifest_df``, loads ``.mat`` files via ``read_figshare_mat``
    or standard images via OpenCV/PIL, runs ``preprocess_slice``, and saves
    ``uint8`` PNGs under ``cache_dir/images`` and ``cache_dir/masks``. Returns a
    copy of ``manifest_df`` with ``cached_path`` and ``cached_mask_path`` columns.
    When ``overwrite=False`` and target PNGs already exist on disk, skips work.
    """
    from mri_tumor.data.figshare import read_figshare_mat

    cache_root = Path(cache_dir)
    images_dir = cache_root / "images"
    masks_dir = cache_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)

    out_df = manifest_df.copy()
    if out_df.empty:
        out_df["cached_path"] = pd.Series(dtype="object")
        out_df["cached_mask_path"] = pd.Series(dtype="object")
        return out_df

    cached_paths: list[str] = []
    cached_mask_paths: list[str] = []

    for idx, (_, row) in enumerate(
        tqdm(out_df.iterrows(), total=len(out_df), desc=f"Caching {cache_root.name}", unit="slice")
    ):
        filepath = Path(row["filepath"])
        safe_id = _sanitize_sample_id(row.get("sample_id"), fallback=filepath.stem)
        is_mat = filepath.suffix.lower() == ".mat"

        img_out_path = images_dir / f"{safe_id}_{idx}.png"
        mask_out_path = masks_dir / f"{safe_id}_{idx}_mask.png"

        if not overwrite and img_out_path.is_file() and (not is_mat or mask_out_path.is_file()):
            cached_paths.append(str(img_out_path))
            cached_mask_paths.append(str(mask_out_path) if is_mat else "")
            continue

        if is_mat:
            sample = read_figshare_mat(filepath)
            raw_img = sample.image
            raw_mask = sample.tumor_mask
        else:
            raw_img = load_rgb_image(filepath)
            raw_mask = None

        img_3ch, mask_2d = preprocess_slice(
            image=raw_img,
            mask=raw_mask,
            target_size=(image_size, image_size),
            crop_margin_ratio=crop_margin_ratio,
            use_clahe=use_clahe,
        )

        img_u8_rgb = np.clip(np.round(img_3ch * 255.0), 0, 255).astype(np.uint8)
        _save_png(img_out_path, img_u8_rgb)
        cached_paths.append(str(img_out_path))

        if mask_2d is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_u8 = (mask_2d > 0).astype(np.uint8) * 255
            _save_png(mask_out_path, mask_u8)
            cached_mask_paths.append(str(mask_out_path))
        else:
            cached_mask_paths.append("")

    out_df["cached_path"] = cached_paths
    out_df["cached_mask_path"] = cached_mask_paths
    return out_df
