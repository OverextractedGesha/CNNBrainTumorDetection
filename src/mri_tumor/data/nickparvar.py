"""NickParvar dataset extractor, manifest builder, and binary tumor labeler."""

from __future__ import annotations

from pathlib import Path
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from mri_tumor.data.dedup import compute_phash

NICKPARVAR_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "notumor")

STAGE1_BINARY_MAP: dict[str, int] = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 1,
    "pituitary": 1,
}

STAGE1_CLASS_NAMES: dict[int, str] = {
    0: "no_tumor",
    1: "tumor",
}


def normalize_nickparvar_subclass(name: str) -> str | None:
    """Normalizes class folder name to standard NickParvar class."""
    cleaned = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    if "notumor" in cleaned or "notumour" in cleaned or "normal" in cleaned:
        return "notumor"
    if "glioma" in cleaned:
        return "glioma"
    if "meningioma" in cleaned:
        return "meningioma"
    if "pituitary" in cleaned:
        return "pituitary"
    return None


def extract_nickparvar_archive(zip_path: str | Path, output_dir: str | Path) -> Path:
    """Extracts NickParvar archive to output_dir.

    Args:
        zip_path: Path to archive file.
        output_dir: Destination folder.

    Returns:
        Path to output_dir.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    return output_dir


def build_nickparvar_manifest(
    nickparvar_dir: str | Path,
    output_csv: str | Path | None = None,
    compute_hashes: bool = True,
    crop_margin_ratio: float = 0.05,
) -> pd.DataFrame:
    """Scans nickparvar_dir for images (.jpg, .jpeg, .png) and builds a manifest DataFrame.

    Returns DataFrame with columns:
        sample_id, filepath, original_split, sub_class, binary_label,
        binary_class_name, height, width, phash
    """
    nickparvar_dir = Path(nickparvar_dir)
    records = []

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        p for p in nickparvar_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions
    )

    for path in tqdm(image_files, desc="Scanning NickParvar", unit="slice"):

        parts_lower = [p.lower() for p in path.parts]

        # Determine original split
        if any("train" in p for p in parts_lower):
            original_split = "train"
        elif any("test" in p for p in parts_lower):
            original_split = "test"
        else:
            original_split = "unknown"

        # Determine sub_class by searching parent folder names
        sub_class = None
        for p in reversed(parts_lower[:-1]):
            norm = normalize_nickparvar_subclass(p)
            if norm is not None:
                sub_class = norm
                break

        if sub_class is None:
            # Check filename as fallback
            sub_class = normalize_nickparvar_subclass(path.stem)
            if sub_class is None:
                continue

        binary_label = STAGE1_BINARY_MAP[sub_class]
        binary_class_name = STAGE1_CLASS_NAMES[binary_label]

        # Read image to obtain dimensions and compute hash
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            try:
                with Image.open(path) as pil_img:
                    width, height = pil_img.size
                    img_np = np.array(pil_img)
            except Exception:
                continue
        else:
            height, width = img_bgr.shape[:2]
            img_np = img_bgr

        phash_val = (
            compute_phash(img_np, crop_first=True, crop_margin_ratio=crop_margin_ratio)
            if compute_hashes
            else 0
        )

        records.append({
            "sample_id": path.stem,
            "filepath": str(path.resolve()),
            "original_split": original_split,
            "sub_class": sub_class,
            "binary_label": binary_label,
            "binary_class_name": binary_class_name,
            "height": height,
            "width": width,
            "phash": phash_val,
        })

    columns = [
        "sample_id",
        "filepath",
        "original_split",
        "sub_class",
        "binary_label",
        "binary_class_name",
        "height",
        "width",
        "phash",
    ]
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)

    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df
