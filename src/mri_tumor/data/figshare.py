"""Figshare brain MRI dataset reader, extractor, and manifest builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import zipfile

import h5py
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

from mri_tumor.data.dedup import compute_phash

FIGSHARE_LABEL_MAP: dict[int, str] = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}

STAGE2_CLASS_TO_IDX: dict[str, int] = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
}

STAGE2_IDX_TO_CLASS: dict[int, str] = {
    0: "glioma",
    1: "meningioma",
    2: "pituitary",
}


@dataclass
class FigshareSample:
    """Represents a single parsed Figshare MRI slice with tumor mask."""

    sample_id: str
    patient_id: str
    label_raw: int
    class_name: str
    class_idx: int
    image: np.ndarray
    tumor_mask: np.ndarray
    tumor_border: np.ndarray
    source_path: str

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def mask_area_ratio(self) -> float:
        total = float(self.height * self.width)
        return float(self.tumor_mask.sum()) / total if total > 0 else 0.0


def extract_figshare_archive(zip_path: str | Path, output_dir: str | Path) -> Path:
    """Extracts Figshare archive containing .mat files (supports nested zip structure).

    Args:
        zip_path: Path to outer zip file (e.g. 'Dataset/Figshare Dataset.zip').
        output_dir: Destination directory for extracted .mat files.

    Returns:
        Path to output_dir.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check if archive contains nested zips (e.g. brainTumorDataPublic_*.zip)
        nested_zips = [name for name in zf.namelist() if name.lower().endswith(".zip")]
        if nested_zips:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                zf.extractall(tmp_path)
                for nested in tmp_path.rglob("*.zip"):
                    with zipfile.ZipFile(nested, "r") as nzf:
                        for member in nzf.namelist():
                            if member.lower().endswith(".mat"):
                                # Extract directly to output_dir flattening directory structure
                                filename = Path(member).name
                                with nzf.open(member) as src, open(output_dir / filename, "wb") as dst:
                                    dst.write(src.read())
        else:
            # Flat zip of .mat files
            for member in zf.namelist():
                if member.lower().endswith(".mat"):
                    filename = Path(member).name
                    with zf.open(member) as src, open(output_dir / filename, "wb") as dst:
                        dst.write(src.read())

    return output_dir


def _decode_pid(raw_pid: np.ndarray | bytes | str | int) -> str:
    """Decodes MATLAB PID representations into a clean string."""
    if isinstance(raw_pid, (str, int)):
        return str(raw_pid).strip()
    if isinstance(raw_pid, bytes):
        return raw_pid.decode("utf-8", errors="ignore").strip()

    arr = np.array(raw_pid).flatten()
    if arr.dtype.kind in ("U", "S", "O"):
        # Array of strings or objects
        return "".join(str(x) for x in arr).strip()
    if np.issubdtype(arr.dtype, np.integer):
        # ASCII character codes in MATLAB v7.3
        chars = [chr(int(c)) for c in arr if int(c) != 0]
        return "".join(chars).strip()

    return str(raw_pid).strip()


def read_figshare_mat(mat_path: str | Path) -> FigshareSample:
    """Reads a Figshare .mat file (supporting both MATLAB v7.3 HDF5 and legacy v5 formats).

    Args:
        mat_path: Path to the .mat slice file.

    Returns:
        FigshareSample dataclass.
    """
    mat_path = Path(mat_path)
    sample_id = mat_path.stem

    try:
        # Try HDF5 MATLAB v7.3 first
        with h5py.File(mat_path, "r") as f:
            if "cjdata" not in f:
                raise KeyError(f"Missing 'cjdata' group in {mat_path}")
            cj = f["cjdata"]

            raw_label = int(np.array(cj["label"]).flatten()[0])
            patient_id = _decode_pid(np.array(cj["PID"]))

            # MATLAB v7.3 arrays are transposed relative to MATLAB column-major
            image = np.array(cj["image"]).T.astype(np.float32)

            raw_mask = np.array(cj["tumorMask"]).T
            tumor_mask = (raw_mask > 0).astype(np.uint8)

            if "tumorBorder" in cj:
                tumor_border = np.array(cj["tumorBorder"]).flatten().astype(np.float32)
            else:
                tumor_border = np.array([], dtype=np.float32)

    except (OSError, KeyError, ValueError):
        # Fallback to legacy MATLAB v5 format
        mat = scipy.io.loadmat(str(mat_path))
        if "cjdata" not in mat:
            raise ValueError(f"Cannot parse {mat_path}: 'cjdata' not found")
        cj = mat["cjdata"]
        # In scipy.io.loadmat, cjdata is a structured array
        raw_label = int(cj["label"][0, 0][0, 0])
        raw_pid = cj["PID"][0, 0]
        patient_id = _decode_pid(raw_pid)

        image = cj["image"][0, 0].astype(np.float32)
        tumor_mask = (cj["tumorMask"][0, 0] > 0).astype(np.uint8)

        if "tumorBorder" in cj.dtype.names:
            tumor_border = cj["tumorBorder"][0, 0].flatten().astype(np.float32)
        else:
            tumor_border = np.array([], dtype=np.float32)

    if raw_label not in FIGSHARE_LABEL_MAP:
        raise ValueError(f"Unknown raw label {raw_label} in {mat_path}")

    class_name = FIGSHARE_LABEL_MAP[raw_label]
    class_idx = STAGE2_CLASS_TO_IDX[class_name]

    return FigshareSample(
        sample_id=sample_id,
        patient_id=patient_id,
        label_raw=raw_label,
        class_name=class_name,
        class_idx=class_idx,
        image=image,
        tumor_mask=tumor_mask,
        tumor_border=tumor_border,
        source_path=str(mat_path.resolve()),
    )


def build_figshare_manifest(
    figshare_dir: str | Path,
    output_csv: str | Path | None = None,
    compute_hashes: bool = True,
    crop_margin_ratio: float = 0.05,
) -> pd.DataFrame:
    """Recursively scans figshare_dir for .mat files and builds a manifest DataFrame.

    Ignores auxiliary files like 'cvind.mat'.

    Returns DataFrame with columns:
        sample_id, filepath, patient_id, label_raw, class_name, class_idx,
        height, width, mask_area_ratio, phash
    """
    figshare_dir = Path(figshare_dir)
    records = []

    mat_files = sorted(figshare_dir.rglob("*.mat"))
    for mf in tqdm(mat_files, desc="Scanning Figshare", unit="slice"):
        # Ignore cvind.mat or any cross-validation index file
        if mf.name.lower() in ("cvind.mat", "cvind"):
            continue

        try:
            sample = read_figshare_mat(mf)
        except Exception:
            # Skip unparseable files without cjdata
            continue

        phash_val = (
            compute_phash(sample.image, crop_first=True, crop_margin_ratio=crop_margin_ratio)
            if compute_hashes
            else 0
        )

        records.append({
            "sample_id": sample.sample_id,
            "filepath": sample.source_path,
            "patient_id": sample.patient_id,
            "label_raw": sample.label_raw,
            "class_name": sample.class_name,
            "class_idx": sample.class_idx,
            "height": sample.height,
            "width": sample.width,
            "mask_area_ratio": sample.mask_area_ratio,
            "phash": phash_val,
        })

    columns = [
        "sample_id",
        "filepath",
        "patient_id",
        "label_raw",
        "class_name",
        "class_idx",
        "height",
        "width",
        "mask_area_ratio",
        "phash",
    ]
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)

    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df
