"""Perceptual hashing and metric-tree deduplication for MRI leak prevention."""

from __future__ import annotations

from typing import Iterable, Sequence
import cv2
import numpy as np
import pandas as pd
import scipy.fft

from mri_tumor.data.preprocess import crop_brain_contour


def compute_phash(
    image: np.ndarray,
    hash_size: int = 8,
    highfreq_factor: int = 4,
    crop_first: bool = True,
    crop_margin_ratio: float = 0.05,
) -> int:
    """Computes a 64-bit DCT perceptual hash from a brain MRI slice.

    Args:
        image: 2D or 3D image array.
        hash_size: Size of low-frequency DCT block (default 8 -> 8x8 = 64-bit hash).
        highfreq_factor: Oversampling factor for initial resize (default 4 -> 32x32).
        crop_first: If True, crops brain contour before computing hash.
        crop_margin_ratio: Margin ratio used when crop_first=True.

    Returns:
        64-bit integer representation of the hash.
    """
    if crop_first:
        cropped, _, _ = crop_brain_contour(image, margin_ratio=crop_margin_ratio)
        img = cropped
    else:
        img = image

    # Convert to 2D grayscale float32
    if img.ndim == 2:
        img_f32 = img.astype(np.float32)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img_f32 = img[:, :, 0].astype(np.float32)
        else:
            img_f32 = np.mean(img, axis=2).astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

    # Resize to (img_size, img_size)
    img_size = hash_size * highfreq_factor
    interp = cv2.INTER_AREA if (img_f32.shape[0] >= img_size and img_f32.shape[1] >= img_size) else cv2.INTER_LINEAR
    resized = cv2.resize(img_f32, (img_size, img_size), interpolation=interp)

    # 2D Discrete Cosine Transform (ortho normalized)
    dct = scipy.fft.dctn(resized, type=2, norm="ortho")

    # Extract top-left (hash_size, hash_size) low-frequency block
    dct_low = dct[:hash_size, :hash_size]

    # Median threshold of AC coefficients (exclude DC component at [0, 0])
    ac_coeffs = dct_low.flatten()[1:]
    if len(ac_coeffs) == 0:
        med = float(dct_low[0, 0])
    else:
        med = float(np.median(ac_coeffs))

    # Binary comparison: 1 if > median, 0 otherwise
    diff = (dct_low.flatten() > med)

    # Pack 64 boolean bits into a 64-bit unsigned integer
    phash = 0
    for bit in diff:
        phash = (phash << 1) | int(bool(bit))

    return phash


def _parse_hash(h: int | str | np.integer) -> int:
    """Safely converts integer or hex/decimal string to integer."""
    if isinstance(h, (int, np.integer)):
        return int(h)
    s = str(h).strip()
    if s.startswith(("0x", "0X")):
        return int(s, 16)
    if any(c in "abcdefABCDEF" for c in s):
        return int(s, 16)
    return int(s)


def hamming_distance(hash1: int | str | np.integer, hash2: int | str | np.integer) -> int:
    """Computes the Hamming distance (number of bit positions where bits differ)."""
    h1 = _parse_hash(hash1)
    h2 = _parse_hash(hash2)
    return (h1 ^ h2).bit_count()


class _BKNode:
    """Node in a Burkhard-Keller metric tree."""

    def __init__(self, item_id: str, phash: int):
        self.item_id = item_id
        self.phash = phash
        self.duplicates: list[str] = []
        self.children: dict[int, _BKNode] = {}


class BKTree:
    """Burkhard-Keller tree for discrete metric search (Hamming distance)."""

    def __init__(self):
        self.root: _BKNode | None = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def add(self, item_id: str, phash: int | str | np.integer) -> None:
        """Inserts a hash into the tree."""
        h = _parse_hash(phash)
        self._size += 1

        if self.root is None:
            self.root = _BKNode(item_id, h)
            return

        curr = self.root
        while True:
            dist = hamming_distance(curr.phash, h)
            if dist == 0:
                curr.duplicates.append(item_id)
                return
            if dist in curr.children:
                curr = curr.children[dist]
            else:
                curr.children[dist] = _BKNode(item_id, h)
                return

    def search(self, phash: int | str | np.integer, max_dist: int = 6) -> list[tuple[str, int, int]]:
        """Finds all items within max_dist Hamming distance.

        Returns:
            list of (item_id, matched_hash, distance) sorted by distance.
        """
        h = _parse_hash(phash)
        if self.root is None:
            return []

        results: list[tuple[str, int, int]] = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            d = hamming_distance(node.phash, h)
            if d <= max_dist:
                results.append((node.item_id, node.phash, d))
                for dup_id in node.duplicates:
                    results.append((dup_id, node.phash, d))

            low = d - max_dist
            high = d + max_dist
            for edge_dist, child in node.children.items():
                if low <= edge_dist <= high:
                    stack.append(child)

        results.sort(key=lambda x: x[2])
        return results


def deduplicate_and_filter_leakage(
    nickparvar_df: pd.DataFrame,
    figshare_holdout_phashes: Iterable[int | str],
    max_hamming_dist: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters cross-dataset holdout leakage and internal duplicates in Nickparvar dataset.

    Checks each row:
      1. Against Figshare holdout phashes (reject with 'figshare_holdout_leakage' if dist <= max_dist)
      2. Against already-accepted Nickparvar phashes (reject with 'internal_near_duplicate' if dist <= max_dist)
      3. Otherwise accepts and indexes the row.

    Returns:
      (clean_df, removed_df)
    """
    holdout_tree = BKTree()
    for idx, ph in enumerate(figshare_holdout_phashes):
        holdout_tree.add(f"figshare_holdout_{idx}", ph)

    internal_tree = BKTree()

    clean_indices: list[int] = []
    removed_records: list[dict] = []

    for idx, row in nickparvar_df.iterrows():
        sample_id = str(row.get("sample_id", idx))
        phash_val = row["phash"]

        # Step 1: Check cross-dataset leakage into Figshare holdout
        holdout_matches = holdout_tree.search(phash_val, max_dist=max_hamming_dist)
        if holdout_matches:
            best_match = holdout_matches[0]
            rec = row.to_dict()
            rec["drop_reason"] = "figshare_holdout_leakage"
            rec["matched_id"] = best_match[0]
            rec["min_hamming_dist"] = best_match[2]
            removed_records.append(rec)
            continue

        # Step 2: Check internal near-duplicate
        internal_matches = internal_tree.search(phash_val, max_dist=max_hamming_dist)
        if internal_matches:
            best_match = internal_matches[0]
            rec = row.to_dict()
            rec["drop_reason"] = "internal_near_duplicate"
            rec["matched_id"] = best_match[0]
            rec["min_hamming_dist"] = best_match[2]
            removed_records.append(rec)
            continue

        # Step 3: Accept and register into internal tree
        clean_indices.append(idx)
        internal_tree.add(sample_id, phash_val)

    clean_df = nickparvar_df.loc[clean_indices].copy().reset_index(drop=True)
    removed_df = pd.DataFrame(removed_records).reset_index(drop=True) if removed_records else pd.DataFrame(
        columns=list(nickparvar_df.columns) + ["drop_reason", "matched_id", "min_hamming_dist"]
    )

    return clean_df, removed_df
