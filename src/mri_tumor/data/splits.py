"""Leak-free dataset partitioning, patient-grouped cross validation, and cascade test generation."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from mri_tumor.data.dedup import deduplicate_and_filter_leakage


def verify_no_patient_leakage(df: pd.DataFrame) -> bool:
    """Verifies that no patient ID overlaps across holdout vs CV folds, or between folds.

    Raises:
        ValueError if patient leakage is detected.

    Returns:
        True if zero leakage is verified.
    """
    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain 'patient_id' column for leakage check.")

    # 1. Holdout leakage check
    if "is_holdout" in df.columns:
        holdout_pts = set(df[df["is_holdout"]]["patient_id"].unique())
        cv_pts = set(df[~df["is_holdout"]]["patient_id"].unique())
        overlap = holdout_pts.intersection(cv_pts)
        if overlap:
            raise ValueError(
                f"Patient leakage detected between holdout and CV sets: {len(overlap)} patients overlap! "
                f"Samples: {list(overlap)[:5]}"
            )

    # 2. Split column leakage check (train / val / test)
    if "split" in df.columns:
        splits = df["split"].unique()
        for i, s1 in enumerate(splits):
            pts1 = set(df[df["split"] == s1]["patient_id"].unique())
            for s2 in splits[i + 1 :]:
                pts2 = set(df[df["split"] == s2]["patient_id"].unique())
                overlap = pts1.intersection(pts2)
                if overlap:
                    raise ValueError(
                        f"Patient leakage detected between split '{s1}' and '{s2}': {len(overlap)} patients overlap! "
                        f"Samples: {list(overlap)[:5]}"
                    )

    # 3. Inter-fold leakage check for CV rows
    if "fold" in df.columns:
        cv_df = df[df["fold"] >= 0]
        folds = sorted(cv_df["fold"].unique())
        for i, f1 in enumerate(folds):
            f1_pts = set(cv_df[cv_df["fold"] == f1]["patient_id"].unique())
            for f2 in folds[i + 1 :]:
                f2_pts = set(cv_df[cv_df["fold"] == f2]["patient_id"].unique())
                overlap = f1_pts.intersection(f2_pts)
                if overlap:
                    raise ValueError(
                        f"Patient leakage detected between fold {f1} and fold {f2}: {len(overlap)} patients overlap! "
                        f"Samples: {list(overlap)[:5]}"
                    )

    return True


def create_stage2_splits(
    figshare_df: pd.DataFrame,
    n_folds: int = 5,
    holdout_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Creates strictly patient-grouped, class-stratified holdout and k-fold cross-validation splits.

    Guarantees:
      - 0% patient leakage between holdout set and training/validation sets.
      - 0% patient leakage between CV folds.
      - Balanced class distributions across folds.

    Adds columns:
      - is_holdout (bool)
      - fold (int): -1 for holdout, 0..n_folds-1 for CV folds
      - split (str): 'test' for holdout, 'val' for fold 0, 'train' for other folds
    """
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(f"holdout_fraction must be between 0 and 1, got {holdout_fraction}")
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    df = figshare_df.copy().reset_index(drop=True)

    # Outer split to isolate patient-grouped holdout set
    n_outer_splits = max(2, int(round(1.0 / holdout_fraction)))
    outer_sgkf = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=seed)

    groups = df["patient_id"].astype(str).values
    strat_labels = df["class_idx"].values
    dummy_x = np.zeros(len(df))

    # Pick split 0 test fold as holdout set
    train_idx, holdout_idx = next(outer_sgkf.split(dummy_x, strat_labels, groups=groups))

    df["is_holdout"] = False
    df.loc[holdout_idx, "is_holdout"] = True
    df["fold"] = -1
    df["split"] = "train"
    df.loc[holdout_idx, "split"] = "test"

    # Inner split on non-holdout patients for n_folds cross-validation
    cv_mask = ~df["is_holdout"]
    cv_indices = df[cv_mask].index.values
    cv_groups = df.loc[cv_mask, "patient_id"].astype(str).values
    cv_labels = df.loc[cv_mask, "class_idx"].values
    cv_dummy_x = np.zeros(len(cv_indices))

    inner_sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_num, (_, val_fold_idx) in enumerate(
        inner_sgkf.split(cv_dummy_x, cv_labels, groups=cv_groups)
    ):
        orig_indices = cv_indices[val_fold_idx]
        df.loc[orig_indices, "fold"] = fold_num

    # Default canonical split: fold 0 is val, folds 1..n-1 are train
    df.loc[df["fold"] == 0, "split"] = "val"

    # Strict leakage validation
    verify_no_patient_leakage(df)

    return df


def create_stage1_splits(
    nickparvar_df: pd.DataFrame,
    figshare_split_df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_hamming_dist: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filters Figshare holdout leakage and splits Nickparvar into stratified train/val/test sets.

    Returns:
        (clean_split_df, removed_df)
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"Sum of val_fraction ({val_fraction}) and test_fraction ({test_fraction}) must be < 1.0"
        )

    holdout_phashes = figshare_split_df[figshare_split_df["is_holdout"]]["phash"].values

    clean_df, removed_df = deduplicate_and_filter_leakage(
        nickparvar_df=nickparvar_df,
        figshare_holdout_phashes=holdout_phashes,
        max_hamming_dist=max_hamming_dist,
    )

    clean_df = clean_df.copy().reset_index(drop=True)
    clean_df["split"] = "train"

    # Stratify by sub_class if available, else by binary_label
    strat_col = "sub_class" if "sub_class" in clean_df.columns else "binary_label"
    strat_labels = clean_df[strat_col].values

    # Step 1: Split off test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(np.zeros(len(clean_df)), strat_labels))

    clean_df.loc[test_idx, "split"] = "test"

    # Step 2: Split remaining trainval into train and val
    relative_val_size = val_fraction / (1.0 - test_fraction)
    trainval_labels = clean_df.loc[trainval_idx, strat_col].values

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_rel_idx, val_rel_idx = next(sss_val.split(np.zeros(len(trainval_idx)), trainval_labels))

    val_idx = trainval_idx[val_rel_idx]
    clean_df.loc[val_idx, "split"] = "val"

    return clean_df, removed_df


def build_cascade_test_manifest(
    stage1_split_df: pd.DataFrame,
    stage2_split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Builds a unified test set for full cascade evaluation.

    Combines:
      - Figshare holdout tumor patients (binary_label=1, subtype_label in {0, 1, 2})
      - NickParvar held-out 'notumor' controls (split='test', binary_label=0, subtype_label=-1)

    Returns:
      DataFrame with columns:
        sample_id, filepath, source_dataset, patient_id,
        binary_label, subtype_label, class_name
    """
    records = []

    # 1. Figshare holdout tumor patients
    figshare_holdout = stage2_split_df[stage2_split_df["is_holdout"]].copy()
    for _, row in figshare_holdout.iterrows():
        records.append({
            "sample_id": str(row["sample_id"]),
            "filepath": str(row["filepath"]),
            "source_dataset": "figshare",
            "patient_id": str(row.get("patient_id", row["sample_id"])),
            "binary_label": 1,
            "subtype_label": int(row["class_idx"]),
            "class_name": str(row["class_name"]),
        })

    # 2. NickParvar held-out no_tumor controls
    np_test_notumor = stage1_split_df[
        (stage1_split_df["split"] == "test") & (stage1_split_df["binary_label"] == 0)
    ].copy()

    for _, row in np_test_notumor.iterrows():
        records.append({
            "sample_id": str(row["sample_id"]),
            "filepath": str(row["filepath"]),
            "source_dataset": "nickparvar",
            "patient_id": f"np_{row['sample_id']}",
            "binary_label": 0,
            "subtype_label": -1,
            "class_name": "no_tumor",
        })

    columns = [
        "sample_id",
        "filepath",
        "source_dataset",
        "patient_id",
        "binary_label",
        "subtype_label",
        "class_name",
    ]
    return pd.DataFrame(records, columns=columns).reset_index(drop=True) if records else pd.DataFrame(columns=columns)
