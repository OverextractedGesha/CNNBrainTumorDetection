"""Data ingestion, transforms, dataset manifests, and deduplication modules."""

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
    STAGE2_IDX_TO_CLASS,
    FigshareSample,
    build_figshare_manifest,
    extract_figshare_archive,
    read_figshare_mat,
)
from mri_tumor.data.nickparvar import (
    NICKPARVAR_CLASSES,
    STAGE1_BINARY_MAP,
    STAGE1_CLASS_NAMES,
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

__all__ = [
    "SliceDataset",
    "resolve_dataloader_kwargs",
    "BKTree",
    "compute_phash",
    "hamming_distance",
    "deduplicate_and_filter_leakage",
    "FIGSHARE_LABEL_MAP",
    "STAGE2_CLASS_TO_IDX",
    "STAGE2_IDX_TO_CLASS",
    "FigshareSample",
    "extract_figshare_archive",
    "read_figshare_mat",
    "build_figshare_manifest",
    "NICKPARVAR_CLASSES",
    "STAGE1_BINARY_MAP",
    "STAGE1_CLASS_NAMES",
    "extract_nickparvar_archive",
    "build_nickparvar_manifest",
    "crop_brain_contour",
    "apply_clahe",
    "normalize_intensity",
    "to_3channel",
    "preprocess_slice",
    "cache_preprocessed_manifest",
    "create_stage2_splits",
    "create_stage1_splits",
    "build_cascade_test_manifest",
    "verify_no_patient_leakage",
    "get_train_transforms",
    "get_eval_transforms",
]
