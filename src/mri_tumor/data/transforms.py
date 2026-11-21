"""Data augmentation and evaluation transforms for MRI slices and masks."""

from __future__ import annotations

import albumentations as A


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Returns training data augmentation pipeline using Albumentations.

    Includes:
      - Random horizontal flip (p=0.5). NOTE: NO vertical flip (axial slice asymmetry).
      - Affine / ShiftScaleRotate: rotation (-10, 10) deg, scale (0.95, 1.05), shift (-0.05, 0.05), p=0.5.
      - RandomBrightnessContrast (limit 0.15, p=0.5).
      - RandomGamma (gamma_limit=(80, 120), p=0.3).
      - Gaussian noise (p=0.2).
      - Resize to (image_size, image_size).
    """
    noise_cls = getattr(A, "GaussNoise", getattr(A, "GaussianNoise", None))

    transform_list = [
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ]

    if noise_cls is not None:
        transform_list.append(noise_cls(p=0.2))

    transform_list.append(A.Resize(image_size, image_size))

    return A.Compose(transform_list)


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Returns deterministic evaluation transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
