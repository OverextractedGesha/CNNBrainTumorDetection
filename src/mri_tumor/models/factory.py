"""Model factory helpers for constructing classifiers from explicit args or YAML configs."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from mri_tumor.models.cnn import SimpleCNN, TimmClassifier


def build_model(
    arch: str,
    backbone: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Constructs a classifier model by architecture name and backbone configuration."""
    arch_norm = arch.strip().lower()
    backbone_norm = backbone.strip().lower()

    if arch_norm == "simple_cnn" or (arch_norm == "cnn" and backbone_norm == "simple_cnn"):
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    if arch_norm == "cnn":
        return TimmClassifier(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    raise ValueError(
        f"Unsupported model architecture {arch!r}. Expected 'cnn' or 'simple_cnn'."
    )


def build_model_from_config(cfg: dict[str, Any]) -> nn.Module:
    """Builds a model from a parsed YAML stage config (or its ``model`` sub-dictionary)."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected config dict, got {type(cfg).__name__}.")
    model_cfg = cfg["model"] if "model" in cfg else cfg
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Expected model config dict, got {type(model_cfg).__name__}.")

    return build_model(
        arch=str(model_cfg.get("arch", "cnn")),
        backbone=str(model_cfg.get("backbone", "efficientnet_b0")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
