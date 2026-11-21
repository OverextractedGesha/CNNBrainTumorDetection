"""Baseline CNN and timm backbone classifiers with Grad-CAM layer resolution and freezing control."""

from __future__ import annotations

from typing import Self

import timm
import torch
import torch.nn as nn


def _has_conv2d(module: nn.Module) -> bool:
    """Returns True if ``module`` contains at least one ``nn.Conv2d`` layer."""
    return any(isinstance(sub, nn.Conv2d) for sub in module.modules())


def disable_inplace_activations(module: nn.Module) -> None:
    """Sets ``inplace=False`` on activation submodules to keep Grad-CAM backward hooks safe."""
    for submodule in module.modules():
        if isinstance(getattr(submodule, "inplace", None), bool):
            submodule.inplace = False


def resolve_target_conv_layer(backbone: nn.Module) -> nn.Module:
    """Resolves the target convolutional layer/block on a CNN backbone for Grad-CAM.

    Resolution order:
      1. Pre-pool ``backbone.conv_head`` for EfficientNet-family models.
      2. ``backbone.layer4[-1]`` for ResNet-family models with Sequential ``layer4``.
      3. ``backbone.stages[-1]`` for ConvNeXt-family models with Sequential ``stages``.
      4. ``backbone.blocks[-1]`` for MobileNet-family models when ``conv_head`` is post-pool/absent.
      5. ``backbone.features[-1]`` for DenseNet/VGG-family models with Sequential ``features``.
      6. Last ``nn.Conv2d`` encountered while iterating ``backbone.modules()``.

    Raises:
        ValueError: If no suitable convolutional layer exists in ``backbone``.
    """
    child_names = [name for name, _ in backbone.named_children()]
    conv_head = getattr(backbone, "conv_head", None)
    if isinstance(conv_head, nn.Module) and not isinstance(conv_head, nn.Identity):
        if "global_pool" not in child_names or (
            "conv_head" in child_names
            and child_names.index("conv_head") < child_names.index("global_pool")
        ):
            return conv_head

    for attr in ("layer4", "stages", "blocks"):
        container = getattr(backbone, attr, None)
        if isinstance(container, nn.Sequential) and len(container) > 0 and _has_conv2d(container[-1]):
            return container[-1]

    features = getattr(backbone, "features", None)
    if isinstance(features, nn.Sequential) and len(features) > 0 and _has_conv2d(features):
        return features[-1]

    last_conv: nn.Conv2d | None = None
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError(
            "Unable to resolve target convolutional layer: no nn.Conv2d found in backbone."
        )
    return last_conv


class _BackboneControlMixin:
    """Mixin providing two-phase backbone freeze/unfreeze with BatchNorm eval locking."""

    backbone: nn.Module

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._backbone_frozen: bool = False

    def freeze_backbone(self) -> None:
        """Freezes backbone parameters and switches backbone submodules to eval mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreezes backbone parameters and syncs backbone train/eval mode with parent."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        self.backbone.train(self.training)

    def train(self, mode: bool = True) -> Self:
        """Sets module training mode while keeping frozen backbone in eval mode."""
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def get_target_layer(self) -> nn.Module:
        """Returns the Grad-CAM target convolutional layer/block of the backbone."""
        return resolve_target_conv_layer(self.backbone)


def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Builds a standard Conv-BN-ReLU-MaxPool block with non-inplace ReLU for Grad-CAM."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def build_simple_cnn_backbone() -> nn.Sequential:
    """Builds the 4-block SimpleCNN convolutional feature extractor (3 -> 256 channels)."""
    return nn.Sequential(
        _make_conv_block(3, 32),
        _make_conv_block(32, 64),
        _make_conv_block(64, 128),
        _make_conv_block(128, 256),
    )


def _validate_classifier_args(num_classes: int, dropout: float) -> None:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")


def _make_pooled_head(
    feature_dim: int, num_classes: int, dropout: float
) -> tuple[nn.AdaptiveAvgPool2d, nn.Sequential]:
    return (
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        ),
    )


class SimpleCNN(_BackboneControlMixin, nn.Module):
    """Lightweight 4-block CNN classifier (3 -> 32 -> 64 -> 128 -> 256)."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        _validate_classifier_args(num_classes, dropout)
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 256

        self.backbone = build_simple_cnn_backbone()
        self.pool, self.head = _make_pooled_head(self.feature_dim, num_classes, dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial feature map of shape (B, 256, H', W')."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits of shape (B, num_classes)."""
        features = self.forward_features(x)
        return self.head(self.pool(features))


class TimmClassifier(_BackboneControlMixin, nn.Module):
    """Pretrained or randomly initialized timm CNN classifier with global pooling head."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        _validate_classifier_args(num_classes, dropout)
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        disable_inplace_activations(self.backbone)
        self.feature_dim = int(self.backbone.num_features)
        self.pool, self.head = _make_pooled_head(self.feature_dim, num_classes, dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts raw spatial feature map (B, C, H, W) from the timm backbone."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits of shape (B, num_classes)."""
        feat = self.forward_features(x)
        return self.head(self.pool(feat))
