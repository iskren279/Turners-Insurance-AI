"""Build the backbone classifier via timm.

The baseline is a single flat ``fine_class`` classifier on top of a pretrained
timm backbone (``convnext_tiny`` / ``efficientnet_b0`` / ``resnet50`` are good,
stable choices). ``timm.create_model`` swaps the head for a ``num_classes``
linear layer, so the backbone string fully determines the architecture.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Create a timm classifier with a fresh ``num_classes`` head.

    ``pretrained=True`` downloads ImageNet weights (needs network); the smoke
    test sets it to ``False`` so the model is constructed fully offline.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
