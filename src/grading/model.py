"""
Deep Learning model definition for ICDR Diabetic Retinopathy Grading.

Uses EfficientNet-B0 via timm (or fallback PyTorch torchvision) with custom classification head.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.utils.config import (
    GRADING_MODEL_NAME,
    GRADING_NUM_CLASSES,
    GRADING_INPUT_SIZE,
    GRADING_NORMALIZE_MEAN,
    GRADING_NORMALIZE_STD,
)


class DRGradingModel(nn.Module):
    """EfficientNet-B0 classifier for 5-class ICDR DR grading."""

    def __init__(self, num_classes: int = GRADING_NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        if HAS_TIMM:
            self.backbone = timm.create_model(
                GRADING_MODEL_NAME, pretrained=pretrained, num_classes=num_classes
            )
        else:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W) normalized.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)

    def predict_probs(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute calibrated class probabilities using temperature scaling.
        """
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=1)


def preprocess_fundus_image(
    img_rgb: np.ndarray, target_size: Tuple[int, int] = (GRADING_INPUT_SIZE, GRADING_INPUT_SIZE)
) -> torch.Tensor:
    """
    Standardized ImageNet preprocessing matching PyTorch DataLoader pipeline.

    Args:
        img_rgb: Input RGB uint8 numpy array (H, W, 3)
        target_size: Output (width, height)

    Returns:
        Tensor of shape (1, 3, target_size[0], target_size[1]) normalized with ImageNet mean & std.
    """
    img_resized = cv2.resize(img_rgb, target_size)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(GRADING_NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(GRADING_NORMALIZE_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def build_dr_model(pretrained: bool = True) -> DRGradingModel:
    model = DRGradingModel(num_classes=GRADING_NUM_CLASSES, pretrained=pretrained)
    model.eval()
    return model


def load_dr_model(
    checkpoint_path: str = "models/dr_grading_efficientnet.pt", pretrained: bool = True
) -> Tuple[DRGradingModel, bool]:
    """
    Build DRGradingModel and load trained weights if present.

    Returns:
        Tuple of (model, is_checkpoint_loaded)
    """
    model = build_dr_model(pretrained=pretrained)
    checkpoint_loaded = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            checkpoint_loaded = True
        except Exception as err:
            print(f"Warning: Failed to load checkpoint from {checkpoint_path}: {err}")

    model.eval()
    return model, checkpoint_loaded

