"""
Deep Learning model definition for ICDR Diabetic Retinopathy Grading.

Uses EfficientNet-B0 via timm (or fallback PyTorch torchvision) with custom classification head.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.utils.config import GRADING_MODEL_NAME, GRADING_NUM_CLASSES


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


def build_dr_model(pretrained: bool = True) -> DRGradingModel:
    model = DRGradingModel(num_classes=GRADING_NUM_CLASSES, pretrained=pretrained)
    model.eval()
    return model
