"""
Grad-CAM Heatmap Generation Module.

Generates Class Activation Maps (CAM) highlighting the visual regions
most influential in the model's ICDR severity prediction.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.config import GRADCAM_HEATMAP_ALPHA, GRADCAM_ATTENTION_THRESHOLD


class GradCAM:
    """Grad-CAM implementation for PyTorch CNN classification models."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model.eval()

        # Find target layer (default to last Conv2d layer)
        if target_layer is None:
            conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            if len(conv_layers) > 0:
                target_layer = conv_layers[-1]
            else:
                raise ValueError("No Conv2d layer found in model for Grad-CAM")

        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward & backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate normalized Grad-CAM heatmap (0.0 to 1.0) for input tensor.

        Args:
            input_tensor: Shape (1, 3, H, W)
            target_class: Target class index (0–4). If None, uses predicted class.

        Returns:
            (heatmap, target_class)
        """
        self.gradients = None
        self.activations = None

        with torch.enable_grad():
            inp = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()
            logits = self.model(inp)

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            score = logits[0, target_class]
            score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't trigger
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32), target_class

        # Global average pooling of gradients
        weights = torch.mean(self.gradients.detach(), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations.detach(), dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to keep positive contributions

        # Resize to match input spatial dimensions
        h, w = input_tensor.shape[2:]
        cam_resized = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_np = cam_resized.squeeze().cpu().numpy()

        # Normalize 0.0 to 1.0
        max_val = np.max(cam_np)
        if max_val > 0:
            cam_np = cam_np / max_val
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class


def overlay_heatmap(
    img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = GRADCAM_HEATMAP_ALPHA
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original fundus image using Jet colormap.

    Args:
        img_rgb: Original RGB image uint8 (H, W, 3)
        heatmap: Float numpy array (H, W) in range 0.0 to 1.0

    Returns:
        RGB image uint8 (H, W, 3) with heatmap overlay
    """
    h, w = img_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return overlay
