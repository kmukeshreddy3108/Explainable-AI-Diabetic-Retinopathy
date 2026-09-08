"""
Sharpness assessment using Laplacian variance.

Measures high-frequency detail in fundus images. Low variance indicates blur
(out of focus, patient movement, or cataract haze).
"""

from typing import Tuple
import cv2
import numpy as np

from src.utils.config import IQA_SHARPNESS_THRESHOLD


def compute_sharpness_score(img_rgb: np.ndarray) -> float:
    """
    Compute Laplacian variance score for image sharpness.

    Args:

        img_rgb: Input RGB image as numpy array uint8 (H, W, 3) or uint8 grayscale (H, W).

    Returns:
        float: Sharpness score (variance of Laplacian). Higher = sharper.
    """
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    # Compute variance of Laplacian operator
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return laplacian_var


def check_sharpness(
    img_rgb: np.ndarray, threshold: float = IQA_SHARPNESS_THRESHOLD
) -> Tuple[float, bool]:
    """
    Check if image meets minimum sharpness threshold.

    Returns:
        (score, is_sharp)
    """
    score = compute_sharpness_score(img_rgb)
    is_sharp = score >= threshold
    return score, is_sharp
