"""
Exposure and Glare assessment module.

Analyzes illuminance and contrast across macular (center) and peripheral zones
within the fundus FOV mask. Detects overexposure, underexposure, and specular glare.
"""

from typing import Dict, Tuple, Any
import cv2
import numpy as np

from src.utils.config import (
    IQA_DARK_THRESHOLD,
    IQA_BRIGHT_THRESHOLD,
    IQA_MACULAR_DARK_FACTOR,
    IQA_PERIPHERAL_BRIGHT_FACTOR,
    IQA_GLARE_MAX_RATIO,
)


def assess_exposure(
    img_rgb: np.ndarray, fov_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Assess exposure and glare within the FOV mask.

    Args:
        img_rgb: Input RGB image (H, W, 3)
        fov_mask: Binary mask of fundus (H, W) uint8

    Returns:
        Dict containing:
            - overall_luminance_mean: float
            - macular_luminance_mean: float
            - peripheral_luminance_mean: float
            - glare_ratio: float
            - is_underexposed: bool
            - is_overexposed: bool
            - has_severe_glare: bool
    """
    # Convert RGB to YCrCb or Lab to get clean luminance channel
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]  # L channel 0..255

    valid_mask = fov_mask > 0
    if not np.any(valid_mask):
        return {
            "overall_luminance_mean": 0.0,
            "macular_luminance_mean": 0.0,
            "peripheral_luminance_mean": 0.0,
            "glare_ratio": 0.0,
            "is_underexposed": True,
            "is_overexposed": False,
            "has_severe_glare": False,
        }

    overall_mean = float(np.mean(luminance[valid_mask]))

    # Macular vs Peripheral Partitioning inside FOV
    # Find bounding circle or center of mass of FOV mask
    y_indices, x_indices = np.where(valid_mask)
    cy, cx = int(np.mean(y_indices)), int(np.mean(x_indices))
    h, w = fov_mask.shape

    # Define macular region as inner circle (radius = 20% of min dimension)
    radius = int(0.20 * min(h, w))
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    macular_mask = valid_mask & (dist_from_center <= radius)
    peripheral_mask = valid_mask & (dist_from_center > radius)

    macular_mean = float(np.mean(luminance[macular_mask])) if np.any(macular_mask) else overall_mean
    peripheral_mean = float(np.mean(luminance[peripheral_mask])) if np.any(peripheral_mask) else overall_mean

    # Check glare: fraction of FOV pixels above 240 in green/luminance channel
    glare_pixels = np.count_nonzero((luminance > 240) & valid_mask)
    fov_pixel_count = np.count_nonzero(valid_mask)
    glare_ratio = float(glare_pixels / fov_pixel_count) if fov_pixel_count > 0 else 0.0

    # Underexposure check
    macular_dark_limit = IQA_DARK_THRESHOLD * IQA_MACULAR_DARK_FACTOR
    is_underexposed = (overall_mean < IQA_DARK_THRESHOLD) or (macular_mean < macular_dark_limit)

    # Overexposure check
    periph_bright_limit = IQA_BRIGHT_THRESHOLD * IQA_PERIPHERAL_BRIGHT_FACTOR
    is_overexposed = (overall_mean > IQA_BRIGHT_THRESHOLD) or (peripheral_mean > periph_bright_limit)

    # Glare check
    has_severe_glare = glare_ratio > IQA_GLARE_MAX_RATIO

    return {
        "overall_luminance_mean": overall_mean,
        "macular_luminance_mean": macular_mean,
        "peripheral_luminance_mean": peripheral_mean,
        "glare_ratio": glare_ratio,
        "is_underexposed": is_underexposed,
        "is_overexposed": is_overexposed,
        "has_severe_glare": has_severe_glare,
    }
