"""
Image enhancement techniques for retinal fundus images.

Provides:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Ben Graham Color Normalization
- Homomorphic / Illumination correction filtering
- Full enhancement pipeline
"""

import cv2
import numpy as np

from src.utils.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, BEN_GRAHAM_SIGMA


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = CLAHE_CLIP_LIMIT) -> np.ndarray:
    """
    Apply CLAHE to the L channel of LAB color space to enhance contrast without color distortion.

    Args:
        img_rgb: Input RGB image (H, W, 3)
        clip_limit: Threshold for contrast limiting.

    Returns:
        Enhanced RGB image (H, W, 3)
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILE_SIZE)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def apply_ben_graham(img_rgb: np.ndarray, sigma: float = BEN_GRAHAM_SIGMA) -> np.ndarray:
    """
    Apply Ben Graham color normalization method (Kaggle DR competition winning technique).
    Subtracts local mean brightness using a Gaussian filter to normalize illumination across fundus images.

    formula: img_norm = 4 * img - 4 * GaussianBlur(img, sigma) + 128
    """
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
    enhanced = cv2.addWeighted(img_rgb, 4.0, blurred, -4.0, 128)

    # Crop to circle mask if possible, or clip values
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def apply_homomorphic_filter(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply homomorphic filtering in frequency domain to reduce illumination variation
    while sharpening reflectance (lesions/vessels).
    """
    # Work on luminance L channel
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    # Take log of image
    l_log = np.log1p(np.float32(l_channel))

    # Low pass filter using Gaussian blur
    l_blur = cv2.GaussianBlur(l_log, (35, 35), 0)

    # High pass = log - low pass
    l_highpass = l_log - l_blur

    # Scale high pass (reflectance) and low pass (illumination)
    gamma_l = 0.5  # reduce illumination
    gamma_h = 1.5  # boost details
    l_filtered = gamma_l * l_blur + gamma_h * l_highpass

    # Exponentiate back
    l_exp = np.expm1(l_filtered)
    l_out = np.clip(l_exp, 0, 255).astype(np.uint8)

    merged = cv2.merge((l_out, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def enhance_fundus_image(img_rgb: np.ndarray, method: str = "ben_graham") -> np.ndarray:
    """
    Unified enhancement wrapper.

    Args:
        img_rgb: Input image
        method: "clahe", "ben_graham", or "homomorphic"
    """
    if method == "ben_graham":
        return apply_ben_graham(img_rgb)
    elif method == "clahe":
        return apply_clahe(img_rgb)
    elif method == "homomorphic":
        return apply_homomorphic_filter(img_rgb)
    else:
        # Default to combined CLAHE + Ben Graham
        clahe_img = apply_clahe(img_rgb)
        return apply_ben_graham(clahe_img)
