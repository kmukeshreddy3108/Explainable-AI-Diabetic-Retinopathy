"""
Field of View (FOV) mask extraction and coverage check.

Uses Otsu thresholding on the Green channel (which has highest contrast for retinal structures)
to locate the circular fundus aperture and measure coverage ratio against total image frame.
"""

from typing import Tuple
import cv2
import numpy as np

from src.utils.config import IQA_FOV_MIN_RATIO


def extract_fov_mask(img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract binary FOV mask using green channel thresholding and morph closing.

    Args:
        img_rgb: Input RGB image (H, W, 3)

    Returns:
        mask: Binary mask (H, W) uint8 with 255 for retinal tissue, 0 for background.
        fov_ratio: Fraction of total frame covered by valid fundus tissue.
    """
    if img_rgb.ndim == 3:
        green_channel = img_rgb[:, :, 1]
    else:
        green_channel = img_rgb

    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)

    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill small internal holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component (the fundus aperture)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels > 1:
        # Index 0 is background, find max area among non-zero labels
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest_label) * 255
    else:
        mask = closed

    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    fov_pixels = np.count_nonzero(mask)
    fov_ratio = float(fov_pixels / total_pixels) if total_pixels > 0 else 0.0

    return mask, fov_ratio


def check_fov(
    img_rgb: np.ndarray, min_ratio: float = IQA_FOV_MIN_RATIO
) -> Tuple[np.ndarray, float, bool]:
    """
    Check if fundus image meets minimum FOV coverage requirement.

    Returns:
        (mask, ratio, passes_fov_check)
    """
    mask, ratio = extract_fov_mask(img_rgb)
    passes = ratio >= min_ratio
    return mask, ratio, passes
