"""
Vessel Segmentation module using Multi-Orientation Gabor Filter Bank.

Extracts blood vessel tree from the green channel. Includes DRIVE benchmark evaluation metrics
(Dice coefficient, Jaccard index, Sensitivity, Specificity).
"""

from typing import Tuple, Dict, Optional
import cv2
import numpy as np

from src.utils.config import (
    GABOR_WAVELENGTH,
    GABOR_SIGMA,
    GABOR_NUM_ORIENTATIONS,
)


def build_gabor_filter_bank(
    ksize: int = 15,
    wavelength: float = GABOR_WAVELENGTH,
    sigma: float = GABOR_SIGMA,
    num_orientations: int = GABOR_NUM_ORIENTATIONS,
) -> list:
    """Construct bank of 2D Gabor kernels tuned for elongated linear structures (vessels)."""
    filters = []
    gamma = 0.5  # spatial aspect ratio (elongated kernel)
    psi = 0.0

    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, wavelength, gamma, psi, ktype=cv2.CV_32F
        )
        filters.append(kernel)
    return filters


def segment_vessels(
    img_rgb: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    od_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Segment blood vessel tree using Gabor filter bank on green channel.

    Args:
        img_rgb: Input RGB image (H, W, 3)
        fov_mask: Optional FOV mask (H, W) uint8
        od_mask: Optional Optic Disc mask to exclude disc rim from vessel count

    Returns:
        (vessel_mask, vessel_density)
    """
    h, w = img_rgb.shape[:2]
    if fov_mask is None:
        fov_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Extract green channel and invert (vessels are dark on bright background)
    green = img_rgb[:, :, 1]
    inverted_green = cv2.bitwise_not(green)

    # CLAHE on inverted green channel to enhance vessel contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted_green)

    # Apply Gabor filter bank
    filters = build_gabor_filter_bank()
    max_gabor_response = np.zeros((h, w), dtype=np.float32)

    for kernel in filters:
        filtered = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
        np.maximum(max_gabor_response, filtered, out=max_gabor_response)

    # Normalize response to 0..255
    norm_response = cv2.normalize(
        max_gabor_response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Threshold response
    _, thresh = cv2.threshold(norm_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply FOV mask
    vessel_mask = cv2.bitwise_and(thresh, thresh, mask=fov_mask)

    # Optionally exclude optic disc region to prevent false vessel count at OD rim
    if od_mask is not None:
        vessel_mask = cv2.bitwise_and(vessel_mask, cv2.bitwise_not(od_mask))

    # Calculate vessel density inside FOV
    fov_pixels = np.count_nonzero(fov_mask)
    vessel_pixels = np.count_nonzero(vessel_mask)
    vessel_density = float(vessel_pixels / fov_pixels) if fov_pixels > 0 else 0.0

    return vessel_mask, vessel_density


def evaluate_vessel_benchmark(
    pred_mask: np.ndarray, ground_truth_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate segmentation accuracy metrics against ground truth (e.g., DRIVE dataset benchmark).

    Metrics calculated:
    - Dice Coefficient (F1 Score)
    - Jaccard Index (IoU)
    - Sensitivity (Recall)
    - Specificity
    - Accuracy
    """
    pred_bin = (pred_mask > 0).astype(bool)
    gt_bin = (ground_truth_mask > 0).astype(bool)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "dice": round(float(dice), 4),
        "jaccard": round(float(jaccard), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "accuracy": round(float(accuracy), 4),
    }
