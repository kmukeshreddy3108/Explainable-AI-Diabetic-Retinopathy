"""
Spatial Fusion Module — Fuses Neural Grad-CAM Attention with Morphological Lesion Masks.

Calculates quantitative agreement metrics between deep neural attention regions and
detected structural lesions (MAs, Exudates, Hemorrhages, NV).
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np

from src.segmentation.types import LesionEvidence
from src.explainability.gradcam import overlay_heatmap
from src.utils.config import GRADCAM_ATTENTION_THRESHOLD, GRADCAM_HEATMAP_ALPHA


def compute_heatmap_lesion_agreement(
    heatmap: np.ndarray,
    lesion_mask: np.ndarray,
    attention_thresh: float = GRADCAM_ATTENTION_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculate spatial overlap between Grad-CAM neural attention and segmented lesions.

    Returns:
        Dict containing:
            - attention_lesion_overlap_ratio: float (0.0 to 1.0)
            - lesion_attention_coverage: float (0.0 to 1.0)
            - agreement_score: float (0.0 to 1.0)
    """
    h, w = heatmap.shape[:2]
    if lesion_mask.shape[:2] != (h, w):
        lesion_mask = cv2.resize(lesion_mask, (w, h))

    attention_bin = heatmap >= attention_thresh
    lesion_bin = lesion_mask > 0

    attention_pixels = np.count_nonzero(attention_bin)
    lesion_pixels = np.count_nonzero(lesion_bin)
    overlap_pixels = np.count_nonzero(attention_bin & lesion_bin)

    if lesion_pixels > 0:
        lesion_attention_coverage = float(overlap_pixels / lesion_pixels)
    else:
        lesion_attention_coverage = 1.0 if attention_pixels == 0 else 0.5

    if attention_pixels > 0:
        attention_lesion_overlap_ratio = float(overlap_pixels / attention_pixels)
    else:
        attention_lesion_overlap_ratio = 1.0 if lesion_pixels == 0 else 0.5

    agreement_score = 0.5 * (lesion_attention_coverage + attention_lesion_overlap_ratio)

    return {
        "attention_lesion_overlap_ratio": round(attention_lesion_overlap_ratio, 4),
        "lesion_attention_coverage": round(lesion_attention_coverage, 4),
        "agreement_score": round(agreement_score, 4),
    }


def create_composite_explanation_panel(
    img_rgb: np.ndarray, heatmap: np.ndarray, evidence: LesionEvidence
) -> np.ndarray:
    """
    Generate 2x2 grid composite image for clinical visual evidence:
    - Top-Left: Original RGB Image
    - Top-Right: Grad-CAM Attention Heatmap
    - Bottom-Left: Morphological Lesion Mask Overlay
    - Bottom-Right: Fused Attention x Lesion Overlap
    """
    h, w = img_rgb.shape[:2]

    # 1. Original
    p1 = img_rgb.copy()

    # 2. Grad-CAM Overlay
    p2 = overlay_heatmap(img_rgb, heatmap, alpha=GRADCAM_HEATMAP_ALPHA)

    # 3. Lesion Overlay (Red for HMs/MAs, Yellow for Exudates, Cyan for NV)
    p3 = img_rgb.copy()
    if evidence.exudate_mask is not None and np.any(evidence.exudate_mask > 0):
        p3[evidence.exudate_mask > 0] = [255, 255, 0]  # Yellow
    if evidence.hemorrhage_mask is not None and np.any(evidence.hemorrhage_mask > 0):
        p3[evidence.hemorrhage_mask > 0] = [255, 0, 0]  # Red
    if evidence.ma_mask is not None and np.any(evidence.ma_mask > 0):
        p3[evidence.ma_mask > 0] = [255, 50, 50]  # Bright Red
    if evidence.nv_mask is not None and np.any(evidence.nv_mask > 0):
        p3[evidence.nv_mask > 0] = [0, 255, 255]  # Cyan

    # 4. Fused Map (Heatmap modulated by Lesion evidence)
    p4 = overlay_heatmap(p3, heatmap, alpha=0.5)

    # Assemble 2x2 grid
    top_row = np.hstack((p1, p2))
    bottom_row = np.hstack((p3, p4))
    composite = np.vstack((top_row, bottom_row))

    return composite
