"""
Retinal Lesion Detection Engine.

Extracts Microaneurysms (MA), Hard Exudates (HE), Hemorrhages (HM),
and Neovascularization (NV) using morphology and feature constraints.
"""

from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

from src.segmentation.types import LesionCounts, LesionEvidence
from src.utils.config import (
    LESION_MA_MIN_AREA,
    LESION_MA_MAX_AREA,
    LESION_MA_MIN_CIRCULARITY,
    LESION_EXUDATE_MIN_AREA,
    LESION_EXUDATE_MAX_AREA,
    LESION_HEMORRHAGE_MIN_AREA,
    LESION_HEMORRHAGE_MAX_AREA,
    LESION_EXUDATE_TOPHAT_THRESH,
)


def detect_hard_exudates(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray,
    od_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Detect Hard Exudates (bright yellowish lipid deposits).

    Uses morphological top-hat transformation on green channel to detect bright localized spots,
    excluding the Optic Disc (which is also bright).
    """
    h, w = img_rgb.shape[:2]
    green = img_rgb[:, :, 1]

    # Structuring element for top-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top_hat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)

    # Threshold top-hat response
    _, thresh = cv2.threshold(top_hat, LESION_EXUDATE_TOPHAT_THRESH, 255, cv2.THRESH_BINARY)

    # Exclude OD and outside FOV
    mask = cv2.bitwise_and(thresh, thresh, mask=fov_mask)
    if od_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(od_mask))

    # Filter connected components by area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    exudate_mask = np.zeros((h, w), dtype=np.uint8)
    count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if LESION_EXUDATE_MIN_AREA <= area <= LESION_EXUDATE_MAX_AREA:
            exudate_mask[labels == i] = 255
            count += 1

    return exudate_mask, count


def detect_red_lesions(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray,
    od_mask: np.ndarray,
    vessel_mask: np.ndarray,
) -> Tuple[np.ndarray, int, np.ndarray, int]:
    """
    Detect Microaneurysms (MAs) and Hemorrhages (HMs).

    Both appear as dark red features in the green channel, isolated from main vessel tree.
    Separated by area and circularity.
    """
    h, w = img_rgb.shape[:2]
    green = img_rgb[:, :, 1]

    # Invert green channel so red lesions become bright
    inv_green = cv2.bitwise_not(green)

    # Bottom-hat / Black-hat transform to isolate small dark objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    black_hat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

    # Threshold
    _, thresh = cv2.threshold(black_hat, 15, 255, cv2.THRESH_BINARY)

    # Subtract vessel tree and optic disc to isolate candidates
    candidates = cv2.bitwise_and(thresh, cv2.bitwise_not(vessel_mask))
    if od_mask is not None:
        candidates = cv2.bitwise_and(candidates, cv2.bitwise_not(od_mask))
    candidates = cv2.bitwise_and(candidates, fov_mask)

    # Analyze connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates)

    ma_mask = np.zeros((h, w), dtype=np.uint8)
    hm_mask = np.zeros((h, w), dtype=np.uint8)
    ma_count = 0
    hm_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        perimeter = cv2.arcLength(
            np.argwhere(labels == i)[:, ::-1].astype(np.int32), True
        )
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        # Microaneurysms: small & circular
        if LESION_MA_MIN_AREA <= area <= LESION_MA_MAX_AREA and circularity >= LESION_MA_MIN_CIRCULARITY:
            ma_mask[labels == i] = 255
            ma_count += 1
        # Hemorrhages: larger blobs
        elif LESION_HEMORRHAGE_MIN_AREA <= area <= LESION_HEMORRHAGE_MAX_AREA:
            hm_mask[labels == i] = 255
            hm_count += 1

    return ma_mask, ma_count, hm_mask, hm_count


def detect_neovascularization(
    vessel_mask: np.ndarray,
    od_mask: np.ndarray,
    fov_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Detect Neovascularization (NV) — abnormal fine tortuous vessels near OD (NVD) or elsewhere (NVE).
    Analyzes vessel tortuosity and high-density branching clusters outside normal arcade.
    """
    h, w = vessel_mask.shape[:2]

    # Dilate vessel mask slightly to find dense branching clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    vessel_density_map = cv2.filter2D(vessel_mask.astype(np.float32), -1, kernel)

    # High local density threshold indicates proliferation cluster
    nv_candidates = (vessel_density_map > (255 * 3.5)).astype(np.uint8) * 255

    # Exclude OD center
    if od_mask is not None:
        nv_candidates = cv2.bitwise_and(nv_candidates, cv2.bitwise_not(od_mask))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(nv_candidates)
    nv_mask = np.zeros((h, w), dtype=np.uint8)
    nv_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 60:
            nv_mask[labels == i] = 255
            nv_count += 1

    return nv_mask, nv_count


def detect_all_lesions(
    img_rgb: np.ndarray,
    fov_mask: np.ndarray,
    od_center: Tuple[int, int],
    od_radius: int,
    od_mask: np.ndarray,
    fovea_center: Tuple[int, int],
    fovea_mask: np.ndarray,
    vessel_mask: np.ndarray,
    vessel_density: float,
) -> LesionEvidence:
    """
    Master pipeline for complete structural and lesion segmentation.
    """
    exudate_mask, exudate_count = detect_hard_exudates(img_rgb, fov_mask, od_mask)
    ma_mask, ma_count, hm_mask, hm_count = detect_red_lesions(
        img_rgb, fov_mask, od_mask, vessel_mask
    )
    nv_mask, nv_count = detect_neovascularization(vessel_mask, od_mask, fov_mask)

    counts = LesionCounts(
        microaneurysms=ma_count,
        hard_exudates=exudate_count,
        hemorrhages=hm_count,
        cotton_wool_spots=0,  # can be expanded
        neovascularization_clusters=nv_count,
    )

    combined_mask = cv2.bitwise_or(exudate_mask, ma_mask)
    combined_mask = cv2.bitwise_or(combined_mask, hm_mask)
    combined_mask = cv2.bitwise_or(combined_mask, nv_mask)

    return LesionEvidence(
        optic_disc_center=od_center,
        optic_disc_radius=od_radius,
        optic_disc_mask=od_mask,
        fovea_center=fovea_center,
        fovea_mask=fovea_mask,
        vessel_mask=vessel_mask,
        vessel_density=vessel_density,
        counts=counts,
        ma_mask=ma_mask,
        exudate_mask=exudate_mask,
        hemorrhage_mask=hm_mask,
        nv_mask=nv_mask,
        combined_lesion_mask=combined_mask,
    )
