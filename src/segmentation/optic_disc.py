"""
Optic Disc and Fovea localization module.

Uses intensity morphological operations and circular Hough transform (or centroid of max brightness)
to detect the Optic Disc, then estimates Fovea position based on anatomical geometry.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from src.utils.config import SEG_DISC_RADIUS_FRAC, SEG_FOVEA_OFFSET_FRAC, SEG_FOVEA_RADIUS_FRAC


def detect_optic_disc(
    img_rgb: np.ndarray, fov_mask: Optional[np.ndarray] = None
) -> Tuple[Tuple[int, int], int, np.ndarray]:
    """
    Detect Optic Disc center coordinates (cx, cy), radius, and binary mask.

    Args:
        img_rgb: Input RGB image (H, W, 3)
        fov_mask: Optional FOV mask (H, W) uint8

    Returns:
        (center_xy, radius, od_mask)
    """
    h, w = img_rgb.shape[:2]
    if fov_mask is None:
        fov_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Red channel (or Gray) usually has highest intensity at optic disc
    red_channel = img_rgb[:, :, 0]
    masked_red = cv2.bitwise_and(red_channel, red_channel, mask=fov_mask)

    # Apply morphological closing with large structuring element to smooth vessels inside OD
    disc_radius_est = max(10, int(min(h, w) * SEG_DISC_RADIUS_FRAC))
    kernel_size = max(5, disc_radius_est | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(masked_red, cv2.MORPH_CLOSE, kernel)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(closed, (15, 15), 0)

    # Locate maximum intensity region
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred, mask=fov_mask)

    # Refine center using circular Hough transform in region around max_loc
    cx, cy = max_loc
    search_r = disc_radius_est * 2
    x1, x2 = max(0, cx - search_r), min(w, cx + search_r)
    y1, y2 = max(0, cy - search_r), min(h, cy + search_r)

    roi = blurred[y1:y2, x1:x2]
    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=disc_radius_est,
        param1=50,
        param2=20,
        minRadius=int(disc_radius_est * 0.6),
        maxRadius=int(disc_radius_est * 1.4),
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = circles[0][0]
        cx = x1 + int(best_circle[0])
        cy = y1 + int(best_circle[1])
        r = int(best_circle[2])
    else:
        r = disc_radius_est

    # Create binary mask for Optic Disc
    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, (cx, cy), r, 255, -1)
    od_mask = cv2.bitwise_and(od_mask, od_mask, mask=fov_mask)

    return (cx, cy), r, od_mask


def estimate_fovea_location(
    img_rgb: np.ndarray,
    od_center: Tuple[int, int],
    od_radius: int,
    fov_mask: Optional[np.ndarray] = None,
) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Estimate Fovea centralis location relative to the Optic Disc center.

    Anatomically, the fovea is located approximately 2.5 disc diameters (~4.5-5.0 mm)
    temporal to the optic disc center, slightly inferior or level with the center.

    Returns:
        (fovea_center_xy, fovea_mask)
    """
    h, w = img_rgb.shape[:2]
    if fov_mask is None:
        fov_mask = np.ones((h, w), dtype=np.uint8) * 255

    cx_od, cy_od = od_center
    offset_dist = int(w * SEG_FOVEA_OFFSET_FRAC)

    # Determine if OD is on Left side or Right side of image
    # If OD is in left half, fovea is to the right (+offset); if OD in right half, fovea is to the left (-offset)
    if cx_od < w // 2:
        target_cx = min(w - 1, cx_od + offset_dist)
    else:
        target_cx = max(0, cx_od - offset_dist)

    target_cy = cy_od  # level with OD

    # Refine fovea center by finding the darkest local region (macula) around target_cx, target_cy
    search_r = int(od_radius * 1.5)
    x1, x2 = max(0, target_cx - search_r), min(w, target_cx + search_r)
    y1, y2 = max(0, target_cy - search_r), min(h, target_cy + search_r)

    green_channel = img_rgb[:, :, 1]
    roi = green_channel[y1:y2, x1:x2]
    if roi.size > 0:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
        final_cx = x1 + min_loc[0]
        final_cy = y1 + min_loc[1]
    else:
        final_cx, final_cy = target_cx, target_cy

    fovea_r = max(5, int(min(h, w) * SEG_FOVEA_RADIUS_FRAC))
    fovea_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fovea_mask, (final_cx, final_cy), fovea_r, 255, -1)
    fovea_mask = cv2.bitwise_and(fovea_mask, fovea_mask, mask=fov_mask)

    return (final_cx, final_cy), fovea_mask
