"""
IQA Gate Module — Primary quality entry-point for the DR screening pipeline.

Combines sharpness, FOV, exposure, and glare analysis into a single decision engine.
Applies automated enhancement for borderline images and returns structured pass/fail results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from src.iqa.sharpness import check_sharpness
from src.iqa.field_of_view import check_fov
from src.iqa.exposure import assess_exposure
from src.iqa.enhance import enhance_fundus_image
from src.utils.config import (
    IQA_SHARPNESS_THRESHOLD,
    IQA_FOV_MIN_RATIO,
    IQA_BORDERLINE_BLUR_FACTOR,
)


@dataclass
class IQAResult:
    """Structured result of Image Quality Assessment."""

    grade: str  # "PASS", "ENHANCED_PASS", "FAIL"
    passable: bool
    sharpness_score: float
    fov_ratio: float
    exposure_metrics: Dict[str, Any]
    rejection_reasons: List[str] = field(default_factory=list)
    enhanced_image: Optional[np.ndarray] = None
    fov_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to clean summary dict (without numpy arrays)."""
        return {
            "grade": self.grade,
            "passable": self.passable,
            "sharpness_score": round(float(self.sharpness_score), 2),
            "fov_ratio": round(float(self.fov_ratio), 4),
            "rejection_reasons": self.rejection_reasons,
            "is_underexposed": self.exposure_metrics.get("is_underexposed", False),
            "is_overexposed": self.exposure_metrics.get("is_overexposed", False),
            "has_severe_glare": self.exposure_metrics.get("has_severe_glare", False),
            "overall_luminance": round(float(self.exposure_metrics.get("overall_luminance_mean", 0.0)), 2),
        }


def assess_and_gate(img_rgb: np.ndarray) -> IQAResult:
    """
    Evaluate fundus image quality and decide whether it can proceed to AI grading.

    Args:
        img_rgb: Input RGB image (H, W, 3)

    Returns:
        IQAResult: Structured assessment object.
    """
    rejection_reasons: List[str] = []

    # 1. FOV Coverage Check
    fov_mask, fov_ratio, fov_pass = check_fov(img_rgb, min_ratio=IQA_FOV_MIN_RATIO)
    if not fov_pass:
        rejection_reasons.append(
            f"Insufficient retinal field of view ({fov_ratio*100:.1f}% < {IQA_FOV_MIN_RATIO*100:.1f}% threshold)."
        )

    # 2. Sharpness / Blur Check
    sharpness_score, sharpness_pass = check_sharpness(
        img_rgb, threshold=IQA_SHARPNESS_THRESHOLD
    )
    if not sharpness_pass:
        rejection_reasons.append(
            f"Severe blur detected (sharpness score {sharpness_score:.1f} < threshold {IQA_SHARPNESS_THRESHOLD})."
        )

    # 3. Exposure and Glare Check
    exposure_metrics = assess_exposure(img_rgb, fov_mask)
    if exposure_metrics["is_underexposed"]:
        rejection_reasons.append(
            f"Image is severely underexposed (mean luminance {exposure_metrics['overall_luminance_mean']:.1f})."
        )
    if exposure_metrics["is_overexposed"]:
        rejection_reasons.append(
            f"Image is severely overexposed (mean luminance {exposure_metrics['overall_luminance_mean']:.1f})."
        )
    if exposure_metrics["has_severe_glare"]:
        rejection_reasons.append(
            f"Specular glare artifact detected ({exposure_metrics['glare_ratio']*100:.1f}% of FOV is saturated)."
        )

    # Determine Grade and Passable Status
    if len(rejection_reasons) == 0:
        grade = "PASS"
        passable = True
        enhanced_image = img_rgb.copy()
    else:
        # Check if borderline (usable if enhanced)
        # Borderline condition: sharpness is within borderline factor and FOV is acceptable
        borderline_sharpness = sharpness_score >= (
            IQA_SHARPNESS_THRESHOLD / IQA_BORDERLINE_BLUR_FACTOR
        )
        borderline_fov = fov_ratio >= 0.45
        glare_ok = exposure_metrics["glare_ratio"] < 0.08

        if borderline_sharpness and borderline_fov and glare_ok:
            grade = "ENHANCED_PASS"
            passable = True
            enhanced_image = enhance_fundus_image(img_rgb, method="ben_graham")
            # Note enhancement in reasons
            rejection_reasons.insert(
                0, "Borderline quality image recovered using Ben Graham color normalization."
            )
        else:
            grade = "FAIL"
            passable = False
            enhanced_image = None

    return IQAResult(
        grade=grade,
        passable=passable,
        sharpness_score=sharpness_score,
        fov_ratio=fov_ratio,
        exposure_metrics=exposure_metrics,
        rejection_reasons=rejection_reasons,
        enhanced_image=enhanced_image,
        fov_mask=fov_mask,
    )
