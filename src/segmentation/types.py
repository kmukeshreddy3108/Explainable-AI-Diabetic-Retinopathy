"""
Data structures for retinal segmentation and lesion detection.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class LesionCounts:
    """Quantitative breakdown of detected retinal lesions."""

    microaneurysms: int = 0
    hard_exudates: int = 0
    hemorrhages: int = 0
    cotton_wool_spots: int = 0
    neovascularization_clusters: int = 0

    def total_lesion_count(self) -> int:
        return (
            self.microaneurysms
            + self.hard_exudates
            + self.hemorrhages
            + self.cotton_wool_spots
            + self.neovascularization_clusters
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "microaneurysms": self.microaneurysms,
            "hard_exudates": self.hard_exudates,
            "hemorrhages": self.hemorrhages,
            "cotton_wool_spots": self.cotton_wool_spots,
            "neovascularization_clusters": self.neovascularization_clusters,
            "total": self.total_lesion_count(),
        }


@dataclass
class LesionEvidence:
    """
    Complete structural and lesion segmentation evidence extracted from fundus image.
    """

    optic_disc_center: Optional[Tuple[int, int]] = None  # (cx, cy)
    optic_disc_radius: int = 0
    optic_disc_mask: Optional[np.ndarray] = None

    fovea_center: Optional[Tuple[int, int]] = None  # (cx, cy)
    fovea_mask: Optional[np.ndarray] = None

    vessel_mask: Optional[np.ndarray] = None
    vessel_density: float = 0.0

    counts: LesionCounts = field(default_factory=LesionCounts)

    ma_mask: Optional[np.ndarray] = None
    exudate_mask: Optional[np.ndarray] = None
    hemorrhage_mask: Optional[np.ndarray] = None
    nv_mask: Optional[np.ndarray] = None
    combined_lesion_mask: Optional[np.ndarray] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Summary dict for API or UI rendering (omits raw numpy arrays)."""
        return {
            "optic_disc": {
                "center": self.optic_disc_center,
                "radius": self.optic_disc_radius,
            },
            "fovea": {
                "center": self.fovea_center,
            },
            "vessel_density": round(float(self.vessel_density), 4),
            "lesion_counts": self.counts.to_dict(),
        }
