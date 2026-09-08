"""
Clinical Rule Engine & Hybrid Fusion Classifier for ICDR DR Grading.

Combines deep neural network probabilities with deterministic clinical rules
derived from extracted structural lesion evidence.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from src.segmentation.types import LesionEvidence
from src.utils.config import (
    GRADING_NN_WEIGHT,
    GRADING_RULE_WEIGHT,
    ICDR_LABELS,
    ICDR_DESCRIPTIONS,
)


@dataclass
class GradingResult:
    """Final output of the DR grading pipeline."""

    predicted_grade: int  # 0–4
    grade_label: str  # e.g., "Moderate NPDR"
    description: str
    confidence: float  # 0.0 – 1.0
    is_referable: bool  # True if grade >= 2
    risk_level: str  # "Low", "Moderate", "High"
    probabilities: np.ndarray  # shape (5,) soft probabilities
    nn_probabilities: np.ndarray
    rule_probabilities: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_grade": self.predicted_grade,
            "grade_label": self.grade_label,
            "description": self.description,
            "confidence": round(float(self.confidence), 4),
            "is_referable": self.is_referable,
            "risk_level": self.risk_level,
            "probabilities": [round(float(p), 4) for p in self.probabilities],
        }


def compute_rule_probabilities(evidence: LesionEvidence) -> np.ndarray:
    """
    Compute clinical rule soft probability distribution based on ICDR diagnostic criteria.

    ICDR Rules:
    - Grade 0: 0 MAs, 0 Exudates, 0 HMs, 0 NV
    - Grade 1: Microaneurysms only (>0 MAs, 0 Exudates, 0 HMs, 0 NV)
    - Grade 2: More than MAs, but less than Severe (Exudates > 0 or 1-5 HMs)
    - Grade 3: Severe NPDR (>5 HMs / extensive lesions)
    - Grade 4: Proliferative DR (NV clusters > 0)
    """
    counts = evidence.counts
    probs = np.zeros(5, dtype=np.float32)

    if counts.neovascularization_clusters > 0:
        # Strong evidence for Grade 4 (PDR)
        probs = np.array([0.01, 0.02, 0.05, 0.12, 0.80], dtype=np.float32)
    elif counts.hemorrhages > 5:
        # Strong evidence for Grade 3 (Severe NPDR)
        probs = np.array([0.01, 0.04, 0.15, 0.70, 0.10], dtype=np.float32)
    elif counts.hard_exudates > 0 or (1 <= counts.hemorrhages <= 5):
        # Evidence for Grade 2 (Moderate NPDR)
        probs = np.array([0.02, 0.15, 0.70, 0.10, 0.03], dtype=np.float32)
    elif counts.microaneurysms > 0:
        # Evidence for Grade 1 (Mild NPDR)
        probs = np.array([0.10, 0.75, 0.12, 0.02, 0.01], dtype=np.float32)
    else:
        # No detected lesions -> Grade 0 (No DR)
        probs = np.array([0.85, 0.10, 0.03, 0.01, 0.01], dtype=np.float32)

    return probs / np.sum(probs)


def fuse_hybrid_grading(
    nn_probs: np.ndarray,
    evidence: LesionEvidence,
    w_nn: float = GRADING_NN_WEIGHT,
    w_rule: float = GRADING_RULE_WEIGHT,
) -> GradingResult:
    """
    Fuse neural network predictions with clinical rule evidence.

    Args:
        nn_probs: Array of shape (5,) with softmax probabilities from NN model
        evidence: Extracted lesion evidence from Module 2

    Returns:
        GradingResult object
    """
    nn_probs = np.asarray(nn_probs, dtype=np.float32).ravel()
    rule_probs = compute_rule_probabilities(evidence)

    # Weighted linear fusion
    fused_probs = w_nn * nn_probs + w_rule * rule_probs
    fused_probs = fused_probs / np.sum(fused_probs)

    predicted_grade = int(np.argmax(fused_probs))
    confidence = float(fused_probs[predicted_grade])

    grade_label = ICDR_LABELS.get(predicted_grade, "Unknown")
    description = ICDR_DESCRIPTIONS.get(predicted_grade, "")

    # Clinical referability: Grade >= 2 is referable to ophthalmologist
    is_referable = predicted_grade >= 2

    # Risk level classification
    if predicted_grade == 0:
        risk_level = "Low"
    elif predicted_grade == 1:
        risk_level = "Low-Moderate"
    elif predicted_grade == 2:
        risk_level = "Moderate"
    elif predicted_grade == 3:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return GradingResult(
        predicted_grade=predicted_grade,
        grade_label=grade_label,
        description=description,
        confidence=confidence,
        is_referable=is_referable,
        risk_level=risk_level,
        probabilities=fused_probs,
        nn_probabilities=nn_probs,
        rule_probabilities=rule_probs,
    )
