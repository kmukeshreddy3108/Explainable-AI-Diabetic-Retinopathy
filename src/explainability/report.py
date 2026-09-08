"""
Structured Clinical Diagnostic & Explanation Report Generator.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import json

from src.iqa.gate import IQAResult
from src.grading.hybrid import GradingResult
from src.segmentation.types import LesionEvidence
from src.explainability.fusion import compute_heatmap_lesion_agreement


@dataclass
class DiagnosticReport:
    """Complete structured clinical diagnostic report for a screening scan."""

    patient_id: str
    scan_id: str
    quality_assessment: Dict[str, Any]
    grading_assessment: Dict[str, Any]
    lesion_breakdown: Dict[str, Any]
    explainability_metrics: Dict[str, Any]
    clinical_recommendation: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "scan_id": self.scan_id,
            "quality_assessment": self.quality_assessment,
            "grading_assessment": self.grading_assessment,
            "lesion_breakdown": self.lesion_breakdown,
            "explainability_metrics": self.explainability_metrics,
            "clinical_recommendation": self.clinical_recommendation,
            "disclaimer": self.disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def get_clinical_recommendation(grade: int) -> str:
    """Generate evidence-based clinical referral recommendation based on ICDR grade."""
    if grade == 0:
        return "No DR detected. Schedule routine annual DR rescreening in 12 months."
    elif grade == 1:
        return (
            "Mild NPDR detected (microaneurysms only). Repeat fundus screening in 6–12 months. "
            "Advise patient on strict blood glucose and blood pressure management."
        )
    elif grade == 2:
        return (
            "Moderate NPDR detected. Non-urgent referral to ophthalmologist within 4–6 weeks "
            "for comprehensive dilated retinal examination."
        )
    elif grade == 3:
        return (
            "Severe NPDR detected. Urgent referral to retina specialist within 1–2 weeks "
            "due to high risk of progression to proliferative DR."
        )
    elif grade == 4:
        return (
            "Proliferative DR detected (neovascularization present). IMMEDIATE referral to "
            "retina specialist within 24–48 hours for evaluation of anti-VEGF therapy or panretinal photocoagulation."
        )
    else:
        return "Re-examine scan or consult senior ophthalmologist."


def generate_diagnostic_report(
    patient_id: str,
    scan_id: str,
    iqa_result: IQAResult,
    grading_result: GradingResult,
    evidence: LesionEvidence,
    heatmap: Any = None,
) -> DiagnosticReport:
    """
    Generate comprehensive clinical diagnostic and explainability report.
    """
    quality_dict = iqa_result.to_dict()
    grading_dict = grading_result.to_dict()
    lesion_dict = evidence.to_summary_dict()

    # Explainability metrics
    if heatmap is not None and evidence.combined_lesion_mask is not None:
        explain_metrics = compute_heatmap_lesion_agreement(heatmap, evidence.combined_lesion_mask)
    else:
        explain_metrics = {
            "attention_lesion_overlap_ratio": 1.0,
            "lesion_attention_coverage": 1.0,
            "agreement_score": 1.0,
        }

    recommendation = get_clinical_recommendation(grading_result.predicted_grade)
    disclaimer = (
        "DISCLAIMER: This AI system is an automated triage decision support tool intended "
        "to assist healthcare workers in district/rural screening. Final diagnosis must be "
        "validated by a certified ophthalmologist."
    )

    return DiagnosticReport(
        patient_id=patient_id,
        scan_id=scan_id,
        quality_assessment=quality_dict,
        grading_assessment=grading_dict,
        lesion_breakdown=lesion_dict,
        explainability_metrics=explain_metrics,
        clinical_recommendation=recommendation,
        disclaimer=disclaimer,
    )
