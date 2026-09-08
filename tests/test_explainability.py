"""
Unit tests for Module 4 — Explainability, Grad-CAM & Diagnostic Report Generation.
"""

import pytest
import numpy as np
import torch

from src.grading.model import build_dr_model
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.explainability.fusion import compute_heatmap_lesion_agreement, create_composite_explanation_panel
from src.explainability.report import generate_diagnostic_report, DiagnosticReport
from src.iqa.gate import IQAResult
from src.grading.hybrid import GradingResult
from src.segmentation.types import LesionEvidence, LesionCounts


def test_gradcam_generation():
    model = build_dr_model(pretrained=False)
    cam = GradCAM(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, target_cls = cam.generate_heatmap(dummy_input, target_class=2)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0
    assert target_cls == 2


def test_overlay_heatmap():
    img_rgb = np.full((200, 200, 3), 128, dtype=np.uint8)
    heatmap = np.random.rand(200, 200).astype(np.float32)

    overlay = overlay_heatmap(img_rgb, heatmap, alpha=0.4)

    assert overlay.shape == img_rgb.shape
    assert overlay.dtype == np.uint8


def test_spatial_fusion_agreement():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[30:70, 30:70] = 0.8  # Center high attention

    lesion_mask = np.zeros((100, 100), dtype=np.uint8)
    lesion_mask[40:60, 40:60] = 255  # Lesion inside attention region

    metrics = compute_heatmap_lesion_agreement(heatmap, lesion_mask, attention_thresh=0.5)

    assert metrics["lesion_attention_coverage"] == 1.0
    assert metrics["attention_lesion_overlap_ratio"] > 0.0
    assert metrics["agreement_score"] > 0.5


def test_diagnostic_report_generation():
    iqa = IQAResult(
        grade="PASS",
        passable=True,
        sharpness_score=250.0,
        fov_ratio=0.85,
        exposure_metrics={"is_underexposed": False, "is_overexposed": False, "has_severe_glare": False},
        rejection_reasons=[],
    )

    grading = GradingResult(
        predicted_grade=2,
        grade_label="Moderate NPDR",
        description="Exudates present",
        confidence=0.88,
        is_referable=True,
        risk_level="Moderate",
        probabilities=np.array([0.05, 0.05, 0.88, 0.01, 0.01]),
        nn_probabilities=np.array([0.05, 0.05, 0.88, 0.01, 0.01]),
        rule_probabilities=np.array([0.05, 0.05, 0.88, 0.01, 0.01]),
    )

    evidence = LesionEvidence(counts=LesionCounts(hard_exudates=3))
    heatmap = np.zeros((100, 100), dtype=np.float32)

    report = generate_diagnostic_report("PATIENT_001", "SCAN_101", iqa, grading, evidence, heatmap)

    assert isinstance(report, DiagnosticReport)
    assert report.patient_id == "PATIENT_001"
    assert report.grading_assessment["predicted_grade"] == 2
    assert "referral" in report.clinical_recommendation.lower()

    json_str = report.to_json()
    assert "PATIENT_001" in json_str
