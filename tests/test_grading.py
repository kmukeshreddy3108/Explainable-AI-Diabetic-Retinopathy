"""
Unit tests for Module 3 — Grading, Clinical Rule Engine & Hybrid Fusion.
"""

import os
import pytest
import numpy as np
import torch

from src.grading.model import build_dr_model, DRGradingModel
from src.grading.hybrid import compute_rule_probabilities, fuse_hybrid_grading, GradingResult
from src.grading.calibrate import calibrate_temperature, TemperatureScaler
from src.grading.train import validate_dataset_role, train_demo_weights
from src.grading.export_onnx import export_to_onnx
from src.grading.compatibility_check import verify_pytorch_onnx_compatibility
from src.segmentation.types import LesionEvidence, LesionCounts


def test_model_forward_pass():
    model = build_dr_model(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    probs = model.predict_probs(x)

    assert logits.shape == (2, 5)
    assert probs.shape == (2, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4)


def test_clinical_rule_and_hybrid_fusion():
    # 1. Test Grade 0 (no lesions)
    ev_0 = LesionEvidence(counts=LesionCounts(microaneurysms=0, hard_exudates=0, hemorrhages=0))
    rule_probs_0 = compute_rule_probabilities(ev_0)
    assert np.argmax(rule_probs_0) == 0

    # 2. Test Grade 4 (PDR with NV)
    ev_4 = LesionEvidence(counts=LesionCounts(neovascularization_clusters=2))
    rule_probs_4 = compute_rule_probabilities(ev_4)
    assert np.argmax(rule_probs_4) == 4

    # 3. Test Hybrid Fusion
    nn_probs = np.array([0.1, 0.2, 0.6, 0.1, 0.0], dtype=np.float32)
    ev_2 = LesionEvidence(counts=LesionCounts(hard_exudates=3))
    res = fuse_hybrid_grading(nn_probs, ev_2)

    assert isinstance(res, GradingResult)
    assert res.predicted_grade in [1, 2]
    assert res.is_referable is True
    assert 0.0 <= res.confidence <= 1.0


def test_dataset_role_constraints():
    # Valid usage
    validate_dataset_role("APTOS_2019", "train")

    # Invalid usage: Messidor-2 for training must fail!
    with pytest.raises(RuntimeError) as exc_info:
        validate_dataset_role("Messidor_2", "train")
    assert "VIOLATION" in str(exc_info.value)


def test_temperature_calibration():
    logits = torch.randn(20, 5) * 5.0  # high variance logits
    labels = torch.randint(0, 5, (20,))

    opt_temp = calibrate_temperature(logits, labels, max_iter=20)
    assert 0.1 <= opt_temp <= 10.0


def test_onnx_export_and_compatibility(tmp_path):
    model = build_dr_model(pretrained=False)
    onnx_file = str(tmp_path / "test_model.onnx")

    exported_path = export_to_onnx(model, output_path=onnx_file)
    assert os.path.exists(exported_path)

    res = verify_pytorch_onnx_compatibility(model, exported_path, tolerance=1e-3)
    assert res["max_diff"] <= 1e-3
    assert res["is_compatible"] is True
