"""
Unit tests for Module 2 — Segmentation & Retinal Feature Extraction.
"""

import pytest
import cv2
import numpy as np

from src.utils.sample_generator import generate_synthetic_fundus
from src.iqa.field_of_view import extract_fov_mask
from src.segmentation.types import LesionEvidence, LesionCounts
from src.segmentation.optic_disc import detect_optic_disc, estimate_fovea_location
from src.segmentation.vessels import segment_vessels, evaluate_vessel_benchmark
from src.segmentation.lesions import (
    detect_hard_exudates,
    detect_red_lesions,
    detect_neovascularization,
    detect_all_lesions,
)


def get_rgb_sample(grade: int = 0, seed: int = 42) -> np.ndarray:
    bgr = generate_synthetic_fundus(grade=grade, quality="good", seed=seed)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_optic_disc_and_fovea_detection():
    img = get_rgb_sample(grade=0, seed=12)
    fov_mask, _ = extract_fov_mask(img)

    (cx, cy), radius, od_mask = detect_optic_disc(img, fov_mask)
    fovea_xy, fovea_mask = estimate_fovea_location(img, (cx, cy), radius, fov_mask)

    assert 0 <= cx < img.shape[1]
    assert 0 <= cy < img.shape[0]
    assert radius > 0
    assert od_mask.shape == fov_mask.shape
    assert fovea_xy[0] >= 0 and fovea_xy[1] >= 0
    assert fovea_mask.shape == fov_mask.shape


def test_vessel_segmentation():
    img = get_rgb_sample(grade=0, seed=15)
    fov_mask, _ = extract_fov_mask(img)

    vessel_mask, density = segment_vessels(img, fov_mask)

    assert vessel_mask.shape == fov_mask.shape
    assert 0.0 <= density <= 1.0


def test_vessel_benchmark_metrics():
    # Synthetic ground truth vs predicted mask
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[30:70, 45:55] = 255  # Vertical stripe

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[32:68, 45:55] = 255  # Slightly shorter stripe

    metrics = evaluate_vessel_benchmark(pred, gt)

    assert "dice" in metrics
    assert "jaccard" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert metrics["dice"] > 0.80
    assert metrics["accuracy"] > 0.90


def test_lesion_detection_pipeline():
    # Grade 3 (severe NPDR sample with synthetic lesions)
    img = get_rgb_sample(grade=3, seed=99)
    fov_mask, _ = extract_fov_mask(img)

    (cx, cy), radius, od_mask = detect_optic_disc(img, fov_mask)
    fovea_xy, fovea_mask = estimate_fovea_location(img, (cx, cy), radius, fov_mask)
    vessel_mask, density = segment_vessels(img, fov_mask, od_mask)

    evidence = detect_all_lesions(
        img_rgb=img,
        fov_mask=fov_mask,
        od_center=(cx, cy),
        od_radius=radius,
        od_mask=od_mask,
        fovea_center=fovea_xy,
        fovea_mask=fovea_mask,
        vessel_mask=vessel_mask,
        vessel_density=density,
    )

    assert isinstance(evidence, LesionEvidence)
    assert isinstance(evidence.counts, LesionCounts)
    summary = evidence.to_summary_dict()
    assert "lesion_counts" in summary
    assert summary["lesion_counts"]["total"] >= 0
