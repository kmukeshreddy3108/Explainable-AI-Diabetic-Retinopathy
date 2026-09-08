"""
Unit tests for Module 1 — Image Quality Assessment & Enhancement.
"""

import pytest
import cv2
import numpy as np

from src.utils.sample_generator import generate_synthetic_fundus
from src.iqa.sharpness import check_sharpness, compute_sharpness_score
from src.iqa.field_of_view import check_fov, extract_fov_mask
from src.iqa.exposure import assess_exposure
from src.iqa.enhance import apply_clahe, apply_ben_graham, enhance_fundus_image
from src.iqa.gate import assess_and_gate, IQAResult


def get_rgb_sample(grade: int = 0, quality: str = "good", seed: int = 42) -> np.ndarray:
    bgr = generate_synthetic_fundus(grade=grade, quality=quality, seed=seed)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_sharpness_good_vs_blurry():
    clear_img = get_rgb_sample(grade=0, quality="good", seed=42)
    blurry_img = get_rgb_sample(grade=0, quality="blurry", seed=42)

    score_clear, is_sharp_clear = check_sharpness(clear_img)
    score_blurry, is_sharp_blurry = check_sharpness(blurry_img)

    assert score_clear > score_blurry, "Clear image must have higher Laplacian variance than blurry image"
    assert is_sharp_clear is True
    assert is_sharp_blurry is False


def test_fov_extraction():
    img = get_rgb_sample(grade=0, quality="good", seed=10)
    mask, ratio, passes = check_fov(img, min_ratio=0.50)

    assert mask.ndim == 2
    assert mask.shape[:2] == img.shape[:2]
    assert ratio >= 0.50
    assert passes is True


def test_exposure_assessment():
    normal_img = get_rgb_sample(grade=0, quality="good", seed=1)
    dark_img = get_rgb_sample(grade=0, quality="dark", seed=1)

    fov_mask, _ = extract_fov_mask(normal_img)

    exp_normal = assess_exposure(normal_img, fov_mask)
    exp_dark = assess_exposure(dark_img, fov_mask)

    assert exp_normal["is_underexposed"] is False
    assert exp_dark["is_underexposed"] is True


def test_enhancement_transforms():
    img = get_rgb_sample(grade=2, quality="good", seed=5)

    clahe_out = apply_clahe(img)
    ben_out = apply_ben_graham(img)
    unified_out = enhance_fundus_image(img, method="ben_graham")

    assert clahe_out.shape == img.shape
    assert ben_out.shape == img.shape
    assert unified_out.shape == img.shape
    assert clahe_out.dtype == np.uint8
    assert ben_out.dtype == np.uint8


def test_iqa_gate_pass():
    good_img = get_rgb_sample(grade=0, quality="good", seed=100)
    result = assess_and_gate(good_img)

    assert isinstance(result, IQAResult)
    assert result.passable is True
    assert result.grade in ["PASS", "ENHANCED_PASS"]
    assert result.enhanced_image is not None


def test_iqa_gate_fail_severe_blur():
    severely_blurry_img = get_rgb_sample(grade=0, quality="blurry", seed=200)
    result = assess_and_gate(severely_blurry_img)

    assert result.passable is False
    assert result.grade == "FAIL"
    assert len(result.rejection_reasons) > 0
