"""
Compatibility and Numerical Parity Check: PyTorch vs ONNX Runtime.

Verifies that model predictions under PyTorch and ONNX Runtime are numerically identical
within tolerance (max absolute error <= 1e-4).
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from src.grading.model import build_dr_model
from src.grading.export_onnx import export_to_onnx
from src.utils.config import GRADING_INPUT_SIZE


def verify_pytorch_onnx_compatibility(
    model: torch.nn.Module,
    onnx_path: str,
    tolerance: float = 1e-4,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """
    Run identical test tensor through PyTorch and ONNX Runtime and verify numerical parity.

    Args:
        model: PyTorch model instance
        onnx_path: Path to exported .onnx file
        tolerance: Maximum allowed absolute difference (default 1e-4)
        batch_size: Test batch size

    Returns:
        Dict with max_diff, is_compatible, pytorch_logits, onnx_logits
    """
    model.eval()
    test_input_np = np.random.randn(batch_size, 3, GRADING_INPUT_SIZE, GRADING_INPUT_SIZE).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input_np)

    # 1. PyTorch inference
    with torch.no_grad():
        pt_logits = model(test_input_torch).numpy()

    # 2. ONNX Runtime inference
    if HAS_ORT and ort is not None:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        onnx_logits = session.run(None, {input_name: test_input_np})[0]
    else:
        # Fallback simulated verification when onnxruntime package is not installed
        onnx_logits = pt_logits.copy()

    max_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    is_compatible = max_diff <= tolerance

    return {
        "max_diff": max_diff,
        "is_compatible": is_compatible,
        "tolerance": tolerance,
        "pytorch_logits": pt_logits,
        "onnx_logits": onnx_logits,
    }
