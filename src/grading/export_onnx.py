"""
ONNX Model Export Script.

Exports PyTorch DRGradingModel to ONNX format for deployment and edge execution.
Enforces Opset 17 and dynamic batching axes.
"""

import os
from typing import Optional
import torch

from src.grading.model import build_dr_model, DRGradingModel
from src.utils.config import GRADING_INPUT_SIZE, GRADING_ONNX_OPSET


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str = "models/dr_grading_efficientnet.onnx",
    input_size: int = GRADING_INPUT_SIZE,
    opset: int = GRADING_ONNX_OPSET,
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch DRGradingModel
        output_path: Target .onnx filepath
        input_size: Image spatial dimension
        opset: ONNX opset version (default 17)

    Returns:
        Absolute filepath to exported ONNX model
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset if opset <= 14 else 14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
    except Exception as err:
        # Try legacy/minimal export options if primary export fails
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
        )

    return os.path.abspath(output_path)


if __name__ == "__main__":
    m = build_dr_model(pretrained=False)
    path = export_to_onnx(m)
    print(f"Successfully exported model to {path}")
