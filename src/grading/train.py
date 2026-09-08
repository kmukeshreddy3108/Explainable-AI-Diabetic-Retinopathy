"""
Model Training Script with Enforced Dataset Roles.

Strictly enforces dataset assignment:
- APTOS 2019 + IDRiD: Training and Internal Validation ONLY
- Messidor-2: External Holdout Validation ONLY (Never touched during training)
- DRIVE: Vessel Segmentation Benchmark ONLY (Never used for DR grading)
"""

import os
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

from src.grading.model import build_dr_model
from src.utils.config import DATASET_ROLES, GRADING_NUM_CLASSES


def validate_dataset_role(dataset_name: str, intended_action: str):
    """
    Hard constraint enforcement function.
    Raises RuntimeError if a dataset is accessed for an illegal operation.
    """
    assigned_role = DATASET_ROLES.get(dataset_name)
    if not assigned_role:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Must be one of {list(DATASET_ROLES.keys())}")

    if intended_action == "train" and assigned_role == "external":
        raise RuntimeError(
            f"VIOLATION: Attempted to use external holdout dataset '{dataset_name}' for training! "
            f"Messidor-2 must remain an untouched holdout."
        )

    if intended_action == "dr_grading" and assigned_role == "vessel_benchmark":
        raise RuntimeError(
            f"VIOLATION: Attempted to use vessel benchmark dataset '{dataset_name}' for DR grading!"
        )


def train_demo_weights(output_path: str = "models/dr_grading_efficientnet.pt") -> str:
    """
    Trains/initializes demo weights for EfficientNet-B0 and saves .pt state_dict.

    Args:
        output_path: Target .pt filepath

    Returns:
        Absolute filepath to saved model weights
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Validate dataset roles before training
    validate_dataset_role("APTOS_2019", "train")
    validate_dataset_role("IDRiD", "train")

    model = build_dr_model(pretrained=True)

    # Save state_dict
    torch.save(model.state_dict(), output_path)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    saved_path = train_demo_weights()
    print(f"Saved initial model weights to {saved_path}")
