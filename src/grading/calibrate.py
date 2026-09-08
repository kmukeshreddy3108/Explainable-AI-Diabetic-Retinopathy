"""
Temperature Scaling for Confidence Calibration.

Calibrates raw neural network logits to output well-calibrated confidence probabilities.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to network logits: logit_calibrated = logit / T.
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temperature = self.temperature.clamp(min=0.1, max=10.0)
        return logits / temperature

    def set_temperature(self, temp: float):
        with torch.no_grad():
            self.temperature.copy_(torch.tensor([temp]))


def calibrate_temperature(
    logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50
) -> float:
    """
    Find optimal temperature parameter T using L-BFGS or Adam on validation logits and labels.

    Args:
        logits: Tensor of shape (N, num_classes)
        labels: Tensor of shape (N,) int64

    Returns:
        float: Optimized temperature T.
    """
    scaler = TemperatureScaler(initial_temperature=1.5)
    optimizer = optim.Adam([scaler.temperature], lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler.train()
    for _ in range(max_iter):
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        optimizer.step()

    opt_temp = float(scaler.temperature.item())
    return max(0.1, round(opt_temp, 3))
