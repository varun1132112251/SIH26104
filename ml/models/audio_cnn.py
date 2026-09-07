"""Baseline convolutional detector for ASVspoof log-Mel features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AudioCNN(nn.Module):
    """Small two-class CNN for ``(1, 80, 126)`` log-Mel inputs."""

    input_shape = (1, 80, 126)
    num_classes = 2

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, self.num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return bonafide/spoof logits for a batch of feature tensors."""
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected input shape (batch, {self.input_shape[0]}, "
                f"{self.input_shape[1]}, {self.input_shape[2]}), got {tuple(inputs.shape)}."
            )
        if not inputs.is_floating_point():
            raise TypeError("Model inputs must be floating-point tensors.")

        return self.classifier(self.features(inputs).flatten(start_dim=1))


__all__ = ["AudioCNN"]