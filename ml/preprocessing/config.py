"""Audio preprocessing configuration.

These settings define the fixed-duration segmentation used for later real-time
inference. The values are intentionally centralized so they can be configured
without hard-coding them throughout the ML pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Configuration values for standardizing audio before model input."""

    sample_rate: int = 16000
    segment_duration: float = 2.0
    hop_duration: float = 1.0
    normalization: str = "peak"
    target_peak_amplitude: float = 0.95
    min_duration_seconds: float = 0.05
    silence_threshold: float = 1e-6
    min_peak_amplitude: float = 1e-8

    @property
    def segment_samples(self) -> int:
        """Return the target number of samples for a fixed segment."""
        return max(1, int(round(self.segment_duration * self.sample_rate)))

    @property
    def hop_samples(self) -> int:
        """Return the step size in samples between segments."""
        return max(1, int(round(self.hop_duration * self.sample_rate)))

    @property
    def min_samples(self) -> int:
        """Minimum number of samples required for valid audio."""
        return max(1, int(round(self.min_duration_seconds * self.sample_rate)))
