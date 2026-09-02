"""Dataset abstractions for audio samples and labels."""

from .base import AudioSample, BaseAudioDataset, DatasetEntry, SampleLabel
from .spoof import SpoofDataset

__all__ = ["AudioSample", "BaseAudioDataset", "DatasetEntry", "SampleLabel", "SpoofDataset"]
