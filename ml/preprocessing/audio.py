"""Audio preprocessing pipeline for voice deepfake detection research."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:  # pragma: no cover - optional dependency
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None

from .config import AudioConfig

PathLike = Union[str, Path]


class AudioProcessingError(ValueError):
    """Raised when audio cannot be loaded or processed safely."""


def _ensure_float32(audio: np.ndarray) -> np.ndarray:
    """Convert input audio to a writable float32 ndarray."""
    array = np.asarray(audio)
    if array.size == 0:
        raise AudioProcessingError("Audio data is empty.")
    if not np.isfinite(array).all():
        raise AudioProcessingError("Audio data contains NaN or Inf values.")
    return array.astype(np.float32, copy=False)


def _convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio into a mono waveform."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return np.mean(audio, axis=1, dtype=np.float32)
    raise AudioProcessingError(
        f"Unsupported audio shape {audio.shape}; only mono or 2D channel data is supported."
    )


def _resample_audio(audio: np.ndarray, input_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to the configured sample rate using polyphase resampling."""
    if input_sr <= 0:
        raise AudioProcessingError("Input sample rate must be positive.")
    if target_sr <= 0:
        raise AudioProcessingError("Target sample rate must be positive.")
    if input_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    gcd = math.gcd(input_sr, target_sr)
    up = target_sr // gcd
    down = input_sr // gcd
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32, copy=False)


def _normalize_audio(audio: np.ndarray, config: AudioConfig) -> np.ndarray:
    """Apply a consistent peak-based amplitude normalization."""
    if audio.size == 0:
        raise AudioProcessingError("Cannot normalize empty audio.")

    peak = float(np.max(np.abs(audio)))
    if peak <= config.silence_threshold:
        return np.zeros_like(audio, dtype=np.float32)

    if config.normalization.lower() == "peak":
        if peak < config.min_peak_amplitude:
            return audio.astype(np.float32, copy=False)
        target = max(float(config.target_peak_amplitude), 0.0)
        scale = min(target / peak, 1.0 / max(peak, 1e-12))
        return (audio * scale).astype(np.float32, copy=False)

    if config.normalization.lower() == "rms":
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms <= config.silence_threshold:
            return np.zeros_like(audio, dtype=np.float32)
        target_rms = max(float(config.target_peak_amplitude), 0.0)
        return (audio * (target_rms / rms)).astype(np.float32, copy=False)

    raise AudioProcessingError(f"Unsupported normalization mode: {config.normalization!r}")


class AudioPreprocessor:
    """Standardize raw audio for later feature extraction and inference."""

    def __init__(self, config: Optional[AudioConfig] = None) -> None:
        self.config = config or AudioConfig()

    def load_audio(self, audio_path: PathLike, *, config: Optional[AudioConfig] = None) -> Tuple[np.ndarray, int]:
        """Load audio from disk and return standardized waveform + output sample rate."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        cfg = config or self.config
        suffix = path.suffix.lower()

        try:
            if suffix == ".wav":
                sample_rate, waveform = wavfile.read(str(path))
            elif sf is not None:
                waveform, sample_rate = sf.read(str(path), always_2d=False)
            else:
                raise AudioProcessingError(
                    "Audio loading requires scipy.wavfile for WAV files or soundfile for other formats."
                )
        except (ValueError, RuntimeError, OSError) as exc:  # pragma: no cover - depends on codec support
            raise AudioProcessingError(f"Could not decode audio file '{path}': {exc}") from exc

        processed = self.preprocess(waveform, sample_rate=sample_rate, config=cfg)
        return processed.astype(np.float32, copy=False), cfg.sample_rate

    def preprocess(
        self,
        audio: Sequence[float] | np.ndarray,
        *,
        sample_rate: int,
        config: Optional[AudioConfig] = None,
    ) -> np.ndarray:
        """Convert arbitrary audio input into mono 16kHz float32 normalized waveform."""
        cfg = config or self.config
        if sample_rate <= 0:
            raise AudioProcessingError("Sample rate must be positive.")

        waveform = _ensure_float32(np.asarray(audio))
        waveform = _convert_to_mono(waveform)
        waveform = _resample_audio(waveform, sample_rate, cfg.sample_rate)
        waveform = _normalize_audio(waveform, cfg)

        if waveform.size < cfg.min_samples:
            pad_length = max(cfg.min_samples - waveform.size, 0)
            waveform = np.pad(waveform, (0, pad_length), mode="constant")

        return waveform.astype(np.float32, copy=False)

    def segment_audio(
        self,
        audio: Sequence[float] | np.ndarray,
        *,
        sample_rate: int = 16000,
        config: Optional[AudioConfig] = None,
    ) -> list[np.ndarray]:
        """Split an audio array into fixed-duration segments using configured hop size."""
        cfg = config or self.config
        if cfg.segment_duration <= 0:
            raise AudioProcessingError("Segment duration must be positive.")
        if cfg.hop_duration <= 0:
            raise AudioProcessingError("Hop duration must be positive.")

        processed = self.preprocess(audio, sample_rate=sample_rate, config=cfg)
        segment_samples = cfg.segment_samples
        hop_samples = cfg.hop_samples

        if processed.size == 0:
            raise AudioProcessingError("Cannot segment empty audio.")

        segments: list[np.ndarray] = []
        start = 0
        while start + segment_samples <= processed.shape[0]:
            segment = processed[start : start + segment_samples]
            segments.append(segment.astype(np.float32, copy=False))
            start += hop_samples

        if not segments:
            padded = np.pad(processed, (0, max(segment_samples - processed.size, 0)), mode="constant")
            segments.append(padded[:segment_samples].astype(np.float32, copy=False))

        return segments


__all__ = ["AudioConfig", "AudioPreprocessor", "AudioProcessingError"]
