"""Feature extraction for ASVspoof audio segments."""

from __future__ import annotations

from collections.abc import Sequence

import librosa
import numpy as np


class MelSpectrogramExtractor:
	"""Extract log-Mel spectrograms from 16 kHz mono float32 audio."""

	def __init__(
		self,
		*,
		sample_rate: int = 16_000,
		n_fft: int = 1024,
		hop_length: int = 256,
		n_mels: int = 80,
		fmin: float = 20.0,
		fmax: float | None = None,
	) -> None:
		if sample_rate <= 0:
			raise ValueError("sample_rate must be positive.")
		if n_fft <= 0:
			raise ValueError("n_fft must be positive.")
		if hop_length <= 0:
			raise ValueError("hop_length must be positive.")
		if n_mels <= 0:
			raise ValueError("n_mels must be positive.")
		if fmin < 0:
			raise ValueError("fmin must be non-negative.")
		resolved_fmax = min(7600.0, sample_rate / 2) if fmax is None else fmax
		if resolved_fmax <= fmin or resolved_fmax > sample_rate / 2:
			raise ValueError("fmax must be greater than fmin and no greater than Nyquist.")

		self.sample_rate = sample_rate
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.n_mels = n_mels
		self.fmin = fmin
		self.fmax = resolved_fmax

	def expected_frames(self, sample_count: int) -> int:
		"""Return the deterministic frame count for a fixed waveform length."""
		if sample_count <= 0:
			raise ValueError("sample_count must be positive.")
		return 1 + sample_count // self.hop_length

	@staticmethod
	def _validate_audio(audio: np.ndarray) -> np.ndarray:
		"""Validate and normalize an audio segment without changing its dtype."""
		if not isinstance(audio, np.ndarray):
			raise TypeError("audio must be a NumPy array.")
		if audio.dtype != np.float32:
			raise TypeError("audio must have dtype float32.")
		if audio.ndim != 1:
			raise ValueError("audio must be a one-dimensional mono segment.")
		if audio.size == 0:
			raise ValueError("audio must not be empty.")
		if not np.isfinite(audio).all():
			raise ValueError("audio must contain only finite values.")
		return audio

	def extract(self, audio: np.ndarray) -> np.ndarray:
		"""Return a log-Mel spectrogram with shape ``(n_mels, frames)``."""
		waveform = self._validate_audio(audio)
		mel = librosa.feature.melspectrogram(
			y=waveform,
			sr=self.sample_rate,
			n_fft=self.n_fft,
			hop_length=self.hop_length,
			n_mels=self.n_mels,
			fmin=self.fmin,
			fmax=self.fmax,
			power=2.0,
		)
		log_mel = librosa.power_to_db(mel, ref=np.max)
		return np.asarray(log_mel, dtype=np.float32)

	def extract_batch(self, audio_batch: Sequence[np.ndarray]) -> np.ndarray:
		"""Return stacked log-Mel spectrograms for equally sized segments."""
		if not audio_batch:
			raise ValueError("audio_batch must not be empty.")

		features = [self.extract(audio) for audio in audio_batch]
		try:
			return np.stack(features).astype(np.float32, copy=False)
		except ValueError as exc:
			raise ValueError("All audio segments must produce the same shape.") from exc


__all__ = ["MelSpectrogramExtractor"]
