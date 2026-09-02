"""AASIST-L inference wrapper for speech anti-spoofing.

The model is the lightweight AASIST countermeasure published by the
SpeechAntiSpoofingBenchmarks project and trained on ASVspoof 2019 LA. The
ONNX checkpoint is downloaded on first use instead of being committed to Git.

AASIST-L expects mono float32 speech at 16 kHz and a deterministic 64,600
sample evaluation window. Its output is converted to a spoof probability,
where 0 means bona fide and 1 means spoof/synthetic.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised only without optional dependency
    ort = None


MODEL_URL = (
    "https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST-L/"
    "resolve/main/aasist-l.onnx"
)
MODEL_SAMPLES = 64_600
SAMPLE_RATE = 16_000


class AASISTModelError(RuntimeError):
    """Raised when AASIST-L cannot be loaded or executed."""


def _pad_fixed(audio: np.ndarray, max_len: int = MODEL_SAMPLES) -> np.ndarray:
    """Create the deterministic evaluation window used by AASIST-L."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise AASISTModelError("Cannot infer on empty audio.")
    if x.size >= max_len:
        return x[:max_len]
    repetitions = max_len // x.size + 1
    return np.tile(x, repetitions)[:max_len].astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class AASISTL:
    """Lightweight ONNXRuntime wrapper around the AASIST-L detector."""

    name = "AASIST-L"
    expected_sample_rate = SAMPLE_RATE

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        self.model_path = Path(model_path) if model_path else Path("models_cache/aasist-l.onnx")
        self.session = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

    def _download_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.model_path.with_suffix(".download")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            tmp_path.replace(self.model_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise AASISTModelError(
                f"Unable to download AASIST-L model from {MODEL_URL}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load the ONNX model, downloading it once when necessary."""
        if ort is None:
            raise AASISTModelError(
                "onnxruntime is required. Install dependencies with `pip install -r requirements.txt`."
            )
        if not self.model_path.exists():
            self._download_model()
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if not inputs or not outputs:
                raise AASISTModelError("AASIST-L ONNX graph has no usable input/output tensors.")
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name
        except Exception as exc:
            raise AASISTModelError(f"Could not load AASIST-L ONNX model: {exc}") from exc

    def _ensure_loaded(self) -> None:
        if self.session is None:
            self.load()

    def score_batch(self, audios: list[np.ndarray]) -> list[float]:
        """Return spoof probabilities for a batch of 16 kHz mono waveforms."""
        self._ensure_loaded()
        assert self.session is not None
        assert self.input_name is not None

        batch = np.stack([_pad_fixed(audio) for audio in audios]).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: batch})
        raw = np.asarray(outputs[0])

        if raw.ndim == 2 and raw.shape[1] == 2:
            # AASIST convention: class 1 is bona fide. Therefore spoof is
            # the probability of class 0.
            probabilities = _softmax(raw)
            spoof = probabilities[:, 0]
        elif raw.ndim == 1:
            # Some exported graphs expose the bona-fide logit directly.
            spoof = 1.0 / (1.0 + np.exp(np.clip(raw, -60.0, 60.0)))
        else:
            raise AASISTModelError(f"Unexpected model output shape: {raw.shape}")

        return [float(np.clip(value, 0.0, 1.0)) for value in spoof]

    def score(self, audio: np.ndarray) -> float:
        """Return spoof probability for one waveform."""
        return self.score_batch([audio])[0]

    def unload(self) -> None:
        """Release the ONNX session."""
        self.session = None
        self.input_name = None
        self.output_name = None


__all__ = ["AASISTL", "AASISTModelError", "MODEL_SAMPLES", "SAMPLE_RATE"]
