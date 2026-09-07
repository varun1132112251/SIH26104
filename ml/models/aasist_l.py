"""AASIST-L inference wrapper for speech anti-spoofing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - optional dependency
    hf_hub_download = None


HF_REPO_ID = "SpeechAntiSpoofingBenchmarks/AASIST-L"
HF_REVISION = "2bc4bf063f34f081cafa5d51c2e72bbfa5d39715"
HF_FILENAME = "aasist-l.onnx"
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


class AASISTL:
    """ONNX Runtime wrapper around the lightweight AASIST-L detector."""

    name = "AASIST-L"
    expected_sample_rate = SAMPLE_RATE

    def __init__(self, model_path: Optional[str | Path] = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.session = None
        self.input_name: Optional[str] = None

    def _download_model(self) -> Path:
        if hf_hub_download is None:
            raise AASISTModelError(
                "huggingface-hub is required. Install dependencies with `pip install -r requirements.txt`."
            )
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                revision=HF_REVISION,
                local_dir="models_cache",
            )
            return Path(downloaded)
        except Exception as exc:
            raise AASISTModelError(
                f"Unable to download AASIST-L checkpoint from Hugging Face: {exc}"
            ) from exc

    def load(self) -> None:
        """Load the official AASIST-L ONNX checkpoint."""
        if ort is None:
            raise AASISTModelError(
                "onnxruntime is required. Install dependencies with `pip install -r requirements.txt`."
            )
        if self.model_path is None:
            self.model_path = self._download_model()
        if not self.model_path.exists():
            raise AASISTModelError(f"AASIST-L model file does not exist: {self.model_path}")
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            inputs = self.session.get_inputs()
            if not inputs:
                raise AASISTModelError("AASIST-L ONNX graph has no input tensor.")
            self.input_name = inputs[0].name
        except Exception as exc:
            raise AASISTModelError(f"Could not load AASIST-L ONNX model: {exc}") from exc

    def _ensure_loaded(self) -> None:
        if self.session is None:
            self.load()

    def score_batch(self, audios: list[np.ndarray]) -> list[float]:
        """Return the AASIST-L bona-fide logit for each 16 kHz mono waveform.

        The exported ONNX graph returns two logits: class 0 is spoof and class 1
        is bona fide. The official model card defines the bona-fide logit as the
        Arena score, where higher values indicate more bona-fide speech.
        """
        if not audios:
            return []
        self._ensure_loaded()
        assert self.session is not None
        assert self.input_name is not None

        batch = np.stack([_pad_fixed(audio) for audio in audios]).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: batch})
        raw = np.asarray(outputs[0])

        if raw.ndim == 2 and raw.shape[1] == 2:
            scores = raw[:, 1]
        elif raw.ndim == 1 and raw.shape[0] == len(audios):
            scores = raw
        else:
            raise AASISTModelError(f"Unexpected model output shape: {raw.shape}")

        return [float(value) for value in scores]

    def score(self, audio: np.ndarray) -> float:
        """Return the bona-fide logit for one waveform."""
        return self.score_batch([audio])[0]

    def unload(self) -> None:
        """Release the ONNX session."""
        self.session = None
        self.input_name = None


__all__ = [
    "AASISTL",
    "AASISTModelError",
    "HF_REPO_ID",
    "HF_REVISION",
    "HF_FILENAME",
    "MODEL_SAMPLES",
    "SAMPLE_RATE",
]
