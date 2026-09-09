"""Inference service for the trained ASVspoof5 baseline CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch

from ml.models.audio_cnn import AudioCNN
from ml.preprocessing.audio import AudioProcessingError, AudioPreprocessor
from ml.preprocessing.features import MelSpectrogramExtractor

DEFAULT_THRESHOLD = 0.80


class CNNInferenceError(RuntimeError):
    """Raised when a CNN checkpoint or audio input cannot be used for inference."""


@dataclass(frozen=True)
class CNNInferenceResult:
    """Stable inference response suitable for API/UI serialization."""

    spoof_probability: float
    bonafide_probability: float
    decision: str
    risk_score: float


class CNNInference:
    """Load and run the baseline CNN on one local audio file."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        preprocessor: AudioPreprocessor | None = None,
        extractor: MelSpectrogramExtractor | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.extractor = extractor or MelSpectrogramExtractor(
            sample_rate=self.preprocessor.config.sample_rate
        )
        self.model = self._load_model()

    def _load_model(self) -> AudioCNN:
        if not self.checkpoint_path.is_file():
            raise CNNInferenceError(f"Checkpoint not found: {self.checkpoint_path}")
        try:
            checkpoint: Any = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
            state_dict = checkpoint["model_state_dict"]
            input_shape = tuple(checkpoint.get("input_shape", AudioCNN.input_shape))
            if input_shape != AudioCNN.input_shape:
                raise CNNInferenceError(
                    f"Checkpoint input shape {input_shape} does not match {AudioCNN.input_shape}."
                )
            model = AudioCNN()
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            return model
        except CNNInferenceError:
            raise
        except (KeyError, RuntimeError, OSError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            raise CNNInferenceError(
                f"Could not load CNN checkpoint '{self.checkpoint_path}': {exc}"
            ) from exc
    def _features(self, audio_path: str | Path) -> torch.Tensor:
        try:
            audio, _ = self.preprocessor.load_audio(audio_path)

            segment_samples = self.preprocessor.config.segment_samples
            hop_samples = self.preprocessor.config.hop_samples

            if audio.size < segment_samples:
                starts = [0]
            else:
                last_start = audio.size - segment_samples
                starts = list(range(0, last_start + 1, hop_samples))
                if starts[-1] != last_start:
                    starts.append(last_start)

            feature_tensors = []

            target_frames = self.extractor.expected_frames(segment_samples)

            for start in starts:
                segment = audio[start:start + segment_samples]

                if segment.size < segment_samples:
                    segment = np.pad(
                        segment,
                        (0, segment_samples - segment.size),
                        mode="constant",
                    )

                features = self.extractor.extract(
                    segment.astype(np.float32, copy=False)
                )

                if features.shape[1] < target_frames:
                    features = np.pad(
                        features,
                        ((0, 0), (0, target_frames - features.shape[1])),
                    )

                features = features[:, :target_frames]

                tensor = torch.from_numpy(
                    features[np.newaxis, np.newaxis, :].copy()
                ).to(
                    device=self.device,
                    dtype=torch.float32,
                )

                feature_tensors.append(tensor)

            batch = torch.cat(feature_tensors, dim=0)

            if tuple(batch.shape[1:]) != AudioCNN.input_shape:
                raise CNNInferenceError(
                    f"Unexpected feature shape: {tuple(batch.shape[1:])}"
                )

            return batch

        except CNNInferenceError:
            raise
        except (
            AudioProcessingError,
            FileNotFoundError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            raise CNNInferenceError(
                f"Could not process audio '{audio_path}': {exc}"
            ) from exc

    def predict(self, audio_path: str | Path) -> CNNInferenceResult:
        """Run inference over the full recording using overlapping segments."""
        inputs = self._features(audio_path)

        try:
            with torch.inference_mode():
                logits = self.model(inputs)
                probabilities = torch.softmax(logits, dim=1)
                mean_probabilities = probabilities.mean(dim=0)
                mean_probabilities = mean_probabilities.detach().cpu().numpy()

        except (RuntimeError, ValueError, TypeError) as exc:
            raise CNNInferenceError(
                f"CNN inference failed for '{audio_path}': {exc}"
            ) from exc

        if mean_probabilities.shape != (2,) or not np.isfinite(mean_probabilities).all():
            raise CNNInferenceError("CNN returned invalid probabilities.")

        bonafide_probability = float(mean_probabilities[0])
        spoof_probability = float(mean_probabilities[1])

        return CNNInferenceResult(
            spoof_probability=spoof_probability,
            bonafide_probability=bonafide_probability,
            decision=(
                "SPOOF"
                if spoof_probability >= DEFAULT_THRESHOLD
                else "BONAFIDE"
            ),
            risk_score=spoof_probability,
        )

__all__ = ["CNNInference", "CNNInferenceError", "CNNInferenceResult"]
