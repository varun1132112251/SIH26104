"""PyTorch feature dataset for the ASVspoof 5 training protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ml.preprocessing.audio import AudioPreprocessor
from ml.preprocessing.features import MelSpectrogramExtractor


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_METADATA_PATH = _PROJECT_ROOT / "data" / "asvspoof5" / "metadata" / "ASVspoof5.train.tsv"
_DEFAULT_AUDIO_ROOT = _PROJECT_ROOT / "data" / "asvspoof5" / "flac_T"
_LABELS = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class ASVspoof5Record:
    """Metadata needed to load one ASVspoof 5 training example."""

    utterance_id: str
    audio_path: Path
    label: int


class ASVspoof5FeatureDataset(Dataset[tuple[Tensor, int]]):
    """Load ASVspoof 5 audio and return fixed-size log-Mel tensors."""

    target_shape = (1, 80, 126)

    def __init__(
        self,
        metadata_path: str | Path = _DEFAULT_METADATA_PATH,
        audio_root: str | Path = _DEFAULT_AUDIO_ROOT,
        *,
        max_samples: int | None = None,
        preprocessor: AudioPreprocessor | None = None,
        extractor: MelSpectrogramExtractor | None = None,
    ) -> None:
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative or None.")

        self.metadata_path = Path(metadata_path)
        self.audio_root = Path(audio_root)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.extractor = extractor or MelSpectrogramExtractor()
        self.records = self._read_metadata(max_samples=max_samples)

    def _read_metadata(self, *, max_samples: int | None) -> list[ASVspoof5Record]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Training metadata not found: {self.metadata_path}")

        records: list[ASVspoof5Record] = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) <= 8:
                    raise ValueError(f"Invalid metadata row at line {line_number}.")

                utterance_id = fields[1]
                label_name = fields[8].lower()
                if label_name not in _LABELS:
                    raise ValueError(f"Unsupported label {fields[8]!r} at line {line_number}.")

                records.append(
                    ASVspoof5Record(
                        utterance_id=utterance_id,
                        audio_path=self.audio_root / f"{utterance_id}.flac",
                        label=_LABELS[label_name],
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    break

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        audio, _ = self.preprocessor.load_audio(record.audio_path)

        segment_samples = self.preprocessor.config.segment_samples
        if audio.size < segment_samples:
            audio = np.pad(audio, (0, segment_samples - audio.size), mode="constant")
        else:
            audio = audio[:segment_samples]

        features = self.extractor.extract(audio.astype(np.float32, copy=False))
        features = self._fit_frame_count(features)
        tensor = torch.from_numpy(features[np.newaxis, :, :].copy()).to(dtype=torch.float32)
        return tensor, record.label

    @classmethod
    def _fit_frame_count(cls, features: np.ndarray) -> np.ndarray:
        """Pad or crop the time axis to the model's fixed input width."""
        target_frames = cls.target_shape[2]
        if features.shape[0] != cls.target_shape[1]:
            raise ValueError(f"Expected {cls.target_shape[1]} Mel bins, got {features.shape[0]}.")
        if features.shape[1] < target_frames:
            features = np.pad(features, ((0, 0), (0, target_frames - features.shape[1])))
        return features[:, :target_frames].astype(np.float32, copy=False)


__all__ = ["ASVspoof5FeatureDataset", "ASVspoof5Record"]