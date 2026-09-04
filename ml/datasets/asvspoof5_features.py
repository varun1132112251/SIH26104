"""PyTorch feature dataset for the ASVspoof 5 training protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ml.preprocessing.audio import AudioProcessingError, AudioPreprocessor
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


@dataclass(frozen=True)
class ASVspoof5Stats:
    """Index statistics, including files absent while archives are incomplete."""

    total_protocol_entries: int
    existing_audio: int
    missing_audio: int
    spoof: int
    bonafide: int


def parse_asvspoof5_protocol(
    metadata_path: str | Path,
    audio_root: str | Path,
    *,
    max_samples: int | None = None,
) -> tuple[list[ASVspoof5Record], ASVspoof5Stats]:
    """Parse an ASVspoof5 protocol and resolve IDs without reading audio."""
    path = Path(metadata_path)
    root = Path(audio_root)
    if not path.exists():
        raise FileNotFoundError(f"Training metadata not found: {path}")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None.")

    records: list[ASVspoof5Record] = []
    spoof = bonafide = existing = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 9:
                raise ValueError(f"Invalid metadata row at line {line_number}.")
            utterance_id = fields[1]
            label_name = fields[8].lower()
            if label_name not in _LABELS:
                raise ValueError(f"Unsupported label {fields[8]!r} at line {line_number}.")
            audio_path = root / f"{utterance_id}.flac"
            existing += int(audio_path.is_file())
            spoof += int(label_name == "spoof")
            bonafide += int(label_name == "bonafide")
            records.append(ASVspoof5Record(utterance_id, audio_path, _LABELS[label_name]))
            if max_samples is not None and len(records) >= max_samples:
                break

    return records, ASVspoof5Stats(
        len(records), existing, len(records) - existing, spoof, bonafide
    )


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
        on_error: str = "zeros",
    ) -> None:
        if on_error not in {"zeros", "raise"}:
            raise ValueError("on_error must be 'zeros' or 'raise'.")

        self.metadata_path = Path(metadata_path)
        self.audio_root = Path(audio_root)
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.extractor = extractor or MelSpectrogramExtractor(
            sample_rate=self.preprocessor.config.sample_rate
        )
        self.on_error = on_error
        self.records, self.stats = parse_asvspoof5_protocol(
            self.metadata_path, self.audio_root, max_samples=max_samples
        )
        self.target_frames = self.extractor.expected_frames(
            self.preprocessor.config.segment_samples
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        try:
            audio, _ = self.preprocessor.load_audio(record.audio_path)
            segment_samples = self.preprocessor.config.segment_samples
            audio = np.pad(audio[:segment_samples], (0, max(segment_samples - audio.size, 0)))
            features = self.extractor.extract(audio.astype(np.float32, copy=False))
            features = self._fit_frame_count(features, self.target_frames)
        except (AudioProcessingError, FileNotFoundError, OSError, ValueError):
            if self.on_error == "raise":
                raise
            features = np.zeros((self.extractor.n_mels, self.target_frames), dtype=np.float32)
        tensor = torch.from_numpy(features[np.newaxis, :, :].copy()).to(dtype=torch.float32)
        return tensor, record.label

    @classmethod
    def _fit_frame_count(cls, features: np.ndarray, target_frames: int = 126) -> np.ndarray:
        """Pad or crop the time axis to the model's fixed input width."""
        if features.shape[0] <= 0:
            raise ValueError("Expected at least one Mel bin.")
        if features.shape[1] < target_frames:
            features = np.pad(features, ((0, 0), (0, target_frames - features.shape[1])))
        return features[:, :target_frames].astype(np.float32, copy=False)


__all__ = ["ASVspoof5FeatureDataset", "ASVspoof5Record", "ASVspoof5Stats", "parse_asvspoof5_protocol"]