"""Generic dataset interfaces for audio classification experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np


class SampleLabel(str, Enum):
    """Enumeration of supported anti-spoofing labels."""

    BONAFIDE = "bonafide"
    SPOOF = "spoof"


@dataclass(frozen=True)
class AudioSample:
    """A single audio waveform and label from a dataset entry."""

    audio: np.ndarray
    sample_rate: int = 16000
    label: SampleLabel = SampleLabel.BONAFIDE
    file_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio", np.asarray(self.audio, dtype=np.float32))
        if self.audio.size == 0:
            object.__setattr__(self, "audio", np.zeros(0, dtype=np.float32))


@dataclass(frozen=True)
class DatasetEntry(AudioSample):
    """Concrete dataset entry used by dataset iterators."""


class BaseAudioDataset(ABC, Sequence[DatasetEntry]):
    """Abstract interface for dataset loaders.

    Concrete dataset implementations should parse row formats specific to a
    dataset or metadata source and yield :class:`DatasetEntry` objects.
    """

    def __init__(self, entries: Optional[Sequence[DatasetEntry]] = None) -> None:
        self._entries = list(entries or [])

    @abstractmethod
    def _parse_row(self, row: Mapping[str, Any]) -> Optional[DatasetEntry]:
        """Return a dataset entry for a single row, or None when the row is invalid."""

    def __iter__(self) -> Iterator[DatasetEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> DatasetEntry:
        return self._entries[index]

    def append(self, entry: DatasetEntry) -> None:
        """Add a new entry to the dataset in memory."""
        self._entries.append(entry)

    @classmethod
    def normalize_label(cls, value: Any) -> SampleLabel:
        """Translate raw label values into the canonical dataset enum."""
        if isinstance(value, SampleLabel):
            return value

        normalized = str(value).strip().lower()
        if normalized in {"bonafide", "real", "genuine", "target"}:
            return SampleLabel.BONAFIDE
        if normalized in {"spoof", "fake", "attack"}:
            return SampleLabel.SPOOF
        raise ValueError(f"Unsupported label value: {value!r}")

    def from_rows(self, rows: Iterable[Mapping[str, Any]]) -> list[DatasetEntry]:
        """Parse a collection of rows and keep only valid entries."""
        entries: list[DatasetEntry] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            parsed = self._parse_row(row)
            if parsed is not None:
                entries.append(parsed)
        self._entries = entries
        return entries
