"""Dataset adapter for spoofing datasets.

This module intentionally keeps the dataset-specific parsing separate from the
generic interface in :mod:`ml.datasets.base`. It is designed to be extended for
ASVspoof or other anti-spoofing corpora later without changing the abstract API.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .base import BaseAudioDataset, DatasetEntry, SampleLabel


class SpoofDataset(BaseAudioDataset):
    """Simple row-based dataset parser for spoofing challenges.

    The dataset accepts a list of record dictionaries such as
    {"audio_path": "spoof.wav", "label": "spoof"}. Concrete future loaders can
    subclass or wrap this class while preserving the same generic interface.
    """

    def __init__(self, rows: Optional[Iterable[Mapping[str, Any]]] = None) -> None:
        self._rows = list(rows or [])
        super().__init__()
        self.from_rows(self._rows)

    def _parse_row(self, row: Mapping[str, Any]) -> Optional[DatasetEntry]:
        """Build a dataset entry from a single row or return None when invalid."""
        if not row:
            return None

        audio_path = row.get("audio_path") or row.get("path") or row.get("file_path")
        if audio_path is None:
            return None

        label_value = row.get("label", row.get("target", row.get("class")))
        if label_value is None:
            return None

        try:
            label = self.normalize_label(label_value)
        except ValueError:
            return None

        return DatasetEntry(
            audio=np.zeros(0, dtype=np.float32),
            sample_rate=16000,
            label=label,
            file_path=str(audio_path),
            metadata={"source": "row"},
        )


__all__ = ["SpoofDataset"]
