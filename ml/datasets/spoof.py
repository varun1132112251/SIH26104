"""ASVspoof 5 dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .base import BaseAudioDataset, DatasetEntry, SampleLabel


class SpoofDataset(BaseAudioDataset):
    """Dataset adapter for ASVspoof 5 protocol files.

    ASVspoof 5 protocol format:

        speaker_id utterance_id gender ... attack_id attack_label label ...

    The canonical detection label is column 8:
        spoof / bonafide
    """

    def __init__(
        self,
        rows: Optional[Iterable[Mapping[str, Any]]] = None,
        *,
        audio_root: Optional[str | Path] = None,
    ) -> None:
        self.audio_root = Path(audio_root) if audio_root else None
        super().__init__()
        if rows is not None:
            self.from_rows(rows)

    def _resolve_audio_path(self, utterance_id: str) -> Optional[str]:
        """Resolve an ASVspoof utterance ID to an audio file."""
        if self.audio_root is None:
            return None

        root = self.audio_root

        candidates = [
            root / f"{utterance_id}.flac",
            root / f"{utterance_id}.wav",
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        # Handles datasets where files are stored in nested directories.
        matches = list(root.rglob(f"{utterance_id}.flac"))
        if matches:
            return str(matches[0])

        matches = list(root.rglob(f"{utterance_id}.wav"))
        if matches:
            return str(matches[0])

        return None

    def _parse_row(self, row: Mapping[str, Any]) -> Optional[DatasetEntry]:
        """Convert a normalized row dictionary into a DatasetEntry."""
        if not row:
            return None

        utterance_id = (
            row.get("utterance_id")
            or row.get("file_id")
            or row.get("id")
            or row.get("path")
        )

        label_value = (
            row.get("label")
            or row.get("target")
            or row.get("class")
        )

        if utterance_id is None or label_value is None:
            return None

        try:
            label = self.normalize_label(label_value)
        except ValueError:
            return None

        audio_path = self._resolve_audio_path(str(utterance_id))

        return DatasetEntry(
            audio=np.zeros(0, dtype=np.float32),
            sample_rate=16000,
            label=label,
            file_path=audio_path,
            metadata={
                "dataset": "ASVspoof5",
                "utterance_id": str(utterance_id),
                "speaker_id": row.get("speaker_id"),
                "gender": row.get("gender"),
                "attack_id": row.get("attack_id"),
                "attack_label": row.get("attack_label"),
            },
        )

    @classmethod
    def from_asvspoof_protocol(
        cls,
        protocol_path: str | Path,
        *,
        audio_root: Optional[str | Path] = None,
    ) -> "SpoofDataset":
        """Load an ASVspoof 5 protocol TSV file."""

        protocol_path = Path(protocol_path)

        if not protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not found: {protocol_path}"
            )

        rows = []

        with protocol_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                fields = line.split()

                if len(fields) != 10:
                    continue

                rows.append(
                    {
                        "speaker_id": fields[0],
                        "utterance_id": fields[1],
                        "gender": fields[2],
                        "field_3": fields[3],
                        "field_4": fields[4],
                        "field_5": fields[5],
                        "attack_id": fields[6],
                        "attack_label": fields[7],
                        "label": fields[8],
                        "field_9": fields[9],
                        "line_number": line_number,
                    }
                )

        dataset = cls(rows, audio_root=audio_root)
        return dataset


__all__ = ["SpoofDataset"]