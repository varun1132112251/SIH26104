"""Run a bounded end-to-end ASVspoof5 dataset smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the ASVspoof5 feature pipeline.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not 500 <= args.max_samples <= 1000:
        parser.error("--max-samples must be between 500 and 1000")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")

    dataset = ASVspoof5FeatureDataset(
        args.metadata, args.audio_root, max_samples=args.max_samples, on_error="zeros"
    )
    if len(dataset) != args.max_samples:
        raise RuntimeError(f"Expected {args.max_samples} protocol entries, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    features, labels = next(iter(loader))
    if tuple(features.shape[1:]) != (1, dataset.extractor.n_mels, dataset.target_frames):
        raise RuntimeError(f"Unexpected feature batch shape: {tuple(features.shape)}")
    if labels.dtype != torch.int64 or not set(labels.tolist()).issubset({0, 1}):
        raise RuntimeError("Labels are not encoded as binary torch integers")

    present_index = next(
        (index for index, record in enumerate(dataset.records) if record.audio_path.is_file()),
        None,
    )
    if present_index is None:
        raise RuntimeError("The smoke sample contains no present audio file to decode")
    present_features, _ = dataset[present_index]
    if tuple(present_features.shape) != tuple(features.shape[1:]):
        raise RuntimeError("Decoded audio produced the wrong feature shape")

    missing = next((record for record in dataset.records if not record.audio_path.exists()), None)
    missing_status = "none in sample"
    if missing is not None:
        missing_index = dataset.records.index(missing)
        fallback, _ = dataset[missing_index]
        if tuple(fallback.shape) != (1, dataset.extractor.n_mels, dataset.target_frames):
            raise RuntimeError("Missing-file fallback has the wrong shape")
        missing_status = "verified"

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        corrupt_root = temporary_path / "flac_T"
        corrupt_root.mkdir()
        corrupt_protocol = temporary_path / "smoke.tsv"
        corrupt_protocol.write_text(
            "T_smoke T_smoke_missing F - - - - - spoof -\n"
            "T_smoke T_smoke_corrupt F - - - - - bonafide -\n",
            encoding="utf-8",
        )
        (corrupt_root / "T_smoke_corrupt.flac").write_bytes(b"not audio")
        error_dataset = ASVspoof5FeatureDataset(corrupt_protocol, corrupt_root, on_error="zeros")
        error_features = [error_dataset[index][0] for index in range(2)]
        if any(tuple(feature.shape) != tuple(error_features[0].shape) for feature in error_features):
            raise RuntimeError("Missing/corrupt fallback shapes differ")

    print(
        f"PASS samples={len(dataset)} batch_shape={tuple(features.shape)} "
        f"protocol_entries={dataset.stats.total_protocol_entries} "
        f"existing_audio={dataset.stats.existing_audio} missing_audio={dataset.stats.missing_audio} "
        f"spoof={dataset.stats.spoof} bonafide={dataset.stats.bonafide} missing_fallback={missing_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())