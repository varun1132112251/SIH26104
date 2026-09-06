"""Evaluate a CNN checkpoint with validation-selected thresholding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset
from ml.datasets.splits import split_asvspoof5
from ml.evaluation.metrics import classification_metrics, select_threshold
from ml.models.audio_cnn import AudioCNN


def _scores(
    model: AudioCNN, dataset: Subset, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for features, batch_labels in loader:
            probabilities = torch.softmax(model(features.to(device)), dim=1)[:, 1]
            scores.append(probabilities.cpu().numpy())
            labels.append(torch.as_tensor(batch_labels).numpy())
    if not scores:
        raise ValueError("Cannot evaluate an empty split.")
    return np.concatenate(scores), np.concatenate(labels)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    metadata: str | Path | None = None,
    audio_root: str | Path | None = None,
    max_samples: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, object]:
    """Select a threshold on validation data and report held-out test metrics."""
    checkpoint_path = Path(checkpoint_path)
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    dataset_config = checkpoint.get("dataset", {})
    split_config = checkpoint.get("split", {})
    dataset = ASVspoof5FeatureDataset(
        metadata or dataset_config.get("metadata_path"),
        audio_root or dataset_config.get("audio_root"),
        max_samples=max_samples if max_samples is not None else dataset_config.get("max_samples"),
    )
    split = split_asvspoof5(
        dataset,
        validation_fraction=split_config.get("validation_fraction", 0.2),
        test_fraction=split_config.get("test_fraction", 0.15),
        seed=split_config.get("seed", checkpoint.get("seed", 42)),
        audio_only=dataset_config.get("audio_only", True),
    )
    model = AudioCNN().to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    train_dataset, validation_dataset, test_dataset = split.datasets(dataset)
    del train_dataset
    validation_scores, validation_labels = _scores(model, validation_dataset, selected_device, batch_size)
    threshold = select_threshold(validation_scores, validation_labels)
    test_scores, test_labels = _scores(model, test_dataset, selected_device, batch_size)
    result = {
        "checkpoint": str(checkpoint_path),
        "split_sizes": {"validation": len(validation_dataset), "test": len(test_dataset)},
        "threshold_selection": {"criterion": "maximum validation F1; ties prefer 0.5", "threshold": threshold},
        "validation": classification_metrics(validation_scores, validation_labels, threshold),
        "test": classification_metrics(test_scores, test_labels, threshold),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an ASVspoof 5 CNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive.")
    print(json.dumps(
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            metadata=args.metadata,
            audio_root=args.audio_root,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=args.device,
        ),
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())