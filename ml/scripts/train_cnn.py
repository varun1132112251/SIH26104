"""Train the baseline CNN detector on ASVspoof 5 features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset
from ml.datasets.splits import split_asvspoof5, split_id_sets
from ml.models.audio_cnn import AudioCNN


def _run_epoch(
    model: AudioCNN,
    loader: DataLoader[tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        item_count = labels.size(0)
        total_loss += loss.item() * item_count
        total_items += item_count
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    if total_items == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")
    return total_loss / total_items, np.concatenate(all_logits), np.concatenate(all_labels)


def _classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = float((predictions == labels).mean()) if labels.size else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _eer(logits: np.ndarray, labels: np.ndarray) -> float | None:
    """Estimate equal error rate; return None when validation lacks a class."""
    if not np.any(labels == 0) or not np.any(labels == 1):
        return None
    scores = logits[:, 1] - logits[:, 0]
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    false_accepts = np.cumsum(sorted_labels == 0) / negatives
    false_rejects = (int((labels == 1).sum()) - np.cumsum(sorted_labels == 1)) / positives
    crossing = int(np.argmin(np.abs(false_accepts - false_rejects)))
    return float((false_accepts[crossing] + false_rejects[crossing]) / 2.0)


def _split_dataset(
    dataset: ASVspoof5FeatureDataset, validation_fraction: float, seed: int
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for a train/validation split.")

    validation_size = max(1, int(round(len(dataset) * validation_fraction)))
    validation_size = min(validation_size, len(dataset) - 1)
    generator = torch.Generator().manual_seed(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, record in enumerate(dataset.records):
        by_label[record.label].append(index)
    validation_indices: list[int] = []
    train_indices: list[int] = []
    for indices in by_label.values():
        shuffled = torch.as_tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
        class_validation_size = min(len(shuffled), int(round(len(shuffled) * validation_fraction)))
        if indices and class_validation_size == 0 and validation_size > 0:
            class_validation_size = 1
        validation_indices.extend(shuffled[:class_validation_size])
        train_indices.extend(shuffled[class_validation_size:])
    while len(validation_indices) < validation_size:
        validation_indices.append(train_indices.pop())
    while len(validation_indices) > validation_size:
        train_indices.append(validation_indices.pop())
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def _class_weights(dataset: ASVspoof5FeatureDataset, indices: list[int]) -> Tensor:
    counts = torch.bincount(torch.tensor([dataset.records[index].label for index in indices]), minlength=2)
    if torch.any(counts == 0):
        return torch.ones(2, dtype=torch.float32)
    return (counts.sum() / (2.0 * counts)).to(dtype=torch.float32)


def train(args: argparse.Namespace) -> Path:
    """Train and save the best checkpoint, returning its path."""
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_kwargs = {"max_samples": args.max_samples}
    if args.metadata is not None:
        dataset_kwargs["metadata_path"] = args.metadata
    if args.audio_root is not None:
        dataset_kwargs["audio_root"] = args.audio_root
    dataset = ASVspoof5FeatureDataset(**dataset_kwargs)
    split = split_asvspoof5(
        dataset,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        audio_only=args.audio_only,
    )
    train_dataset, validation_dataset, _ = split.datasets(dataset)
    train_indices = list(split.train)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, **loader_kwargs)
    print(
        f"protocol_entries={dataset.stats.total_protocol_entries} "
        f"existing_audio={dataset.stats.existing_audio} missing_audio={dataset.stats.missing_audio} "
        f"spoof={dataset.stats.spoof} bonafide={dataset.stats.bonafide}"
    )

    model = AudioCNN().to(device)
    weights = _class_weights(dataset, train_indices) if args.class_balance else None
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            validation_loss, validation_logits, validation_labels = _run_epoch(
                model, validation_loader, criterion, device
            )
        metrics = _classification_metrics(validation_logits, validation_labels)
        metrics["eer"] = _eer(validation_logits, validation_labels)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            + " ".join(f"val_{name}={value:.4f}" for name, value in metrics.items() if value is not None)
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_metrics": metrics,
                    "class_weights": weights.tolist() if weights is not None else None,
                    "seed": args.seed,
                    "input_shape": AudioCNN.input_shape,
                    "class_mapping": {"bonafide": 0, "spoof": 1},
                    "dataset": {
                        "metadata_path": str(dataset.metadata_path),
                        "audio_root": str(dataset.audio_root),
                        "max_samples": args.max_samples,
                        "audio_only": args.audio_only,
                    },
                    "split": {
                        "seed": split.seed,
                        "validation_fraction": split.validation_fraction,
                        "test_fraction": split.test_fraction,
                        "sizes": {
                            "train": len(split.train),
                            "validation": len(split.validation),
                            "test": len(split.test),
                        },
                        "utterance_ids": {
                            name: sorted(ids) for name, ids in split_id_sets(dataset, split).items()
                        },
                    },
                    "training": {
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "num_workers": args.num_workers,
                        "class_balance": args.class_balance,
                    },
                },
                checkpoint_path,
            )

    print(f"saved_checkpoint={checkpoint_path} device={device}")
    return checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ASVspoof 5 baseline CNN detector.")
    parser.add_argument("--metadata", type=Path, default=None, help="Training TSV path")
    parser.add_argument("--audio-root", type=Path, default=None, help="Directory containing FLAC files")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for smoke testing")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--audio-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include only protocol records whose local audio file exists.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--class-balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models_cache/audio_cnn.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2 for train/validation split.")
    if args.batch_size <= 0 or args.epochs <= 0 or args.learning_rate <= 0 or args.num_workers < 0:
        raise SystemExit("batch size, epochs, and learning rate must be positive; num-workers cannot be negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())