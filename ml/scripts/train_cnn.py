"""Train the baseline CNN detector on ASVspoof 5 features."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset
from ml.models.audio_cnn import AudioCNN


def _run_epoch(
    model: AudioCNN,
    loader: DataLoader[tuple[Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            loss = criterion(model(features), labels)
            if training:
                loss.backward()
                optimizer.step()

        item_count = labels.size(0)
        total_loss += loss.item() * item_count
        total_items += item_count

    if total_items == 0:
        raise ValueError("Cannot run an epoch with an empty DataLoader.")
    return total_loss / total_items


def _accuracy(model: AudioCNN, loader: DataLoader[tuple[Tensor, int]], device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features.to(device)).argmax(dim=1).cpu()
            labels = torch.as_tensor(labels, dtype=torch.long)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total if total else 0.0


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
    return random_split(dataset, [len(dataset) - validation_size, validation_size], generator=generator)


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
    train_dataset, validation_dataset = _split_dataset(dataset, args.validation_fraction, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            validation_loss = _run_epoch(model, validation_loader, criterion, device)
        validation_accuracy = _accuracy(model, validation_loader, device)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "input_shape": AudioCNN.input_shape,
                    "class_mapping": {"bonafide": 0, "spoof": 1},
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("models_cache/audio_cnn.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples < 2:
        raise SystemExit("--max-samples must be at least 2 for train/validation split.")
    if args.batch_size <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise SystemExit("batch size, epochs, and learning rate must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())