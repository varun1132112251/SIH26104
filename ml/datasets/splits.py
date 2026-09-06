"""Deterministic, stratified splits for ASVspoof records."""

from __future__ import annotations

from dataclasses import dataclass
import random

from torch.utils.data import Dataset, Subset

from .asvspoof5_features import ASVspoof5FeatureDataset


@dataclass(frozen=True)
class ASVspoof5Split:
    """Index-based train, validation, and test partitions."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    seed: int
    validation_fraction: float
    test_fraction: float
    audio_only: bool

    def datasets(self, dataset: Dataset) -> tuple[Subset, Subset, Subset]:
        return (
            Subset(dataset, list(self.train)),
            Subset(dataset, list(self.validation)),
            Subset(dataset, list(self.test)),
        )


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")


def split_asvspoof5(
    dataset: ASVspoof5FeatureDataset,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    audio_only: bool = True,
) -> ASVspoof5Split:
    """Create deterministic, stratified, ID-disjoint dataset partitions."""
    _validate_fractions(validation_fraction, test_fraction)
    candidates = [
        index
        for index, record in enumerate(dataset.records)
        if not audio_only or record.audio_path.is_file()
    ]
    if len(candidates) < 3:
        raise ValueError("At least three eligible samples are required for a three-way split.")

    by_label: dict[int, list[int]] = {}
    for index in candidates:
        by_label.setdefault(dataset.records[index].label, []).append(index)
    if len(by_label) < 2:
        raise ValueError("A split requires both bonafide and spoof samples.")

    rng = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in sorted(by_label):
        indices = sorted(by_label[label], key=lambda item: dataset.records[item].utterance_id)
        rng.shuffle(indices)
        count = len(indices)
        test_size = max(1, round(count * test_fraction))
        validation_size = max(1, round(count * validation_fraction))
        if test_size + validation_size >= count:
            raise ValueError("Each class needs at least one training sample.")
        test.extend(indices[:test_size])
        validation.extend(indices[test_size : test_size + validation_size])
        train.extend(indices[test_size + validation_size :])

    split = ASVspoof5Split(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        audio_only=audio_only,
    )
    _assert_disjoint_ids(dataset, split)
    return split


def split_id_sets(dataset: ASVspoof5FeatureDataset, split: ASVspoof5Split) -> dict[str, set[str]]:
    """Return utterance IDs for each partition, useful for audits and tests."""
    return {
        name: {dataset.records[index].utterance_id for index in indices}
        for name, indices in {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        }.items()
    }


def _assert_disjoint_ids(dataset: ASVspoof5FeatureDataset, split: ASVspoof5Split) -> None:
    id_sets = split_id_sets(dataset, split)
    names = tuple(id_sets)
    for left_position, left_name in enumerate(names):
        for right_name in names[left_position + 1 :]:
            if id_sets[left_name] & id_sets[right_name]:
                raise ValueError("Dataset split contains overlapping utterance IDs.")


__all__ = ["ASVspoof5Split", "split_asvspoof5", "split_id_sets"]