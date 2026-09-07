import numpy as np
import soundfile as sf
import torch

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset
from ml.datasets.splits import split_asvspoof5, split_id_sets
from ml.evaluation.metrics import classification_metrics, select_threshold
from ml.scripts.evaluate_cnn import evaluate_checkpoint
from ml.scripts.train_cnn import build_parser, train


def _dataset(tmp_path, count=10):
    metadata = tmp_path / "train.tsv"
    audio_root = tmp_path / "flac_T"
    audio_root.mkdir()
    rows = []
    for index in range(count):
        label = "bonafide" if index % 2 == 0 else "spoof"
        utterance_id = f"T_{index:04d}"
        rows.append(f"T_{index} {utterance_id} F - - - - - {label} -\n")
        sf.write(audio_root / f"{utterance_id}.flac", np.zeros(16_000, dtype=np.float32), 16_000)
    metadata.write_text("".join(rows), encoding="utf-8")
    return ASVspoof5FeatureDataset(metadata, audio_root)


def test_three_way_split_is_deterministic_stratified_and_id_disjoint(tmp_path):
    dataset = _dataset(tmp_path, count=20)
    first = split_asvspoof5(dataset, validation_fraction=0.2, test_fraction=0.2, seed=7)
    second = split_asvspoof5(dataset, validation_fraction=0.2, test_fraction=0.2, seed=7)

    assert first == second
    id_sets = split_id_sets(dataset, first)
    assert not (id_sets["train"] & id_sets["validation"])
    assert not (id_sets["train"] & id_sets["test"])
    assert not (id_sets["validation"] & id_sets["test"])
    assert [len(first.train), len(first.validation), len(first.test)] == [12, 4, 4]
    assert all(len({dataset.records[index].label for index in indices}) == 2 for indices in (first.train, first.validation, first.test))


def test_threshold_selection_and_metrics_are_binary_and_thresholded():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.4, 0.45, 0.6])
    threshold = select_threshold(scores, labels)
    metrics = classification_metrics(scores, labels, threshold)

    assert threshold == 0.45
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["eer"] == 0.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["far"] == 0.0
    assert metrics["frr"] == 0.0


def test_bounded_training_saves_metadata_and_evaluator_uses_test_split(tmp_path):
    dataset = _dataset(tmp_path, count=10)
    checkpoint = tmp_path / "checkpoint.pt"
    args = build_parser().parse_args(
        [
            "--metadata",
            str(dataset.metadata_path),
            "--audio-root",
            str(dataset.audio_root),
            "--max-samples",
            "10",
            "--validation-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    )
    train(args)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert saved["split"]["sizes"] == {"train": 6, "validation": 2, "test": 2}
    assert set(saved["split"]["utterance_ids"]) == {"train", "validation", "test"}
    result = evaluate_checkpoint(checkpoint, device="cpu", batch_size=2)
    assert result["threshold_selection"]["criterion"].startswith("maximum validation F1")
    assert set(result["test"]) >= {"accuracy", "precision", "recall", "f1", "eer", "confusion_matrix", "far", "frr"}