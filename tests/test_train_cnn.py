import numpy as np

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset, ASVspoof5Record
from ml.scripts.train_cnn import _classification_metrics, _eer, _split_dataset


def _dataset_with_labels(tmp_path):
    metadata = tmp_path / "train.tsv"
    audio_root = tmp_path / "flac_T"
    audio_root.mkdir()
    metadata.write_text(
        "T_0 T_0 F - - - - - bonafide -\n"
        "T_1 T_1 F - - - - - spoof -\n"
        "T_2 T_2 F - - - - - bonafide -\n"
        "T_3 T_3 F - - - - - spoof -\n",
        encoding="utf-8",
    )
    return ASVspoof5FeatureDataset(metadata, audio_root)


def test_split_is_deterministic_and_contains_both_classes(tmp_path):
    dataset = _dataset_with_labels(tmp_path)
    first_train, first_validation = _split_dataset(dataset, 0.5, 42)
    second_train, second_validation = _split_dataset(dataset, 0.5, 42)

    assert list(first_train.indices) == list(second_train.indices)
    assert list(first_validation.indices) == list(second_validation.indices)
    assert {dataset.records[index].label for index in first_validation.indices} == {0, 1}


def test_metrics_and_eer_are_computed_for_binary_validation():
    logits = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 3.0], [4.0, 0.0]])
    labels = np.array([0, 1, 1, 0])

    metrics = _classification_metrics(logits, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert _eer(logits, labels) == 0.0


def test_eer_is_none_when_validation_has_one_class():
    logits = np.array([[1.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0])

    assert _eer(logits, labels) is None