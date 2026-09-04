import numpy as np
import soundfile as sf
import torch

from torch.utils.data import DataLoader

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset, parse_asvspoof5_protocol


def _write_training_metadata(path, utterance_id="T_test", label="bonafide"):
    path.write_text(f"T_0001 {utterance_id} F - - - - - {label} -\n", encoding="utf-8")


def test_asvspoof5_metadata_maps_utterance_to_flac(tmp_path):
    metadata_path = tmp_path / "ASVspoof5.train.tsv"
    audio_root = tmp_path / "flac_T"
    audio_root.mkdir()
    _write_training_metadata(metadata_path, utterance_id="T_1234", label="spoof")

    dataset = ASVspoof5FeatureDataset(metadata_path, audio_root, max_samples=1)

    assert len(dataset) == 1
    assert dataset.records[0].utterance_id == "T_1234"
    assert dataset.records[0].audio_path == audio_root / "T_1234.flac"
    assert dataset.records[0].label == 1
    assert dataset.stats.total_protocol_entries == 1
    assert dataset.stats.existing_audio == 0
    assert dataset.stats.missing_audio == 1


def test_protocol_parser_counts_labels_and_existing_audio(tmp_path):
    metadata_path = tmp_path / "train.tsv"
    audio_root = tmp_path / "flac_T"
    audio_root.mkdir()
    metadata_path.write_text(
        "T_spk1 T_real F - - - AC3 A01 bonafide -\n"
        "T_spk2 T_fake M - - - AC3 A02 spoof -\n",
        encoding="utf-8",
    )
    (audio_root / "T_real.flac").touch()

    records, stats = parse_asvspoof5_protocol(metadata_path, audio_root)
    assert [record.label for record in records] == [0, 1]
    assert stats.total_protocol_entries == 2
    assert stats.existing_audio == 1
    assert stats.missing_audio == 1
    assert stats.spoof == 1
    assert stats.bonafide == 1


def test_asvspoof5_dataset_returns_fixed_tensor_shape(tmp_path):
    metadata_path = tmp_path / "ASVspoof5.train.tsv"
    audio_root = tmp_path / "flac_T"
    audio_root.mkdir()
    _write_training_metadata(metadata_path, utterance_id="T_5678", label="bonafide")
    sf.write(audio_root / "T_5678.flac", np.zeros(16_000, dtype=np.float32), 16_000)

    dataset = ASVspoof5FeatureDataset(metadata_path, audio_root, max_samples=1)
    features, label = dataset[0]

    assert tuple(features.shape) == (1, 80, 126)
    assert features.dtype == torch.float32
    assert label == 0


def test_missing_or_corrupt_audio_returns_safe_batch(tmp_path):
    metadata_path = tmp_path / "ASVspoof5.train.tsv"
    audio_root = tmp_path / "flac_T"
    audio_root.mkdir()
    metadata_path.write_text(
        "T_0001 T_missing F - - - - - spoof -\n"
        "T_0002 T_corrupt F - - - - - bonafide -\n",
        encoding="utf-8",
    )
    (audio_root / "T_corrupt.flac").write_bytes(b"not audio")
    dataset = ASVspoof5FeatureDataset(metadata_path, audio_root)
    features, labels = next(iter(DataLoader(dataset, batch_size=2)))
    assert tuple(features.shape) == (2, 1, 80, 126)
    assert labels.tolist() == [1, 0]