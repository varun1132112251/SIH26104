import numpy as np
import soundfile as sf
import torch

from ml.datasets.asvspoof5_features import ASVspoof5FeatureDataset


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