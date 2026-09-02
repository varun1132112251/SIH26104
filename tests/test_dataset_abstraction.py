import numpy as np

from ml.datasets import AudioSample, DatasetEntry, SampleLabel, SpoofDataset


def test_audio_sample_and_label_model_construction():
    sample = AudioSample(
        audio=np.array([0.1, -0.2, 0.3], dtype=np.float32),
        sample_rate=16000,
        label=SampleLabel.BONAFIDE,
        file_path="/tmp/real.wav",
    )

    assert sample.label == SampleLabel.BONAFIDE
    assert sample.audio.dtype == np.float32
    assert sample.sample_rate == 16000


def test_spoof_dataset_abstracts_parsing_from_rows():
    rows = [
        {"audio_path": "real1.wav", "label": "bonafide"},
        {"audio_path": "spoof1.wav", "label": "spoof"},
    ]

    dataset = SpoofDataset(rows=rows)
    entries = list(dataset)

    assert len(entries) == 2
    assert entries[0].label == SampleLabel.BONAFIDE
    assert entries[1].label == SampleLabel.SPOOF
    assert isinstance(entries[0], DatasetEntry)


def test_dataset_strips_empty_or_invalid_rows():
    rows = [
        {"audio_path": "real1.wav", "label": "bonafide"},
        {},
        {"audio_path": "spoof1.wav", "label": "unknown"},
    ]

    dataset = SpoofDataset(rows=rows)
    entries = list(dataset)

    assert len(entries) == 1
    assert entries[0].label == SampleLabel.BONAFIDE
