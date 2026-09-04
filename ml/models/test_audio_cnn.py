import pytest
import torch

from ml.models.audio_cnn import AudioCNN


def test_audio_cnn_returns_two_class_logits():
    model = AudioCNN()

    logits = model(torch.randn(4, 1, 80, 126))

    assert logits.shape == (4, 2)
    assert logits.dtype == torch.float32


def test_audio_cnn_rejects_wrong_feature_shape():
    model = AudioCNN()

    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.randn(2, 1, 80, 125))