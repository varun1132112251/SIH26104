import numpy as np
import soundfile as sf
import torch

from ml.inference.cnn import CNNInference, CNNInferenceError
from ml.models.audio_cnn import AudioCNN


def _checkpoint(path):
    torch.manual_seed(1)
    model = AudioCNN()
    torch.save(
        {"model_state_dict": model.state_dict(), "input_shape": AudioCNN.input_shape},
        path,
    )


def test_inference_loads_checkpoint_and_returns_valid_result(tmp_path):
    checkpoint = tmp_path / "model.pt"
    audio = tmp_path / "tone.flac"
    _checkpoint(checkpoint)
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)

    result = CNNInference(checkpoint, device="cpu").predict(audio)

    assert result.decision in {"SPOOF", "BONAFIDE"}
    assert 0.0 <= result.spoof_probability <= 1.0
    assert 0.0 <= result.bonafide_probability <= 1.0
    assert np.isclose(result.spoof_probability + result.bonafide_probability, 1.0)
    assert result.risk_score == result.spoof_probability


def test_inference_handles_missing_and_invalid_checkpoints(tmp_path):
    with_missing = tmp_path / "missing.pt"
    try:
        CNNInference(with_missing, device="cpu")
    except CNNInferenceError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("Expected a missing checkpoint error")

    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a checkpoint")
    try:
        CNNInference(invalid, device="cpu")
    except CNNInferenceError:
        pass
    else:
        raise AssertionError("Expected an invalid checkpoint error")


def test_inference_handles_missing_audio(tmp_path):
    checkpoint = tmp_path / "model.pt"
    _checkpoint(checkpoint)
    detector = CNNInference(checkpoint, device="cpu")

    try:
        detector.predict(tmp_path / "missing.flac")
    except CNNInferenceError as exc:
        assert "Could not process audio" in str(exc)
    else:
        raise AssertionError("Expected a missing audio error")