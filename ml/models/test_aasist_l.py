import numpy as np

from ml.models.aasist_l import AASISTL, AASISTModelError, MODEL_SAMPLES, _pad_fixed


def test_pad_fixed_repeats_short_audio():
    audio = np.array([0.1, -0.2], dtype=np.float32)
    result = _pad_fixed(audio, max_len=5)
    assert result.shape == (5,)
    np.testing.assert_allclose(result, [0.1, -0.2, 0.1, -0.2, 0.1])


def test_pad_fixed_truncates_long_audio():
    audio = np.arange(10, dtype=np.float32)
    result = _pad_fixed(audio, max_len=5)
    np.testing.assert_allclose(result, [0, 1, 2, 3, 4])


def test_pad_fixed_rejects_empty_audio():
    try:
        _pad_fixed(np.array([], dtype=np.float32))
    except AASISTModelError:
        pass
    else:
        raise AssertionError("Empty audio must be rejected")


def test_model_constants_match_aasist_l_evaluation_window():
    assert MODEL_SAMPLES == 64600
    assert AASISTL.expected_sample_rate == 16000


def test_empty_batch_returns_empty_result_without_loading_model():
    assert AASISTL().score_batch([]) == []
