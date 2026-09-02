import numpy as np
from scipy.io import wavfile

from ml.preprocessing.audio import AudioConfig, AudioPreprocessor


def test_preprocessor_resamples_and_normalizes_audio(tmp_path):
    sr = 8000
    samples = np.linspace(0, 2 * np.pi * 440, int(sr * 0.25), endpoint=False)
    waveform = np.stack([np.sin(samples), np.sin(samples)], axis=1)
    wav_path = tmp_path / "tone.wav"
    wavfile.write(wav_path, sr, (waveform * 0.5 * np.iinfo(np.int16).max).astype(np.int16))

    preprocessor = AudioPreprocessor()
    audio, out_sr = preprocessor.load_audio(wav_path)

    assert out_sr == 16000
    assert audio.dtype.kind == "f"
    assert audio.ndim == 1
    assert audio.size > 0
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6


def test_segment_audio_uses_configured_duration_and_hop():
    preprocessor = AudioPreprocessor()
    config = AudioConfig(sample_rate=16000, segment_duration=0.5, hop_duration=0.25)
    audio = np.zeros(int(16000 * 1.5), dtype=np.float32)
    audio[:8000] = 0.2

    segments = preprocessor.segment_audio(audio, config=config)

    assert len(segments) == 5
    assert all(segment.shape[0] == 8000 for segment in segments)


def test_preprocessor_handles_silence_and_short_audio():
    preprocessor = AudioPreprocessor()
    silence = np.zeros(1000, dtype=np.float32)
    processed = preprocessor.preprocess(silence, sample_rate=8000)
    assert processed.shape[0] > 0
    assert np.allclose(processed, 0.0)

    short = np.array([0.1, -0.2, 0.0], dtype=np.float32)
    processed_short = preprocessor.preprocess(short, sample_rate=8000)
    assert processed_short.shape[0] > 0
    assert np.all(np.isfinite(processed_short))
