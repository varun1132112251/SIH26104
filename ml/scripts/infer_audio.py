"""Run AASIST-L on a WAV/audio file and print the bona-fide score."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.models.aasist_l import AASISTL, AASISTModelError
from ml.preprocessing.audio import AudioPreprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AASIST-L voice anti-spoofing inference.")
    parser.add_argument("audio", type=Path, help="Path to a WAV/audio file")
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"Audio file not found: {args.audio}")

    try:
        preprocessor = AudioPreprocessor()
        waveform, sample_rate = preprocessor.load_audio(args.audio)

        detector = AASISTL()
        detector.load()
        score = detector.score(waveform)

        print(f"AUDIO: {args.audio}")
        print(f"SAMPLE_RATE: {sample_rate} Hz")
        print(f"SAMPLES: {waveform.size}")
        print(f"BONA_FIDE_SCORE: {score:.6f}")
        print("NOTE: Higher AASIST-L score indicates more bona-fide speech.")
        print("SIH risk calibration is intentionally not applied yet.")
        return 0
    except (AASISTModelError, ValueError, OSError) as exc:
        print(f"INFERENCE_ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
