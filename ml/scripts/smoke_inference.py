"""Run one bounded real-audio inference check with the trained CNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.inference.cnn import CNNInference, CNNInferenceError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CNN inference on one ASVspoof5 FLAC file.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        detector = CNNInference(args.checkpoint, device=args.device)
        result = detector.predict(args.audio)
    except CNNInferenceError as exc:
        print(f"INFERENCE_ERROR: {exc}")
        return 1
    print(f"audio={args.audio}")
    print(f"device={detector.device}")
    print(f"spoof_probability={result.spoof_probability:.8f}")
    print(f"bonafide_probability={result.bonafide_probability:.8f}")
    print(f"decision={result.decision}")
    print(f"risk_score={result.risk_score:.8f}")
    print("checkpoint_load=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())