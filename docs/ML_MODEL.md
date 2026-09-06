\# SIH26104 – ML Model Handoff



\## Model

Baseline CNN for ASVspoof 5 voice deepfake detection.



\## Input

\- Audio format: WAV / FLAC

\- Converted to mono

\- Resampled to 16 kHz

\- Peak normalized

\- First fixed-duration segment used

\- Log-Mel spectrogram extracted

\- Mel bins: 80

\- Model input shape: (1, 80, 126)



\## Classes

\- 0 = BONAFIDE

\- 1 = SPOOF



\## Checkpoint

models\_cache/audio\_cnn.pt



Best checkpoint:

\- Epoch: 5

\- Training configured for: 10 epochs

\- Batch size: 32

\- Learning rate: 0.001

\- Class balancing: enabled

\- Seed: 42



\## Decision Threshold

Calibrated validation threshold:



0.7404227



Decision:

\- spoof\_probability >= 0.7404227 → SPOOF

\- spoof\_probability < 0.7404227 → BONAFIDE



\## Test Performance

\- Accuracy: 98.98%

\- Precision: 99.17%

\- Recall: 99.69%

\- F1: 99.43%

\- ROC-AUC: 99.90%

\- EER: 1.75%

\- FAR: 7.19%

\- FRR: 0.31%



\## Real Audio Verification

Bonafide sample:

T\_0000036275

\- Spoof probability: 0.59372312

\- Decision at calibrated threshold: BONAFIDE



Known spoof:

T\_0000036286

\- Spoof probability: 0.99995029

\- Decision: SPOOF



\## Inference Interface



Input:

audio file



Output:

{

&#x20; "spoof\_probability": float,

&#x20; "bonafide\_probability": float,

&#x20; "decision": "SPOOF" | "BONAFIDE",

&#x20; "risk\_score": float

}



\## Integration Notes



Backend should:

1\. Receive uploaded audio.

2\. Pass audio to CNN inference.

3\. Return spoof probability and decision.

4\. Use threshold 0.7404227.

5\. Expose risk\_score to frontend.

