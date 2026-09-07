# V-SHIELD — Voice Deepfake Detection System

> **AI-powered detection of synthetic, cloned, and spoofed speech for safer voice-based interactions.**

V-SHIELD is a Smart India Hackathon (SIH) prototype that analyzes an uploaded audio sample and estimates whether the speech is **bona-fide human speech** or **synthetic/spoofed speech**. The system combines a React/TypeScript frontend, a FastAPI inference API, audio preprocessing, and a trained PyTorch CNN model.

## 🎯 Problem

Voice cloning and synthetic speech can be used to impersonate people and support social-engineering or fraud attempts. V-SHIELD provides a prototype detection layer that helps users avoid relying on a suspicious voice alone.

The current workflow supports:

- Uploading an audio sample from a browser
- Audio validation and preprocessing
- Anti-spoofing CNN inference
- Spoof and bona-fide probabilities
- Risk scoring and classification
- Clear user-facing recommendations for additional verification

> **Prototype scope:** V-SHIELD currently focuses on single-audio-sample analysis. It is not a production-grade identity verification, forensic, or fraud-prevention system.

## ✨ Key Features

### Audio Analysis
- WAV, FLAC, OGG, AIFF, and AIF input support in the current API
- Mono conversion, resampling, normalization, and fixed-duration segmentation
- Log-Mel spectrogram feature extraction

### Machine Learning
- PyTorch-based convolutional neural network (CNN)
- Spoof and bona-fide probabilities
- Configurable decision threshold
- Risk score derived from spoof probability
- Inference service reuses the loaded model within the application process

### Web Application
- React + TypeScript + Vite frontend
- Browser audio upload interface
- Browser recording interface
- Detection status and result visualization
- Confidence and risk-level presentation
- Secondary-verification recommendation for suspicious results

### Backend API
- FastAPI REST API
- `GET /health` health check
- `POST /api/v1/predict` audio detection endpoint
- Temporary upload handling with cleanup after inference
- Docker-based deployment

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                         V-SHIELD                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  React + TypeScript Frontend                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Audio upload / recording                               │  │
│  │ Detection status                                       │  │
│  │ Classification • confidence • risk • recommendation   │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │ HTTP multipart/form-data           │
│                          ▼                                   │
│  FastAPI Backend                                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ /health                                                │  │
│  │ /api/v1/predict                                        │  │
│  │ Validation • temporary file handling • response       │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ML Inference Pipeline                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Audio preprocessing → Log-Mel → PyTorch CNN            │  │
│  │                         ↓                              │  │
│  │              probabilities + risk score                │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          ▼                                   │
│                   Detection Result                          │
└──────────────────────────────────────────────────────────────┘
```

## 🧠 Machine Learning Pipeline

```text
Audio file
   ↓
Waveform / format validation
   ↓
Mono conversion + resampling
   ↓
Normalization
   ↓
Fixed-duration crop / padding
   ↓
Log-Mel spectrogram
   ↓
PyTorch CNN
   ↓
Spoof probability
   ↓
Decision + risk score
```

The repository also contains ASVspoof5 dataset utilities for protocol parsing, lazy audio-path resolution, label handling, and deterministic feature preparation.

### Integrated model

```text
models_cache/audio_cnn_m4_stage2_5k.pt
```

## 📊 Detection Output

Example API response:

```json
{
  "decision": "SPOOF",
  "spoof_probability": 0.9721,
  "bonafide_probability": 0.0279,
  "risk_score": 0.9721,
  "model": "audio_cnn_m4_stage2_5k.pt"
}
```

The frontend presents this as:

- **Classification:** Synthetic / Human
- **Confidence:** percentage for the selected class
- **Risk:** Low / Medium / High / Critical
- **Recommendation:** additional verification guidance

> Model confidence is not certainty. Results should be treated as decision support, not proof of identity or fraud.

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI, Uvicorn |
| ML | PyTorch, NumPy, SciPy, audio feature extraction |
| Dataset pipeline | ASVspoof5 utilities |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Deployment | Docker, Render, Vercel |
| Development | Git, GitHub, Python virtual environment |

## 📁 Project Structure

```text
SIH26104/
├── backend/
│   ├── app/
│   │   ├── core/              # Configuration
│   │   └── main.py             # FastAPI application + API routes
│   └── tests/                  # Backend tests
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main UI
│   │   ├── services/            # Backend API integration
│   │   ├── styles.css           # UI styling
│   │   └── types.ts             # Frontend types
│   ├── package.json
│   └── vite.config.ts
├── ml/
│   ├── datasets/               # Dataset and ASVspoof utilities
│   ├── evaluation/             # Metrics and evaluation
│   ├── inference/              # CNN inference
│   ├── models/                 # PyTorch model definitions
│   ├── preprocessing/          # Audio preprocessing and features
│   └── scripts/                # Training and evaluation scripts
├── models_cache/               # Trained model artifacts
├── tests/                      # ML/integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── setup.cfg
└── README.md
```

## 🚀 Quick Start — Local Demo

For a live SIH presentation, running the complete application locally is recommended. It avoids cold-start delays from a free cloud instance and can use the local machine's available compute resources.

### Prerequisites

- Python 3.9+
- Node.js and npm
- Git
- FFmpeg where required by the audio workflow
- Optional: compatible NVIDIA GPU/PyTorch installation for faster inference

### 1. Clone

```bash
git clone https://github.com/varun1132112251/SIH26104.git
cd SIH26104
```

### 2. Create the Python environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the backend

From the repository root:

```bash
python -m backend.app.main
```

Backend:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

### 5. Configure the frontend

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_API=false
VITE_DETECTION_ENDPOINT=/api/v1/predict
```

### 6. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:5173
```

### 7. Run a detection

1. Upload a supported audio file.
2. Click **Analyze audio**.
3. Wait for the backend to run the ML model.
4. Review classification, confidence, risk, and recommendation.

## 🐳 Docker

Build and run the backend:

```bash
docker build -t v-shield-api .
docker run -p 8000:8000 v-shield-api
```

Or run the development stack:

```bash
docker-compose up --build
```

## 🔌 API Reference

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Audio Prediction

```http
POST /api/v1/predict
Content-Type: multipart/form-data
```

Form field:

```text
file=<audio file>
```

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@sample.wav"
```

Current API-supported extensions:

```text
.wav  .flac  .ogg  .aiff  .aif
```

## 🧪 Testing

Run the integrated test suite:

```bash
python -m pytest --import-mode=importlib
```

Run with coverage:

```bash
python -m pytest --import-mode=importlib --cov=backend --cov=ml
```

## ☁️ Deployment

The prototype supports a cloud deployment architecture of:

```text
Vercel (Frontend)
       ↓
Render (FastAPI Backend)
       ↓
PyTorch CNN Model
```

The deployed backend health endpoint is:

```text
https://sih26104.onrender.com/health
```

For SIH judging, local execution is recommended when low latency is important because a free cloud instance can sleep after inactivity.

## ⚠️ Current Prototype Limitations

- Detection performance depends on training data, model quality, and recording conditions.
- The API currently supports a defined set of audio formats; some browser recording formats may require conversion before submission.
- A free cloud backend can experience cold-start latency after inactivity.
- A spoof probability is not definitive proof of malicious intent.
- Production deployments should restrict CORS to trusted frontend origins and add authentication, rate limiting, observability, and stronger request controls.
- Persistent audit trails, external verification integrations, and production-grade monitoring are outside the current prototype scope.

## 🔮 Future Enhancements

- Stronger anti-spoofing architectures and model ensembles
- Larger and more diverse training/evaluation datasets
- Robustness testing across codecs, microphones, noise levels, and languages
- Real-time streaming inference
- Speaker verification combined with spoof detection
- Context-aware risk scoring
- Secondary verification workflows
- Authentication and role-based access
- Audit logging and monitoring
- Model versioning and automated evaluation
- Production-grade cloud scaling

## 👥 Team

**Smart India Hackathon — SIH26104**

The prototype was developed collaboratively across ML, backend, frontend, testing, integration, and deployment workstreams.

| Workstream | Primary responsibility |
|---|---|
| M1 | Backend/API and system integration |
| M2 | Backend/data and supporting infrastructure |
| M3 | Project support and integration |
| M4 | Machine learning, training, evaluation, and inference |
| M5 | Frontend and user experience |
| M6 | Integration, testing, deployment, and presentation support |

## 📌 Project Status

**Status:** Functional SIH prototype  
**Product:** V-SHIELD — Voice Deepfake Detection  
**Primary branch:** `master`

The repository contains the integrated frontend, FastAPI backend, ML inference pipeline, trained model artifact, tests, and Docker deployment configuration.

## 📄 License

This project is developed as an academic/hackathon prototype for **Smart India Hackathon (SIH)**.

A formal open-source license can be added if the project is released publicly.

---

<p align="center">
  <strong>V-SHIELD</strong><br>
  Voice Deepfake Detection for Safer Digital Communication
</p>
