# SIH26104: Voice Cloning Detection System

## Project Objective

**AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks**

This project develops a comprehensive system to:
- **Detect** voice cloning and deepfake audio in real-time
- **Analyze** audio signals using advanced ML models for anti-spoofing
- **Score** risk levels contextually (who, when, where, content)
- **Prevent** impersonation attacks through secondary verification
- **Respond** with appropriate security measures

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                         │
│  - User interface for audio submission/monitoring           │
│  - Risk visualization and alerts                            │
│  - Secondary verification challenges                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               Backend API (FastAPI)                         │
│  - Audio intake and validation                              │
│  - Request routing and rate limiting                        │
│  - Risk scoring and decision logic                          │
│  - Secondary verification coordination                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│             ML Inference Engine                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Audio Preprocessing                                   │ │
│  │  - Format normalization, resampling                    │ │
│  │  - Feature extraction (MFCC, spectral)                 │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Anti-Spoofing Model                                   │ │
│  │  - Speaker verification and liveness detection         │ │
│  │  - Deepfake/synthesis detection                        │ │
│  │  - Confidence scores and feature importance            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Real-Time Inference                                   │ │
│  │  - Streaming audio processing                          │ │
│  │  - Batch inference with caching                        │ │
│  │  - Model versioning and A/B testing                    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│         Risk Scoring & Context Engine                       │
│  - Rolling risk assessment (temporal analysis)              │
│  - Contextual factors (caller ID, caller history, etc.)     │
│  - Threat level classification                              │
│  - Prevention recommendations                               │
└─────────────────────────────────────────────────────────────┘
```

### Microservices

1. **backend/app/** - FastAPI application
   - `api/` - REST endpoint handlers
   - `core/` - Configuration and shared utilities
   - `models/` - Pydantic schemas
   - `services/` - Business logic

2. **ml/** - Machine learning pipeline
   - `preprocessing/` - Audio format normalization, resampling, segmentation, and validation
   - `datasets/` - Generic audio sample abstraction and dataset-specific parsing hooks
   - `models/` - Model definitions and loading (not implemented yet)
   - `inference/` - Real-time inference engine (not implemented yet)
   - `evaluation/` - Benchmarking and validation (not implemented yet)

3. **frontend/** - React/TypeScript UI (TODO)

4. **data/** - Dataset and model artifact storage
   - `raw/` - Original audio samples
   - `processed/` - Preprocessed features
   - `samples/` - Test/demo samples

## Current Status

### ✅ Completed
- [x] Project repository structure
- [x] Python package scaffolding
- [x] Backend framework setup (FastAPI)
- [x] Configuration management
- [x] Test infrastructure (pytest)
- [x] Docker Compose setup
- [x] Dependency management

### M4: ASVspoof5 data pipeline

The M4 pipeline reads `ASVspoof5.train.tsv` lazily: field 1 is mapped from
`T_XXXXXXXXXX` to `<audio-root>/T_XXXXXXXXXX.flac`, and field 8 is encoded as
bonafide `0` or spoof `1`. The index reports protocol, present, missing, and
class counts without copying dataset files into the repository. At access time,
`AudioPreprocessor` converts audio to mono, resamples it to the configured
sample rate, normalizes it, and crops or pads a fixed-duration segment.
`MelSpectrogramExtractor` then produces a deterministic log-Mel tensor for the
PyTorch `Dataset` and `DataLoader` (default shape `(batch, 1, 80, 126)`).

Run a bounded smoke test with the local dataset:

```bash
python -m ml.scripts.smoke_asvspoof5 --metadata C:\Users\Administrator\Downloads\ASVspoof5\protocols\ASVspoof5.train.tsv --audio-root C:\Users\Administrator\Downloads\ASVspoof5\flac_T --max-samples 500
```

### 🚧 In Progress / TODO

#### Phase 1: ML Foundation
- [x] Audio preprocessing pipeline foundation (format normalization, resampling, segmentation, and safe handling of edge cases)
- [x] Dataset abstraction foundation for audio samples and labels (generic interface + spoof dataset adapter placeholder)
- [ ] Model evaluation framework
- [ ] Baseline model selection/integration

> The current implementation covers the preprocessing and data-layer foundation only. The actual anti-spoofing model and dataset integration for ASVspoof or other corpora are not implemented yet.

#### Phase 2: Inference Engine
- [ ] Real-time audio streaming ingestion
- [ ] Batch inference optimization
- [ ] Model caching and versioning
- [ ] Inference API endpoints

#### Phase 3: Risk Scoring
- [ ] Risk scoring logic
- [ ] Contextual risk analysis
- [ ] Temporal rolling assessment
- [ ] Decision/recommendation engine

#### Phase 4: Prevention & Response
- [ ] Secondary verification challenges
- [ ] Alert and notification system
- [ ] Audit logging
- [ ] Integration points for external systems

#### Phase 5: Frontend
- [ ] React UI
- [ ] Real-time audio recording interface
- [ ] Risk visualization dashboard
- [ ] Admin/monitoring interface

#### Phase 6: Deployment & Monitoring
- [ ] Kubernetes manifests
- [ ] Monitoring and alerting
- [ ] Performance profiling
- [ ] Documentation and runbooks

## Tech Stack

### Current
- **Backend:** Python 3.9+, FastAPI, Uvicorn
- **ML:** NumPy, SciPy
- **Testing:** pytest, pytest-asyncio, pytest-cov
- **DevOps:** Docker, Docker Compose

### Planned
- **ML/Audio:** PyTorch, librosa, torchaudio, scikit-learn
- **Frontend:** React, TypeScript
- **Database:** PostgreSQL (for persistence)
- **Caching:** Redis (already in docker-compose.yml)
- **Deployment:** Kubernetes
- **Monitoring:** Prometheus, Grafana

## Project Structure

```
SIH26104/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # API route handlers
│   │   ├── core/           # Config and utilities
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── main.py         # FastAPI app factory
│   └── tests/              # Backend unit tests
│
├── ml/                     # ML pipeline and models
│   ├── datasets/           # Dataset loaders
│   ├── preprocessing/      # Audio feature extraction
│   ├── models/             # Model definitions
│   ├── inference/          # Real-time inference
│   ├── evaluation/         # Benchmarking
│   └── scripts/            # Training/utility scripts
│
├── frontend/               # React UI (TODO)
│
├── data/                   # Datasets and artifacts
│   ├── raw/                # Original audio
│   ├── processed/          # Processed features
│   └── samples/            # Test samples
│
├── docs/                   # Documentation
├── tests/                  # Integration tests
├── .gitignore
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project config
├── setup.cfg               # Setup configuration
└── docker-compose.yml      # Docker Compose config
```

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose (optional)
- Git

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourorg/sih26104.git
   cd SIH26104
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run tests:
   ```bash
   pytest
   ```

5. Start the backend API:
   ```bash
   python -m backend.app.main
   ```
   The API will be available at `http://localhost:8000`

### Docker Development

```bash
docker-compose up
```

This starts:
- Backend API on port 8000
- Redis on port 6379

## Configuration

Configuration is managed via environment variables in `backend/app/core/config.py`. 

Example `.env` file:
```
DEBUG=False
API_PORT=8000
API_HOST=0.0.0.0
AUDIO_SAMPLE_RATE=16000
LOG_LEVEL=INFO
```

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=backend --cov=ml
```

Run specific test file:
```bash
pytest backend/tests/test_api.py
```

## Contributing

1. Create a feature branch
2. Make changes following the code style
3. Add tests for new functionality
4. Run tests and linting
5. Submit a pull request

## Code Style

- **Formatting:** Black (line length: 100)
- **Linting:** Flake8
- **Type checking:** mypy
- **Sorting:** isort

Run code quality checks:
```bash
black .
flake8 .
mypy backend ml
```

## Documentation

Documentation will be generated using Sphinx. To build:
```bash
cd docs
make html
```

## License

MIT License - See LICENSE file for details

## Contact

SIH26104 Team

---

**Last Updated:** 2024-09-02  
**Status:** Alpha - Active Development
