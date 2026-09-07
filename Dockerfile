# Dockerfile for SIH26104 Voice Cloning Detection System

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.cfg .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY backend/ backend/
COPY ml/ ml/

# Copy the trained model
COPY models_cache/ models_cache/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "backend.app.main"]
