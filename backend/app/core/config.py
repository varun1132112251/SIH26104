"""
Application configuration management.

Handles environment variables and application settings.
Future: Add pydantic-settings for environment variable loading.
"""

import os
from typing import Optional


class Settings:
    """Application settings."""

    # Application
    APP_NAME: str = "Voice Cloning Detection System"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # API
    API_PREFIX: str = "/api/v1"
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    # ML Model
    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH", None)
    INFERENCE_BATCH_SIZE: int = int(os.getenv("INFERENCE_BATCH_SIZE", "32"))

    # Audio Processing
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_SEGMENT_DURATION: float = float(os.getenv("AUDIO_SEGMENT_DURATION", "2.0"))
    AUDIO_HOP_DURATION: float = float(os.getenv("AUDIO_HOP_DURATION", "1.0"))
    AUDIO_NORMALIZATION: str = os.getenv("AUDIO_NORMALIZATION", "peak")
    AUDIO_TARGET_PEAK_AMPLITUDE: float = float(os.getenv("AUDIO_TARGET_PEAK_AMPLITUDE", "0.95"))
    AUDIO_MIN_DURATION_SECONDS: float = float(os.getenv("AUDIO_MIN_DURATION_SECONDS", "0.05"))
    AUDIO_MAX_DURATION: float = float(os.getenv("AUDIO_MAX_DURATION", "30.0"))

    # Risk Scoring
    RISK_THRESHOLD_HIGH: float = float(os.getenv("RISK_THRESHOLD_HIGH", "0.8"))
    RISK_THRESHOLD_MEDIUM: float = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.5"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# Create global settings instance
settings = Settings()
