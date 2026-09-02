"""
Test fixtures and utilities.
"""

import pytest


@pytest.fixture
def sample_audio_data():
    """Fixture providing sample audio data for testing.
    
    Returns:
        bytes: Sample audio data.
    """
    # TODO: Create or load sample audio for testing
    return b""


@pytest.fixture
def sample_voice_sample():
    """Fixture providing sample voice samples for reference.
    
    Returns:
        dict: Sample voice data.
    """
    return {
        "audio": b"",
        "sample_rate": 16000,
        "duration": 5.0,
    }
