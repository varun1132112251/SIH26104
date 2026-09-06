from io import BytesIO
import os
from pathlib import Path

import httpx
import pytest

from backend.app import main


class _FakeResult:
    decision = "SPOOF"
    spoof_probability = 0.8
    bonafide_probability = 0.2
    risk_score = 0.8


class _FakeDetector:
    checkpoint_path = "models_cache/test.pt"

    def predict(self, audio_path):
        assert audio_path
        return _FakeResult()


class _FailingDetector:
    checkpoint_path = "models_cache/test.pt"

    def predict(self, audio_path):
        raise RuntimeError("Unable to process the uploaded audio.")


@pytest.mark.asyncio
async def test_inference_endpoint_returns_detector_result(monkeypatch):
    monkeypatch.setattr(main, "get_inference_service", lambda: _FakeDetector())
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("sample.wav", BytesIO(b"synthetic audio"), "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "decision": "SPOOF",
        "spoof_probability": 0.8,
        "bonafide_probability": 0.2,
        "risk_score": 0.8,
        "model": "test.pt",
    }


@pytest.mark.asyncio
async def test_inference_endpoint_rejects_unsupported_audio(monkeypatch):
    monkeypatch.setattr(main, "get_inference_service", lambda: _FakeDetector())
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("sample.txt", BytesIO(b"not audio"), "text/plain")},
        )

    assert response.status_code == 415


@pytest.mark.asyncio
async def test_inference_endpoint_requires_file():
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/v1/predict")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_returns_inference_error(monkeypatch):
    monkeypatch.setattr(main, "get_inference_service", lambda: _FailingDetector())
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("sample.wav", BytesIO(b"synthetic audio"), "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unable to process the uploaded audio."


@pytest.mark.asyncio
async def test_predict_endpoint_removes_temporary_upload(monkeypatch):
    observed_paths = []

    class _CleanupDetector(_FakeDetector):
        def predict(self, audio_path):
            path = Path(audio_path)
            observed_paths.append(path)
            assert path.is_file()
            return _FakeResult()

    monkeypatch.setattr(main, "get_inference_service", lambda: _CleanupDetector())
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("sample.wav", BytesIO(b"synthetic audio"), "audio/wav")},
        )

    assert response.status_code == 200
    assert len(observed_paths) == 1
    assert not observed_paths[0].exists()


@pytest.mark.asyncio
async def test_inference_endpoint_remains_a_predict_alias(monkeypatch):
    monkeypatch.setattr(main, "get_inference_service", lambda: _FakeDetector())
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/inference",
            files={"file": ("sample.wav", BytesIO(b"synthetic audio"), "audio/wav")},
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_real_asvspoof5_audio_inference_smoke():
    audio_path = Path(os.getenv("ASVSPOOF5_SMOKE_AUDIO", ""))
    checkpoint_path = Path(os.getenv("MODEL_PATH", ""))
    if not audio_path.is_file() or not checkpoint_path.is_file():
        pytest.skip("Set ASVSPOOF5_SMOKE_AUDIO and MODEL_PATH to run real-audio inference.")

    transport = httpx.ASGITransport(app=main.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": (audio_path.name, audio_path.read_bytes(), "audio/flac")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] in {"SPOOF", "BONAFIDE"}
    assert 0.0 <= payload["spoof_probability"] <= 1.0
    assert 0.0 <= payload["bonafide_probability"] <= 1.0
    assert 0.0 <= payload["risk_score"] <= 1.0
