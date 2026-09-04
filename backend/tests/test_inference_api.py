from io import BytesIO
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


@pytest.mark.asyncio
async def test_inference_endpoint_returns_detector_result(monkeypatch):
    monkeypatch.setattr(main, "get_inference_service", lambda: _FakeDetector())
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/inference",
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
            "/api/v1/inference",
            files={"file": ("sample.txt", BytesIO(b"not audio"), "text/plain")},
        )

    assert response.status_code == 415


@pytest.mark.asyncio
async def test_inference_endpoint_requires_file():
    transport = httpx.ASGITransport(app=main.create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/v1/inference")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_real_asvspoof5_audio_inference_smoke():
    audio_path = Path(r"C:\SIH26104\data\asvspoof5\flac_T\T_0000000000.flac")
    checkpoint_path = Path(r"C:\SIH26104\models_cache\audio_cnn_m4_stage2_5k.pt")
    if not audio_path.is_file() or not checkpoint_path.is_file():
        pytest.skip("Local ASVspoof5 smoke artifacts are unavailable")

    transport = httpx.ASGITransport(app=main.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/inference",
            files={"file": (audio_path.name, audio_path.read_bytes(), "audio/flac")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] in {"SPOOF", "BONAFIDE"}
    assert 0.0 <= payload["spoof_probability"] <= 1.0
    assert 0.0 <= payload["bonafide_probability"] <= 1.0
    assert 0.0 <= payload["risk_score"] <= 1.0