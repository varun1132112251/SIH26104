"""
Main FastAPI application entry point.

Sets up the API server with middleware, routes, and event handlers.
"""

from fastapi import FastAPI
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile

from backend.app.core.config import settings
from ml.inference.cnn import CNNInference, CNNInferenceError


_SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models_cache" / "audio_cnn_m4_stage2_5k.pt"


@lru_cache(maxsize=1)
def get_inference_service() -> CNNInference:
    """Create the detector once and reuse it across API requests."""
    checkpoint = Path(settings.MODEL_PATH) if settings.MODEL_PATH else _DEFAULT_CHECKPOINT
    return CNNInference(checkpoint, device="auto")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": settings.APP_VERSION}

    @app.post(f"{settings.API_PREFIX}/inference")
    async def infer_audio(file: UploadFile = File(...)):
        """Classify one uploaded audio file without retaining the upload."""
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in _SUPPORTED_AUDIO_SUFFIXES:
            raise HTTPException(status_code=415, detail="Unsupported audio format.")

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                temporary_path = temporary_file.name
                shutil.copyfileobj(file.file, temporary_file)
            detector = get_inference_service()
            result = detector.predict(temporary_path)
            return {
                "decision": result.decision,
                "spoof_probability": result.spoof_probability,
                "bonafide_probability": result.bonafide_probability,
                "risk_score": result.risk_score,
                "model": Path(detector.checkpoint_path).name,
            }
        except CNNInferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            await file.close()
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    return app


# Create app instance for development and testing
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
