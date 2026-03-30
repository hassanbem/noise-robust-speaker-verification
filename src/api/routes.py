"""FastAPI routes for speaker verification."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.models import HealthResponse, VerifyResponse
from src.api.service import verify_audio_pair
from src.api.utils import cleanup_temp_files, save_upload_to_temp

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Quick readiness endpoint."""
    return HealthResponse(status="ok", message="API is running")


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    enroll_audio: UploadFile = File(...),
    test_audio: UploadFile = File(...),
    enhancement: bool = Form(False),
) -> VerifyResponse:
    """Verify speaker similarity for one enrollment/test pair."""
    temp_files: list[Path] = []
    try:
        enroll_path = await save_upload_to_temp(enroll_audio)
        test_path = await save_upload_to_temp(test_audio)
        temp_files.extend([enroll_path, test_path])

        response = verify_audio_pair(
            enroll_path=enroll_path,
            test_path=test_path,
            enhancement=enhancement,
        )
        return VerifyResponse(**response)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
    finally:
        cleanup_temp_files(temp_files)
