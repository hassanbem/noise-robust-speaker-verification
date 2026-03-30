"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Noise-Robust Speaker Verification API",
    version="0.1.0",
    description="Baseline verification API with ECAPA-TDNN and fixed thresholding.",
)
app.include_router(router)
