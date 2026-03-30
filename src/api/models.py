"""Pydantic models for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Simple health-check payload."""

    status: str
    message: str

    model_config = ConfigDict(extra="forbid")


class VerifyResponse(BaseModel):
    """Verification response aligned with docs/api_contract.md."""

    score: float
    threshold: float
    decision: bool
    decision_label: str
    latency_ms: int
    model_name: str
    sample_rate: int
    enhancement: bool
    threshold_mode: str
    message: str

    model_config = ConfigDict(extra="forbid")
