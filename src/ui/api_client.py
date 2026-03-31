"""HTTP client utilities for the Gradio UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

VERIFY_API_URL = "http://127.0.0.1:8000/verify"
REQUEST_TIMEOUT_SECONDS = 30

RESPONSE_FIELDS = (
    "score",
    "threshold",
    "decision",
    "decision_label",
    "latency_ms",
    "model_name",
    "sample_rate",
    "enhancement",
    "threshold_mode",
    "message",
)


class APIClientError(RuntimeError):
    """Raised when backend verification request fails."""


def _normalize_response(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in RESPONSE_FIELDS if field not in payload]
    if missing:
        raise APIClientError(
            f"Backend response is missing required fields: {', '.join(missing)}"
        )

    return {
        "score": float(payload["score"]),
        "threshold": float(payload["threshold"]),
        "decision": bool(payload["decision"]),
        "decision_label": str(payload["decision_label"]),
        "latency_ms": int(payload["latency_ms"]),
        "model_name": str(payload["model_name"]),
        "sample_rate": int(payload["sample_rate"]),
        "enhancement": bool(payload["enhancement"]),
        "threshold_mode": str(payload["threshold_mode"]),
        "message": str(payload["message"]),
    }


def verify_with_api(
    enroll_audio_path: str | Path,
    test_audio_path: str | Path,
    enhancement: bool = False,
    verify_api_url: str = VERIFY_API_URL,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Send multipart verification request to FastAPI backend."""
    enroll_path = Path(enroll_audio_path)
    test_path = Path(test_audio_path)

    if not enroll_path.is_file():
        raise FileNotFoundError(f"Enrollment audio file not found: {enroll_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test audio file not found: {test_path}")

    try:
        with enroll_path.open("rb") as enroll_file, test_path.open("rb") as test_file:
            response = requests.post(
                verify_api_url,
                files={
                    "enroll_audio": (enroll_path.name, enroll_file, "audio/wav"),
                    "test_audio": (test_path.name, test_file, "audio/wav"),
                },
                data={"enhancement": str(bool(enhancement)).lower()},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
    except requests.RequestException as exc:
        raise APIClientError(
            f"Could not reach backend at {verify_api_url}. "
            "Make sure FastAPI server is running."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIClientError("Backend returned invalid JSON response.") from exc

    if not isinstance(payload, dict):
        raise APIClientError("Backend JSON response must be an object.")

    return _normalize_response(payload)
