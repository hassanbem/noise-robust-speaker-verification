"""Simple HTTP client for the Gradio speaker-verification UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

DEFAULT_VERIFY_URL = "http://127.0.0.1:8000/verify"
REQUEST_TIMEOUT_SECONDS = 30

REQUIRED_RESPONSE_FIELDS = (
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


def _ensure_file(path_value: str | Path | None, label: str) -> Path:
    if path_value is None:
        raise FileNotFoundError(f"{label} file is missing.")

    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise APIClientError("Backend JSON response must be an object.")

    missing = [name for name in REQUIRED_RESPONSE_FIELDS if name not in payload]
    if missing:
        missing_text = ", ".join(missing)
        raise APIClientError(f"Backend response is missing required fields: {missing_text}")

    return {name: payload[name] for name in REQUIRED_RESPONSE_FIELDS}


def verify_with_api(
    enroll_audio_path: str | Path,
    test_audio_path: str | Path,
    enhancement: bool = False,
    verify_api_url: str = DEFAULT_VERIFY_URL,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Call POST /verify with multipart audio files and return backend JSON."""
    enroll_path = _ensure_file(enroll_audio_path, "Enrollment audio")
    test_path = _ensure_file(test_audio_path, "Test audio")

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
    except requests.Timeout as exc:
        raise APIClientError(
            f"Backend request timed out after {timeout_seconds}s. URL: {verify_api_url}"
        ) from exc
    except requests.ConnectionError as exc:
        raise APIClientError(
            f"Cannot connect to backend at {verify_api_url}. Is FastAPI running?"
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.text if exc.response is not None else str(exc)
        raise APIClientError(f"Backend returned HTTP {status}: {detail}") from exc
    except requests.RequestException as exc:
        raise APIClientError(f"Backend request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIClientError("Backend returned invalid JSON response.") from exc

    return _validate_payload(payload)
