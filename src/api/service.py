"""Service layer for API endpoints."""

from __future__ import annotations

from pathlib import Path

from src.inference.constants import DEFAULT_CALIBRATION_PATH, DEFAULT_MODEL_CONFIG_PATH
from src.inference.io import ensure_file_exists
from src.inference.score_pair import resolve_threshold, score_pair_response
from src.inference.schema import VerificationResponse
from src.models.speechbrain_verifier import SpeechBrainVerifier

_VERIFIER_CACHE: dict[Path, SpeechBrainVerifier] = {}


def get_verifier(config_path: Path = DEFAULT_MODEL_CONFIG_PATH) -> SpeechBrainVerifier:
    """Load verifier once per config path and reuse it."""
    key = config_path.resolve()
    cached = _VERIFIER_CACHE.get(key)
    if cached is None:
        _VERIFIER_CACHE[key] = SpeechBrainVerifier(config_path=key)
    return _VERIFIER_CACHE[key]


def verify_audio_pair(
    *,
    enroll_path: Path,
    test_path: Path,
    enhancement: bool = False,
    threshold_override: float | None = None,
    threshold_file: Path = DEFAULT_CALIBRATION_PATH,
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH,
) -> VerificationResponse:
    """Run one verification request end-to-end."""
    ensure_file_exists(config_path, "Model config")
    verifier = get_verifier(config_path)
    threshold, threshold_mode = resolve_threshold(
        threshold_override=threshold_override,
        threshold_file=threshold_file,
    )
    response = score_pair_response(
        verifier=verifier,
        enroll_path=enroll_path,
        test_path=test_path,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )
    response["enhancement"] = bool(enhancement)
    response["message"] = "verification completed successfully"
    return response
