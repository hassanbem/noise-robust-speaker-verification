"""Mock backend for Student 3 UI development.

This module intentionally mirrors the JSON contract in docs/api_contract.md
so the Gradio UI can be built without waiting for the real API.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import hashlib

MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
FIXED_THRESHOLD = 0.50


def _stable_score(enroll_path: str, test_path: str) -> float:
    """Generate a deterministic pseudo-score from file paths."""
    digest = hashlib.sha256(f"{enroll_path}|{test_path}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:2], "big") / 65535
    return round(0.35 + 0.55 * value, 4)


def fake_verify(
    enroll_path: str,
    test_path: str,
    threshold_mode: str = "fixed",
    enhancement: bool = False,
) -> dict:
    """Return a mock verification response using the shared contract."""
    start = perf_counter()

    if not Path(enroll_path).is_file() or not Path(test_path).is_file():
        return {
            "score": 0.0,
            "threshold": FIXED_THRESHOLD,
            "decision": False,
            "decision_label": "different speaker",
            "latency_ms": 0,
            "model_name": MODEL_NAME,
            "sample_rate": SAMPLE_RATE,
            "enhancement": enhancement,
            "threshold_mode": "fixed",
            "message": "invalid audio path(s)"
        }

    score = _stable_score(enroll_path, test_path)

    threshold = FIXED_THRESHOLD
    if threshold_mode == "eer":
        threshold = 0.58
    elif threshold_mode == "far_1":
        threshold = 0.66

    decision = score >= threshold
    latency_ms = int((perf_counter() - start) * 1000) + (30 if enhancement else 12)

    return {
        "score": score,
        "threshold": round(threshold, 2),
        "decision": decision,
        "decision_label": "same speaker" if decision else "different speaker",
        "latency_ms": latency_ms,
        "model_name": MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "enhancement": enhancement,
        "threshold_mode": threshold_mode,
        "message": "verification completed successfully"
    }
