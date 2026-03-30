"""Typed response schemas for inference outputs."""

from __future__ import annotations

from typing import TypedDict

from src.inference.constants import SCORE_DECIMALS


class VerificationResponse(TypedDict):
    """Schema used by pair-scoring CLI and upcoming API responses."""

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


LocalCaseResult = TypedDict(
    "LocalCaseResult",
    {
        "case": str,
        "enroll": str,
        "test": str,
        "score": float,
        "threshold": float,
        "decision": bool,
        "expected_decision": bool,
        "pass": bool,
        "latency_ms": int,
    },
)


def _latency_to_int_ms(latency_ms: float) -> int:
    """Convert latency value in milliseconds to integer milliseconds."""
    return int(round(latency_ms))


def build_verification_response(
    *,
    score: float,
    threshold: float,
    latency_ms: float,
    model_name: str,
    sample_rate: int,
    threshold_mode: str,
    enhancement: bool = False,
    message: str = "scoring completed successfully",
) -> VerificationResponse:
    """Build a contract-aligned verification response."""
    score_value = float(round(float(score), SCORE_DECIMALS))
    threshold_value = float(round(float(threshold), SCORE_DECIMALS))
    decision_value = bool(score_value >= threshold_value)
    decision_label_value = "same speaker" if decision_value else "different speaker"
    latency_value = int(_latency_to_int_ms(float(latency_ms)))

    return {
        "score": score_value,
        "threshold": threshold_value,
        "decision": decision_value,
        "decision_label": str(decision_label_value),
        "latency_ms": latency_value,
        "model_name": model_name,
        "sample_rate": sample_rate,
        "enhancement": enhancement,
        "threshold_mode": threshold_mode,
        "message": message,
    }


def build_local_case_result(
    *,
    name: str,
    enroll_path: str,
    test_path: str,
    score: float,
    threshold: float,
    latency_ms: float,
    expected_decision: bool,
) -> LocalCaseResult:
    """Build one standardized local smoke-test case result."""
    decision = score >= threshold
    return {
        "case": name,
        "enroll": enroll_path,
        "test": test_path,
        "score": round(score, SCORE_DECIMALS),
        "threshold": round(threshold, SCORE_DECIMALS),
        "decision": decision,
        "expected_decision": expected_decision,
        "pass": decision == expected_decision,
        "latency_ms": _latency_to_int_ms(latency_ms),
    }
