"""CLI utility to score one enrollment/test audio pair."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.inference.constants import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_MODEL_CONFIG_PATH,
    FIXED_THRESHOLD_MODE,
)
from src.inference.io import ensure_file_exists, load_threshold_settings
from src.inference.schema import VerificationResponse, build_verification_response
from src.models.speechbrain_verifier import SpeechBrainVerifier


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute speaker verification score for one audio pair."
    )
    parser.add_argument("--enroll", type=Path, required=True, help="Enrollment WAV path")
    parser.add_argument("--test", type=Path, required=True, help="Test WAV path")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="Model config YAML path",
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Calibration JSON path (used when --threshold is not provided)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold override (temporary fixed threshold)",
    )
    return parser.parse_args()


def resolve_threshold(
    *, threshold_override: float | None, threshold_file: Path
) -> tuple[float, str]:
    """Resolve threshold/mode from CLI override or calibration JSON."""
    if threshold_override is not None:
        return threshold_override, FIXED_THRESHOLD_MODE
    return load_threshold_settings(threshold_file)


def score_pair_response(
    *,
    verifier: SpeechBrainVerifier,
    enroll_path: Path,
    test_path: Path,
    threshold: float,
    threshold_mode: str,
) -> VerificationResponse:
    """Return one contract-aligned score response for an audio pair."""
    ensure_file_exists(enroll_path, "Enrollment audio")
    ensure_file_exists(test_path, "Test audio")

    start = time.perf_counter()
    score = verifier.score(enroll_path, test_path)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return build_verification_response(
        score=score,
        threshold=threshold,
        latency_ms=latency_ms,
        model_name=verifier.config.model_name,
        sample_rate=verifier.config.sample_rate,
        threshold_mode=threshold_mode,
    )


def main() -> int:
    """Run CLI workflow and print JSON response."""
    args = parse_args()
    ensure_file_exists(args.config, "Config")

    verifier = SpeechBrainVerifier(config_path=args.config)
    threshold, threshold_mode = resolve_threshold(
        threshold_override=args.threshold,
        threshold_file=args.threshold_file,
    )
    response = score_pair_response(
        verifier=verifier,
        enroll_path=args.enroll,
        test_path=args.test,
        threshold=threshold,
        threshold_mode=threshold_mode,
    )
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
