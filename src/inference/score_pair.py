"""CLI utility to score one enrollment/test audio pair."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.models.speechbrain_verifier import SpeechBrainVerifier

DEFAULT_THRESHOLD = 0.50


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
        default=Path("configs/model/ecapa.yaml"),
        help="Model config YAML path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold (temporary fixed threshold, default: 0.50)",
    )
    return parser.parse_args()


def _ensure_file(path: Path, label: str) -> None:
    """Fail fast if a required file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")


def main() -> int:
    """Run CLI workflow and print JSON response."""
    args = parse_args()
    _ensure_file(args.enroll, "Enrollment audio")
    _ensure_file(args.test, "Test audio")
    _ensure_file(args.config, "Config")

    verifier = SpeechBrainVerifier(config_path=args.config)

    start = time.perf_counter()
    score = verifier.score(args.enroll, args.test)
    latency_ms = (time.perf_counter() - start) * 1000.0

    decision = score >= args.threshold
    response = {
        "score": round(score, 6),
        "threshold": round(args.threshold, 6),
        "decision": decision,
        "decision_label": "same speaker" if decision else "different speaker",
        "latency_ms": round(latency_ms, 3),
        "model_name": verifier.config.model_name,
        "sample_rate": verifier.config.sample_rate,
        "enhancement": False,
        "threshold_mode": "fixed",
        "message": "scoring completed successfully",
    }
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
