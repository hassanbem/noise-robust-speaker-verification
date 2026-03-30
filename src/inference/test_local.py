"""Local smoke test for 3 demo WAV files.

Expected setup:
- 2 WAV files from the same speaker (you)
- 1 WAV file from a different speaker
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.models.speechbrain_verifier import SpeechBrainVerifier

DEFAULT_THRESHOLD = 0.50
DEFAULT_SAMPLES_DIR = Path("assets/demo_samples")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run local smoke tests with 3 WAV files (2 same speaker, 1 other)."
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory containing demo WAV files",
    )
    parser.add_argument(
        "--same-a",
        type=Path,
        default=None,
        help="First WAV from target speaker",
    )
    parser.add_argument(
        "--same-b",
        type=Path,
        default=None,
        help="Second WAV from target speaker",
    )
    parser.add_argument(
        "--other",
        type=Path,
        default=None,
        help="WAV from a different speaker",
    )
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
        help="Temporary fixed threshold (default: 0.50)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only",
    )
    return parser.parse_args()


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _discover_three_wavs(samples_dir: Path) -> tuple[Path, Path, Path]:
    wavs = sorted(samples_dir.glob("*.wav"))
    if len(wavs) < 3:
        raise ValueError(
            f"Need at least 3 WAV files in {samples_dir}. "
            f"Found {len(wavs)}."
        )
    return wavs[0], wavs[1], wavs[2]


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.same_a and args.same_b and args.other:
        return args.same_a, args.same_b, args.other
    return _discover_three_wavs(args.samples_dir)


def _run_case(
    verifier: SpeechBrainVerifier,
    enroll_path: Path,
    test_path: Path,
    threshold: float,
    expected_decision: bool,
    name: str,
) -> dict:
    start = time.perf_counter()
    score = verifier.score(enroll_path, test_path)
    latency_ms = (time.perf_counter() - start) * 1000.0
    decision = score >= threshold
    return {
        "case": name,
        "enroll": str(enroll_path),
        "test": str(test_path),
        "score": round(score, 6),
        "threshold": round(threshold, 6),
        "decision": decision,
        "expected_decision": expected_decision,
        "pass": decision == expected_decision,
        "latency_ms": round(latency_ms, 3),
    }


def main() -> int:
    args = parse_args()
    _ensure_file(args.config, "Config file")

    same_a, same_b, other = _resolve_paths(args)
    _ensure_file(same_a, "same-a WAV")
    _ensure_file(same_b, "same-b WAV")
    _ensure_file(other, "other WAV")

    verifier = SpeechBrainVerifier(config_path=args.config)

    cases = [
        _run_case(
            verifier=verifier,
            enroll_path=same_a,
            test_path=same_b,
            threshold=args.threshold,
            expected_decision=True,
            name="same_speaker",
        ),
        _run_case(
            verifier=verifier,
            enroll_path=same_a,
            test_path=other,
            threshold=args.threshold,
            expected_decision=False,
            name="different_speaker",
        ),
    ]

    passed = all(case["pass"] for case in cases)
    output = {
        "summary": {
            "passed": passed,
            "cases_passed": sum(1 for c in cases if c["pass"]),
            "cases_total": len(cases),
            "threshold_mode": "fixed",
            "threshold": round(args.threshold, 6),
            "model_name": verifier.config.model_name,
            "sample_rate": verifier.config.sample_rate,
            "message": (
                "local smoke test passed"
                if passed
                else "local smoke test failed (threshold is temporary)"
            ),
        },
        "cases": cases,
    }

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print("Local Smoke Test")
        print(f"Model: {verifier.config.model_name}")
        print(f"Threshold (fixed): {args.threshold:.2f}")
        print("-" * 64)
        for case in cases:
            status = "PASS" if case["pass"] else "FAIL"
            print(
                f"{status:4} | {case['case']:17} | score={case['score']:.4f} | "
                f"decision={case['decision']} | expected={case['expected_decision']}"
            )
        print("-" * 64)
        print(
            f"Summary: {output['summary']['cases_passed']}/{output['summary']['cases_total']} "
            f"cases passed"
        )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
