"""Batch-score verification trials from a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.inference.constants import DEFAULT_CALIBRATION_PATH, DEFAULT_MODEL_CONFIG_PATH
from src.inference.io import (
    ensure_file_exists,
    load_trials_csv,
    parse_binary_label,
    resolve_audio_path,
)
from src.inference.score_pair import resolve_threshold, score_pair_response
from src.models.speechbrain_verifier import SpeechBrainVerifier


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Score a trials CSV and export per-trial scores to CSV."
    )
    parser.add_argument(
        "--trials",
        type=Path,
        required=True,
        help="Input trials CSV path (must contain enroll_path,test_path columns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scores/trials_scored.csv"),
        help="Output scored CSV path",
    )
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
        help="Decision threshold override",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of trials to score",
    )
    parser.add_argument(
        "--enhancement",
        action="store_true",
        help="Set enhancement=true in output contract fields",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print summary as JSON instead of plain text",
    )
    return parser.parse_args()


def _write_scored_csv(
    output_path: Path, rows: list[dict[str, Any]], original_columns: list[str]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_columns = [
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
    ]
    fieldnames = original_columns + [c for c in new_columns if c not in original_columns]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    ensure_file_exists(args.config, "Config")

    trials_rows, original_columns, column_mapping = load_trials_csv(args.trials)

    verifier = SpeechBrainVerifier(config_path=args.config)
    threshold, threshold_mode = resolve_threshold(
        threshold_override=args.threshold,
        threshold_file=args.threshold_file,
    )

    scored_rows: list[dict[str, Any]] = []
    tp = tn = fp = fn = 0
    labeled_count = 0

    for idx, row in enumerate(trials_rows):
        if args.limit is not None and idx >= args.limit:
            break

        try:
            enroll_path = resolve_audio_path(
                row["enroll_path"],
                trials_csv_path=args.trials,
            )
            test_path = resolve_audio_path(
                row["test_path"],
                trials_csv_path=args.trials,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Invalid trial row #{idx + 2} in {args.trials}.\n{exc}"
            ) from exc

        response = score_pair_response(
            verifier=verifier,
            enroll_path=enroll_path,
            test_path=test_path,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
        response["enhancement"] = bool(args.enhancement)
        response["message"] = "verification completed successfully"

        scored_row = dict(row)
        scored_row.update(
            {
                "score": response["score"],
                "threshold": response["threshold"],
                "decision": response["decision"],
                "decision_label": response["decision_label"],
                "latency_ms": response["latency_ms"],
                "model_name": response["model_name"],
                "sample_rate": response["sample_rate"],
                "enhancement": response["enhancement"],
                "threshold_mode": response["threshold_mode"],
                "message": response["message"],
            }
        )
        scored_rows.append(scored_row)

        expected = parse_binary_label(row.get("label"))
        if expected is None:
            continue
        labeled_count += 1
        predicted = bool(response["decision"])
        if predicted and expected == 1:
            tp += 1
        elif (not predicted) and expected == 0:
            tn += 1
        elif predicted and expected == 0:
            fp += 1
        else:
            fn += 1

    _write_scored_csv(args.output, scored_rows, original_columns)

    summary: dict[str, Any] = {
        "trials_input": str(args.trials),
        "scores_output": str(args.output),
        "scored_trials": len(scored_rows),
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "model_name": verifier.config.model_name,
        "sample_rate": verifier.config.sample_rate,
        "column_mapping": column_mapping,
    }
    if labeled_count > 0:
        summary.update(
            {
                "labeled_trials": labeled_count,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": (tp + tn) / labeled_count,
            }
        )

    if args.json_summary:
        print(json.dumps(summary, indent=2))
    else:
        print("Trials scoring completed")
        print(f"Input: {args.trials}")
        print(f"Output: {args.output}")
        print(f"Scored trials: {len(scored_rows)}")
        print(f"Threshold ({threshold_mode}): {threshold:.4f}")
        print(f"Column mapping: {column_mapping}")
        if labeled_count > 0:
            print(f"Labeled trials: {labeled_count}")
            print(f"Confusion counts: TP={tp} TN={tn} FP={fp} FN={fn}")
            print(f"Accuracy: {(tp + tn) / labeled_count:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
