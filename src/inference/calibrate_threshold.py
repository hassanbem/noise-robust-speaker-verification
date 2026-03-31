"""Calibrate decision threshold from validation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.eval.metrics import (
    compute_confusion_at_threshold,
    compute_eer,
    find_threshold_for_target_far,
)
from src.inference.io import parse_binary_label

DEFAULT_OUTPUT = Path("artifacts/calibration/threshold.json")
SCORE_COLUMN_ALIASES = ("score", "similarity_score", "cosine_score")
LABEL_COLUMN_ALIASES = ("label", "target", "is_target", "same_speaker", "is_same")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate threshold from validation score CSV."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="Validation scores CSV path (must include score and label columns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Threshold calibration JSON output path.",
    )
    parser.add_argument(
        "--far-target",
        type=float,
        default=0.01,
        help="Target FAR used for threshold_far_1 computation.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("eer", "far_1"),
        default="eer",
        help="Which threshold to select for runtime decisioning.",
    )
    return parser.parse_args()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _pick_column(columns: Iterable[str], aliases: Iterable[str], what: str) -> str:
    normalized = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        col = normalized.get(_normalize_name(alias))
        if col is not None:
            return col
    raise ValueError(f"Could not find {what} column. Accepted aliases: {list(aliases)}")


def _load_labels_and_scores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str, str]:
    score_col = _pick_column(df.columns, SCORE_COLUMN_ALIASES, "score")
    label_col = _pick_column(df.columns, LABEL_COLUMN_ALIASES, "label")

    labels = df[label_col].map(parse_binary_label)
    invalid_mask = labels.isna()
    if invalid_mask.any():
        bad_examples = (
            df.loc[invalid_mask, label_col].astype(str).head(5).tolist()
        )
        raise ValueError(
            f"Label column '{label_col}' contains non-binary values. "
            f"Examples: {bad_examples}"
        )

    scores = pd.to_numeric(df[score_col], errors="coerce")
    if scores.isna().any():
        raise ValueError(f"Score column '{score_col}' contains non-numeric values.")

    return labels.astype(int), scores.astype(float), label_col, score_col


def main() -> int:
    args = parse_args()
    if not args.scores.is_file():
        raise FileNotFoundError(f"Scores CSV not found: {args.scores}")

    df = pd.read_csv(args.scores)
    if df.empty:
        raise ValueError(f"Scores CSV is empty: {args.scores}")

    labels, scores, label_col, score_col = _load_labels_and_scores(df)

    eer_data = compute_eer(labels.to_numpy(), scores.to_numpy())
    threshold_eer = float(eer_data["threshold_eer"])

    far_data = find_threshold_for_target_far(
        labels.to_numpy(), scores.to_numpy(), target_far=float(args.far_target)
    )
    threshold_far_1 = (
        float(far_data["threshold_far_1"]) if far_data is not None else None
    )

    selected_mode = args.threshold_mode
    if selected_mode == "far_1" and threshold_far_1 is None:
        selected_mode = "eer"

    selected_threshold = threshold_eer
    if selected_mode == "far_1" and threshold_far_1 is not None:
        selected_threshold = threshold_far_1

    selected_conf = compute_confusion_at_threshold(
        labels.to_numpy(), scores.to_numpy(), selected_threshold
    )

    payload = {
        "threshold_eer": threshold_eer,
        "threshold_far_1": threshold_far_1,
        "selected_threshold": float(selected_threshold),
        "threshold": float(selected_threshold),
        "threshold_mode": selected_mode,
        "eer": float(eer_data["eer"]),
        "far_target": float(args.far_target),
        "selected_far": float(selected_conf["far"]),
        "selected_frr": float(selected_conf["frr"]),
        "num_trials": int(len(df)),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "scores_file": str(args.scores),
        "label_column": label_col,
        "score_column": score_col,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Threshold calibration completed")
    print(f"Input scores: {args.scores}")
    print(f"Output JSON: {args.output}")
    print(f"Threshold mode: {selected_mode}")
    print(f"Selected threshold: {selected_threshold:.6f}")
    print(f"EER: {payload['eer']:.6f}")
    if threshold_far_1 is None:
        print(f"FAR={args.far_target:.2%} threshold: unavailable")
    else:
        print(f"FAR={args.far_target:.2%} threshold: {threshold_far_1:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
