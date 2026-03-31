"""Evaluate scored trials CSV and export report-ready artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.eval.metrics import (
    compute_confusion_at_threshold,
    compute_eer,
    find_threshold_for_target_far,
    plot_det_curve,
    plot_roc_curve,
)
from src.inference.constants import DEFAULT_CALIBRATION_PATH
from src.inference.io import load_threshold_settings, parse_binary_label

DEFAULT_OUTPUT_DIR = Path("results/eval")
DEFAULT_ABLATION_OUT = Path("results/ablation/noise_ablation.csv")
DEFAULT_TABLES_DIR = Path("results/tables")
DEFAULT_FAR_TARGET = 0.01

SCORE_COLUMN_ALIASES = ("score", "similarity_score", "cosine_score")
LABEL_COLUMN_ALIASES = ("label", "target", "is_target", "same_speaker", "is_same")
NOISE_COLUMN_ALIASES = ("noise_type", "noise")
SNR_COLUMN_ALIASES = ("snr_db", "snr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate score CSV: EER, FAR/FRR/accuracy, ROC/DET, ablations."
    )
    parser.add_argument("--scores", type=Path, required=True, help="Input scored CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary JSON/CSV and ROC/DET plots.",
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Calibration JSON path used when --threshold is not provided.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override.",
    )
    parser.add_argument(
        "--far-target",
        type=float,
        default=DEFAULT_FAR_TARGET,
        help="FAR target used for FAR-constrained threshold analysis.",
    )
    parser.add_argument(
        "--ablation-out",
        type=Path,
        default=DEFAULT_ABLATION_OUT,
        help="Combined noise robustness ablation CSV output path.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Directory for grouped summary tables (noise_type / snr_db).",
    )
    return parser.parse_args()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        found = normalized.get(_normalize_name(alias))
        if found is not None:
            return found
    return None


def _require_column(columns: Iterable[str], aliases: Iterable[str], what: str) -> str:
    found = _find_column(columns, aliases)
    if found is None:
        raise ValueError(f"Missing {what} column. Accepted aliases: {list(aliases)}")
    return found


def _prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    label_col = _require_column(df.columns, LABEL_COLUMN_ALIASES, "label")
    score_col = _require_column(df.columns, SCORE_COLUMN_ALIASES, "score")

    labels = df[label_col].map(parse_binary_label)
    if labels.isna().any():
        bad = df.loc[labels.isna(), label_col].astype(str).head(5).tolist()
        raise ValueError(
            f"Could not parse some labels in '{label_col}' to binary 0/1. Examples: {bad}"
        )

    scores = pd.to_numeric(df[score_col], errors="coerce")
    if scores.isna().any():
        raise ValueError(f"Score column '{score_col}' contains non-numeric values.")

    prepared = df.copy()
    prepared["_label_bin"] = labels.astype(int)
    prepared["_score"] = scores.astype(float)
    return prepared, label_col, score_col


def _group_summary(
    df: pd.DataFrame, group_col: str, *, threshold: float, threshold_mode: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = df.groupby(group_col, dropna=False)
    for group_value, group_df in grouped:
        labels = group_df["_label_bin"].to_numpy(dtype=int)
        scores = group_df["_score"].to_numpy(dtype=float)
        confusion = compute_confusion_at_threshold(labels, scores, threshold)

        eer = None
        eer_threshold = None
        if len(np.unique(labels)) == 2:
            eer_data = compute_eer(labels, scores)
            eer = float(eer_data["eer"])
            eer_threshold = float(eer_data["threshold_eer"])

        normalized_group = "unknown"
        if group_value is not None:
            text = str(group_value).strip()
            if text:
                normalized_group = text

        rows.append(
            {
                group_col: normalized_group,
                "num_trials": int(len(group_df)),
                "num_positive": int((labels == 1).sum()),
                "num_negative": int((labels == 0).sum()),
                "threshold_mode": threshold_mode,
                "threshold": float(threshold),
                "eer": eer,
                "eer_threshold": eer_threshold,
                "accuracy": float(confusion["accuracy"]),
                "far": float(confusion["far"]),
                "frr": float(confusion["frr"]),
                "tp": int(confusion["tp"]),
                "tn": int(confusion["tn"]),
                "fp": int(confusion["fp"]),
                "fn": int(confusion["fn"]),
            }
        )

    return pd.DataFrame(rows).sort_values(by=group_col).reset_index(drop=True)


def _write_noise_ablation(
    df: pd.DataFrame,
    *,
    threshold: float,
    threshold_mode: str,
    noise_col: str | None,
    snr_col: str | None,
    tables_dir: Path,
    ablation_out: Path,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[pd.DataFrame] = []

    if noise_col is not None:
        noise_table = _group_summary(
            df, noise_col, threshold=threshold, threshold_mode=threshold_mode
        )
        noise_table_out = tables_dir / "noise_type_summary.csv"
        noise_table.to_csv(noise_table_out, index=False)
        outputs["noise_type_summary"] = str(noise_table_out)

        noise_ablation = noise_table.rename(columns={noise_col: "group_value"})
        noise_ablation.insert(0, "group_by", "noise_type")
        combined_rows.append(noise_ablation)

    if snr_col is not None:
        snr_table = _group_summary(
            df, snr_col, threshold=threshold, threshold_mode=threshold_mode
        )
        snr_table_out = tables_dir / "snr_db_summary.csv"
        snr_table.to_csv(snr_table_out, index=False)
        outputs["snr_db_summary"] = str(snr_table_out)

        snr_ablation = snr_table.rename(columns={snr_col: "group_value"})
        snr_ablation.insert(0, "group_by", "snr_db")
        combined_rows.append(snr_ablation)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        ablation_out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(ablation_out, index=False)
        outputs["noise_ablation"] = str(ablation_out)

    return outputs


def main() -> int:
    args = parse_args()
    if not args.scores.is_file():
        raise FileNotFoundError(f"Scores CSV not found: {args.scores}")

    df = pd.read_csv(args.scores)
    if df.empty:
        raise ValueError(f"Scores CSV is empty: {args.scores}")

    prepared_df, label_col, score_col = _prepare_dataframe(df)
    labels = prepared_df["_label_bin"].to_numpy(dtype=int)
    scores = prepared_df["_score"].to_numpy(dtype=float)

    eer_data = compute_eer(labels, scores)
    far_target_data = find_threshold_for_target_far(
        labels, scores, target_far=float(args.far_target)
    )

    if args.threshold is not None:
        selected_threshold = float(args.threshold)
        threshold_mode = "fixed"
    else:
        selected_threshold, threshold_mode = load_threshold_settings(args.threshold_file)

    selected_metrics = compute_confusion_at_threshold(labels, scores, selected_threshold)

    name = args.scores.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = args.output_dir / f"{name}_summary.json"
    summary_csv_path = args.output_dir / f"{name}_summary.csv"
    roc_path = args.output_dir / f"{name}_roc.png"
    det_path = args.output_dir / f"{name}_det.png"

    summary_payload: dict[str, object] = {
        "scores_file": str(args.scores),
        "num_trials": int(len(prepared_df)),
        "num_positive": int((labels == 1).sum()),
        "num_negative": int((labels == 0).sum()),
        "label_column": label_col,
        "score_column": score_col,
        "threshold_mode": threshold_mode,
        "selected_threshold": float(selected_threshold),
        "eer": float(eer_data["eer"]),
        "threshold_eer": float(eer_data["threshold_eer"]),
        "accuracy": float(selected_metrics["accuracy"]),
        "far": float(selected_metrics["far"]),
        "frr": float(selected_metrics["frr"]),
        "tp": int(selected_metrics["tp"]),
        "tn": int(selected_metrics["tn"]),
        "fp": int(selected_metrics["fp"]),
        "fn": int(selected_metrics["fn"]),
        "far_target": float(args.far_target),
        "threshold_far_1": (
            None if far_target_data is None else float(far_target_data["threshold_far_1"])
        ),
    }

    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    pd.DataFrame([summary_payload]).to_csv(summary_csv_path, index=False)

    plot_roc_curve(eer_data["fpr"], eer_data["tpr"], roc_path)
    plot_det_curve(labels, scores, det_path)

    noise_col = _find_column(prepared_df.columns, NOISE_COLUMN_ALIASES)
    snr_col = _find_column(prepared_df.columns, SNR_COLUMN_ALIASES)
    ablation_outputs = _write_noise_ablation(
        prepared_df,
        threshold=selected_threshold,
        threshold_mode=threshold_mode,
        noise_col=noise_col,
        snr_col=snr_col,
        tables_dir=args.tables_dir,
        ablation_out=args.ablation_out,
    )

    print("Evaluation completed")
    print(f"Scores: {args.scores}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"ROC: {roc_path}")
    print(f"DET: {det_path}")
    if ablation_outputs:
        print(f"Ablation outputs: {json.dumps(ablation_outputs, indent=2)}")
    else:
        print("Ablation outputs: skipped (noise_type/snr_db columns not found).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
