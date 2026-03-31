"""Evaluation metrics and plotting utilities for speaker verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import det_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def _as_binary_numpy(labels: Any) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    unique_values = set(np.unique(labels_arr).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Labels must be binary (0/1). Got values: {sorted(unique_values)}")
    if len(unique_values) < 2:
        raise ValueError("Need both positive and negative labels to compute ROC/EER.")
    return labels_arr


def _as_score_numpy(scores: Any) -> np.ndarray:
    return np.asarray(scores, dtype=float).reshape(-1)


def compute_roc_data(labels: Any, scores: Any) -> dict[str, np.ndarray]:
    """Compute ROC arrays used by several evaluation functions."""
    labels_arr = _as_binary_numpy(labels)
    scores_arr = _as_score_numpy(scores)
    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr, drop_intermediate=False)
    fnr = 1.0 - tpr
    return {
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "thresholds": thresholds,
    }


def compute_eer(labels: Any, scores: Any) -> dict[str, Any]:
    """Compute equal error rate and corresponding threshold."""
    roc_data = compute_roc_data(labels, scores)
    fpr = roc_data["fpr"]
    fnr = roc_data["fnr"]
    thresholds = roc_data["thresholds"]

    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return {
        "eer": eer,
        "threshold_eer": threshold,
        "eer_index": idx,
        "fpr": fpr,
        "tpr": roc_data["tpr"],
        "fnr": fnr,
        "thresholds": thresholds,
    }


def compute_confusion_at_threshold(
    labels: Any, scores: Any, threshold: float
) -> dict[str, float | int]:
    """Compute confusion counts and derived metrics at one threshold."""
    labels_arr = np.asarray(labels, dtype=int).reshape(-1)
    scores_arr = _as_score_numpy(scores)
    preds = (scores_arr >= float(threshold)).astype(int)

    tp = int(np.sum((preds == 1) & (labels_arr == 1)))
    tn = int(np.sum((preds == 0) & (labels_arr == 0)))
    fp = int(np.sum((preds == 1) & (labels_arr == 0)))
    fn = int(np.sum((preds == 0) & (labels_arr == 1)))

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    accuracy = float((tp + tn) / len(labels_arr)) if len(labels_arr) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "far": far,
        "frr": frr,
        "accuracy": accuracy,
    }


def find_threshold_for_target_far(
    labels: Any, scores: Any, target_far: float
) -> dict[str, float | int] | None:
    """Find threshold that satisfies FAR <= target with best TPR among candidates."""
    if not (0.0 < float(target_far) < 1.0):
        raise ValueError("target_far must be in (0, 1).")

    roc_data = compute_roc_data(labels, scores)
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    fnr = roc_data["fnr"]
    thresholds = roc_data["thresholds"]

    valid = np.where(fpr <= float(target_far))[0]
    if valid.size == 0:
        return None

    best_tpr = np.max(tpr[valid])
    best_by_tpr = valid[np.where(tpr[valid] == best_tpr)[0]]
    gaps = np.abs(fpr[best_by_tpr] - float(target_far))
    idx = int(best_by_tpr[int(np.argmin(gaps))])

    return {
        "threshold_far_1": float(thresholds[idx]),
        "far": float(fpr[idx]),
        "frr": float(fnr[idx]),
        "tpr": float(tpr[idx]),
        "index": idx,
    }


def plot_roc_curve(fpr: Any, tpr: Any, out_path: Path) -> None:
    """Save ROC curve PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_det_curve(labels: Any, scores: Any, out_path: Path) -> None:
    """Save DET curve PNG."""
    labels_arr = _as_binary_numpy(labels)
    scores_arr = _as_score_numpy(scores)

    fpr, fnr, _ = det_curve(labels_arr, scores_arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, fnr, label="DET")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Legacy CLI: evaluate one scored CSV and write summary + ROC/DET."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing scored CSV: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    required_cols = {"label", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labels = df["label"].astype(int).to_numpy()
    scores = df["score"].astype(float).to_numpy()

    eer_data = compute_eer(labels, scores)
    confusion = compute_confusion_at_threshold(labels, scores, eer_data["threshold_eer"])

    summary = pd.DataFrame(
        [
            {
                "name": args.name,
                "num_trials": int(len(df)),
                "num_positive": int((labels == 1).sum()),
                "num_negative": int((labels == 0).sum()),
                "eer": eer_data["eer"],
                "eer_threshold": eer_data["threshold_eer"],
                "far_at_eer_threshold": confusion["far"],
                "frr_at_eer_threshold": confusion["frr"],
                "accuracy_at_eer_threshold": confusion["accuracy"],
                "tp": confusion["tp"],
                "tn": confusion["tn"],
                "fp": confusion["fp"],
                "fn": confusion["fn"],
            }
        ]
    )

    summary_out = TABLES_DIR / f"{args.name}_summary.csv"
    roc_out = FIGURES_DIR / f"{args.name}_roc.png"
    det_out = FIGURES_DIR / f"{args.name}_det.png"

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    plot_roc_curve(eer_data["fpr"], eer_data["tpr"], roc_out)
    plot_det_curve(labels, scores, det_out)

    print(f"[OK] Summary saved to {summary_out}")
    print(f"[OK] ROC saved to {roc_out}")
    print(f"[OK] DET saved to {det_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
