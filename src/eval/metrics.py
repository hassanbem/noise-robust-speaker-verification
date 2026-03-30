from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import det_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def compute_eer(labels, scores):
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = roc_thresholds[idx]

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "roc_thresholds": roc_thresholds,
    }


def compute_far_frr_at_threshold(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    preds = (scores >= threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "far": float(far),
        "frr": float(frr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def plot_roc(fpr, tpr, out_path: Path):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_det(labels, scores, out_path: Path):
    fpr, fnr, _ = det_curve(labels, scores)
    plt.figure()
    plt.plot(fpr, fnr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve")
    plt.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    name = args.name

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing scored CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {"label", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    labels = df["label"].values
    scores = df["score"].values

    eer_data = compute_eer(labels, scores)
    threshold_data = compute_far_frr_at_threshold(
        labels, scores, eer_data["eer_threshold"]
    )

    summary = pd.DataFrame(
        [
            {
                "name": name,
                "num_trials": len(df),
                "num_positive": int((df["label"] == 1).sum()),
                "num_negative": int((df["label"] == 0).sum()),
                "eer": eer_data["eer"],
                "eer_threshold": eer_data["eer_threshold"],
                "far_at_eer_threshold": threshold_data["far"],
                "frr_at_eer_threshold": threshold_data["frr"],
                "tp": threshold_data["tp"],
                "tn": threshold_data["tn"],
                "fp": threshold_data["fp"],
                "fn": threshold_data["fn"],
            }
        ]
    )

    summary_out = TABLES_DIR / f"{name}_summary.csv"
    roc_out = FIGURES_DIR / f"{name}_roc.png"
    det_out = FIGURES_DIR / f"{name}_det.png"

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    plot_roc(eer_data["fpr"], eer_data["tpr"], roc_out)
    plot_det(labels, scores, det_out)

    print(f"[OK] Summary saved to {summary_out}")
    print(f"[OK] ROC saved to {roc_out}")
    print(f"[OK] DET saved to {det_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
