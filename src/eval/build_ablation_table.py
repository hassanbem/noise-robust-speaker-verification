from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_ABLATION_OUT = PROJECT_ROOT / "results" / "ablation" / "noise_ablation.csv"
DEFAULT_FIGURE_OUT = PROJECT_ROOT / "results" / "figures" / "noise_ablation_bar.png"


def load_summary(path: Path, split: str, condition: str, noise_type: str, snr_db):
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(df)}")

    row = df.iloc[0].to_dict()
    row["split"] = split
    row["condition"] = condition
    row["noise_type"] = noise_type
    row["snr_db"] = snr_db
    return row


def get_summary_specs(tables_dir: Path):
    return [
        ("val", "clean", "clean", "", tables_dir / "val_clean_summary.csv"),
        ("test", "clean", "clean", "", tables_dir / "test_clean_summary.csv"),
        ("val", "noisy", "noise", 10, tables_dir / "val_noisy_snr_10_summary.csv"),
        ("val", "noisy", "noise", 5, tables_dir / "val_noisy_snr_5_summary.csv"),
        ("val", "noisy", "noise", 0, tables_dir / "val_noisy_snr_0_summary.csv"),
        ("test", "noisy", "noise", 10, tables_dir / "test_noisy_snr_10_summary.csv"),
        ("test", "noisy", "noise", 5, tables_dir / "test_noisy_snr_5_summary.csv"),
        ("test", "noisy", "noise", 0, tables_dir / "test_noisy_snr_0_summary.csv"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--ablation_out", type=Path, default=DEFAULT_ABLATION_OUT)
    parser.add_argument("--figure_out", type=Path, default=DEFAULT_FIGURE_OUT)
    args = parser.parse_args()

    rows = []
    summary_specs = get_summary_specs(args.tables_dir)
    missing_paths = [path for _, _, _, _, path in summary_specs if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing summary files:\n{missing_text}")

    for split, condition, noise_type, snr_db, path in summary_specs:
        rows.append(load_summary(path, split, condition, noise_type, snr_db))

    df = pd.DataFrame(rows)

    preferred_cols = [
        "split",
        "condition",
        "noise_type",
        "snr_db",
        "name",
        "num_trials",
        "num_positive",
        "num_negative",
        "eer",
        "eer_threshold",
        "far_at_eer_threshold",
        "frr_at_eer_threshold",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    existing_cols = [col for col in preferred_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_cols]
    df = df[existing_cols + other_cols]

    args.ablation_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.ablation_out, index=False)
    print(f"[OK] Ablation table saved to {args.ablation_out}")
    print(df.to_string(index=False))

    plot_df = df.copy()
    plot_df["label"] = plot_df.apply(
        lambda row: f"{row['split']}-{row['condition']}"
        if row["condition"] == "clean"
        else f"{row['split']}-snr{row['snr_db']}",
        axis=1,
    )

    plt.figure()
    plt.bar(plot_df["label"], plot_df["eer"])
    plt.xlabel("Condition")
    plt.ylabel("EER")
    plt.title("Clean vs Noisy EER Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure_out, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure saved to {args.figure_out}")


if __name__ == "__main__":
    main()
