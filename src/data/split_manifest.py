from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
INPUT_PATH = MANIFEST_DIR / "manifest_all.csv"
VAL_OUTPUT = MANIFEST_DIR / "manifest_val.csv"
TEST_OUTPUT = MANIFEST_DIR / "manifest_test.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = {
        "speaker_id",
        "audio_path",
        "usage",
        "split",
        "duration_sec",
        "sample_rate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in manifest_all.csv: {missing}")

    speakers = sorted(df["speaker_id"].unique())

    if len(speakers) != 16:
        print(f"[WARN] Expected 16 speakers, found {len(speakers)}")

    midpoint = len(speakers) // 2
    val_speakers = speakers[:midpoint]
    test_speakers = speakers[midpoint:]

    df_val = df[df["speaker_id"].isin(val_speakers)].copy()
    df_test = df[df["speaker_id"].isin(test_speakers)].copy()

    df_val["split"] = "val"
    df_test["split"] = "test"

    VAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_val.to_csv(VAL_OUTPUT, index=False)
    df_test.to_csv(TEST_OUTPUT, index=False)

    print(f"[OK] Validation speakers: {val_speakers}")
    print(f"[OK] Test speakers: {test_speakers}")
    print(f"[OK] Saved: {VAL_OUTPUT} ({len(df_val)} rows)")
    print(f"[OK] Saved: {TEST_OUTPUT} ({len(df_test)} rows)")


if __name__ == "__main__":
    main()
