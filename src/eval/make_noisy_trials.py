from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
ENROLL_DIR = DATA_DIR / "processed" / "enrollment"
AUG_ROOT = DATA_DIR / "augmented" / "noise"

VAL_MANIFEST = MANIFEST_DIR / "manifest_val.csv"
TEST_MANIFEST = MANIFEST_DIR / "manifest_test.csv"

SNR_LEVELS = [0, 5, 10]

VAL_ENROLL_DIR = ENROLL_DIR / "val"
TEST_ENROLL_DIR = ENROLL_DIR / "test"

VAL_OUT = MANIFEST_DIR / "trials_val_noisy.csv"
TEST_OUT = MANIFEST_DIR / "trials_test_noisy.csv"


def build_noisy_trials(df_split: pd.DataFrame, enroll_dir: Path, split_name: str) -> pd.DataFrame:
    speakers = sorted(df_split["speaker_id"].unique())
    clean_verify_df = df_split[df_split["usage"] == "verify_clean"].copy()
    clean_verify_df = clean_verify_df.sort_values(["speaker_id", "audio_path"])

    rows = []

    for snr_db in SNR_LEVELS:
        for i, speaker_id in enumerate(speakers):
            enroll_path = enroll_dir / f"{speaker_id}_enroll_full.wav"

            if not enroll_path.exists():
                raise FileNotFoundError(f"Missing enrollment file: {enroll_path}")

            # Positive trials
            pos_df = clean_verify_df[clean_verify_df["speaker_id"] == speaker_id].copy()
            for _, row in pos_df.iterrows():
                clean_test_path = Path(row["audio_path"])
                noisy_test_path = AUG_ROOT / f"snr_{snr_db}" / speaker_id / clean_test_path.name

                if not noisy_test_path.exists():
                    raise FileNotFoundError(f"Missing noisy verify file: {noisy_test_path}")

                rows.append({
                    "enroll_path": str(enroll_path).replace("\\", "/"),
                    "test_path": str(noisy_test_path).replace("\\", "/"),
                    "label": 1,
                    "noise_type": "noise",
                    "snr_db": snr_db,
                    "speaker_id_enroll": speaker_id,
                    "speaker_id_test": speaker_id,
                    "split": split_name,
                })

            # Negative trials
            other_speakers = [s for s in speakers if s != speaker_id]
            neg_speaker = other_speakers[i % len(other_speakers)]
            neg_df = clean_verify_df[clean_verify_df["speaker_id"] == neg_speaker].head(2)

            for _, row in neg_df.iterrows():
                clean_test_path = Path(row["audio_path"])
                noisy_test_path = AUG_ROOT / f"snr_{snr_db}" / neg_speaker / clean_test_path.name

                if not noisy_test_path.exists():
                    raise FileNotFoundError(f"Missing noisy verify file: {noisy_test_path}")

                rows.append({
                    "enroll_path": str(enroll_path).replace("\\", "/"),
                    "test_path": str(noisy_test_path).replace("\\", "/"),
                    "label": 0,
                    "noise_type": "noise",
                    "snr_db": snr_db,
                    "speaker_id_enroll": speaker_id,
                    "speaker_id_test": neg_speaker,
                    "split": split_name,
                })

    return pd.DataFrame(rows)


def main() -> None:
    df_val = pd.read_csv(VAL_MANIFEST)
    df_test = pd.read_csv(TEST_MANIFEST)

    trials_val = build_noisy_trials(df_val, VAL_ENROLL_DIR, "val")
    trials_test = build_noisy_trials(df_test, TEST_ENROLL_DIR, "test")

    VAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    trials_val.to_csv(VAL_OUT, index=False)
    trials_test.to_csv(TEST_OUT, index=False)

    print(f"[OK] Saved {VAL_OUT} with {len(trials_val)} rows")
    print(f"[OK] Saved {TEST_OUT} with {len(trials_test)} rows")


if __name__ == "__main__":
    main()
