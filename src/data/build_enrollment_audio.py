from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VAL_MANIFEST = DATA_DIR / "manifests" / "manifest_val.csv"
TEST_MANIFEST = DATA_DIR / "manifests" / "manifest_test.csv"

VAL_OUT_DIR = DATA_DIR / "processed" / "enrollment" / "val"
TEST_OUT_DIR = DATA_DIR / "processed" / "enrollment" / "test"


def concatenate_enrollment_files(df_split: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    speakers = sorted(df_split["speaker_id"].unique())

    for speaker_id in speakers:
        speaker_df = df_split[
            (df_split["speaker_id"] == speaker_id) &
            (df_split["usage"] == "enroll")
        ].copy()

        speaker_df = speaker_df.sort_values("audio_path")

        if len(speaker_df) != 3:
            raise ValueError(f"{speaker_id} should have exactly 3 enroll files, found {len(speaker_df)}")

        audio_list = []
        sample_rate = None

        for audio_path_str in speaker_df["audio_path"]:
            audio_path = Path(audio_path_str)
            y, sr = sf.read(audio_path)

            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch for {speaker_id}: {audio_path}")

            if y.ndim > 1:
                y = np.mean(y, axis=1)

            audio_list.append(y.astype(np.float32))

        full_audio = np.concatenate(audio_list, axis=0)

        out_path = output_dir / f"{speaker_id}_enroll_full.wav"
        sf.write(out_path, full_audio, sample_rate)

        duration_sec = len(full_audio) / sample_rate
        print(f"[OK] {speaker_id} -> {out_path} ({duration_sec:.2f}s)")


def main() -> None:
    if not VAL_MANIFEST.exists():
        raise FileNotFoundError(f"Missing file: {VAL_MANIFEST}")
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(f"Missing file: {TEST_MANIFEST}")

    df_val = pd.read_csv(VAL_MANIFEST)
    df_test = pd.read_csv(TEST_MANIFEST)

    concatenate_enrollment_files(df_val, VAL_OUT_DIR)
    concatenate_enrollment_files(df_test, TEST_OUT_DIR)

    print("\nDone building enrollment audio.")


if __name__ == "__main__":
    main()
