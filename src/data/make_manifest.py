from pathlib import Path
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "processed"
OUTPUT_PATH = DATA_DIR / "manifests" / "manifest_all.csv"


def infer_usage(filename: str) -> str:
    name = filename.lower()
    if name.startswith("enroll_"):
        return "enroll"
    elif name.startswith("verify_clean_"):
        return "verify_clean"
    else:
        return "unknown"


def get_duration_and_sr(audio_path: Path) -> tuple[float, int]:
    info = sf.info(audio_path)
    duration_sec = info.frames / info.samplerate
    return round(duration_sec, 3), info.samplerate


def main() -> None:
    rows = []

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input folder: {INPUT_DIR}")

    speaker_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])

    if not speaker_dirs:
        raise FileNotFoundError(f"No speaker folders found in: {INPUT_DIR}")

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        wav_files = sorted(speaker_dir.glob("*.wav"))
        for wav_path in wav_files:
            usage = infer_usage(wav_path.name)
            duration_sec, sample_rate = get_duration_and_sr(wav_path)

            rows.append({
                "speaker_id": speaker_id,
                "audio_path": str(wav_path).replace("\\", "/"),
                "usage": usage,
                "split": "unassigned",
                "duration_sec": duration_sec,
                "sample_rate": sample_rate,
            })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Manifest created: {OUTPUT_PATH}")
    print(f"[OK] Total rows: {len(df)}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
