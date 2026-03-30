from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_SR = 16000


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95
    return y.astype(np.float32)


def process_file(input_path: Path, output_path: Path) -> None:
    y, sr = librosa.load(input_path, sr=None, mono=False)

    # Convert to mono if needed
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Resample to target sample rate
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # Normalize amplitude
    y = normalize_audio(y)

    # Make sure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as 16 kHz mono wav
    sf.write(output_path, y, TARGET_SR)
    print(f"[OK] {input_path} -> {output_path}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    wav_files = sorted(INPUT_DIR.rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under: {INPUT_DIR}")

    for input_path in wav_files:
        relative_path = input_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path
        process_file(input_path, output_path)

    print(f"\nDone. Processed {len(wav_files)} files into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
