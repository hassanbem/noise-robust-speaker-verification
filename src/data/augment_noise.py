from pathlib import Path
import random
import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VAL_MANIFEST = DATA_DIR / "manifests" / "manifest_val.csv"
TEST_MANIFEST = DATA_DIR / "manifests" / "manifest_test.csv"
NOISE_DIR = DATA_DIR / "noise_sources"
OUTPUT_ROOT = DATA_DIR / "augmented" / "noise"

SNR_LEVELS = [0, 5, 10]


def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(x)) + 1e-10)


def match_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    if len(noise) >= target_len:
        start = 0 if len(noise) == target_len else random.randint(0, len(noise) - target_len)
        return noise[start:start + target_len]
    repeats = int(np.ceil(target_len / len(noise)))
    tiled = np.tile(noise, repeats)
    return tiled[:target_len]


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)

    if noise_rms < 1e-10:
        return clean.copy()

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise = noise * (desired_noise_rms / noise_rms)

    mixed = clean + noise

    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.95

    return mixed.astype(np.float32)


def load_mono(path: Path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), sr


def process_manifest(manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    verify_df = df[df["usage"] == "verify_clean"].copy()

    noise_files = sorted(NOISE_DIR.glob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"No noise WAV files found in {NOISE_DIR}")

    for _, row in verify_df.iterrows():
        audio_path = Path(row["audio_path"])
        speaker_id = row["speaker_id"]

        clean, sr = load_mono(audio_path)

        for snr_db in SNR_LEVELS:
            noise_path = random.choice(noise_files)
            noise, noise_sr = load_mono(noise_path)

            if noise_sr != sr:
                raise ValueError(f"Sample rate mismatch: {noise_path} has {noise_sr}, expected {sr}")

            noise = match_length(noise, len(clean))
            mixed = mix_with_snr(clean, noise, snr_db)

            out_dir = OUTPUT_ROOT / f"snr_{snr_db}" / speaker_id
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / audio_path.name
            sf.write(out_path, mixed, sr)
            print(f"[OK] {audio_path.name} -> {out_path}")


def main():
    if not NOISE_DIR.exists():
        raise FileNotFoundError(f"Missing noise folder: {NOISE_DIR}")

    process_manifest(VAL_MANIFEST)
    process_manifest(TEST_MANIFEST)
    print("\nDone creating noisy verification audio.")


if __name__ == "__main__":
    main()
