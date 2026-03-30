# Data Layout

## Raw Data
- Put downloaded or recorded audio in `data/raw/`.
- Do not commit raw audio to Git.

## Processed Data
- Write derived waveforms or features to `data/processed/`.
- Keep large generated files out of version control.

## Tracked Files
- Commit manifests, trial lists, and other lightweight metadata to `data/manifests/`.
- Store reusable external metadata in `data/external/` only when it is small and safe to version.

## Expected Workflow
1. Add or link raw audio locally under `data/raw/`.
2. Run preprocessing and augmentation jobs to create derived assets.
3. Generate manifests and trial files under `data/manifests/`.
4. Commit only the small metadata needed to reproduce experiments.
