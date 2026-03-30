# Architecture

## High-Level Flow
1. Raw recordings are stored locally in `data/raw/`.
2. Preprocessing and augmentation code in `src/data/` creates derived files in `data/processed/`.
3. Tracked manifests and trial files live in `data/manifests/`.
4. Model code in `src/models/` produces embeddings, while `src/inference/` handles scoring and thresholding.
5. The FastAPI service in `src/api/` exposes `/verify`.
6. The Gradio UI in `src/ui/` calls the API and renders the response.
7. Final artifacts and report-ready outputs are stored under `artifacts/` and `results/`.

## Ownership Boundaries
- Student 1: `src/data/`, `src/eval/`, `data/manifests/`, `results/ablation/`, `results/tables/`
- Student 2: `src/models/`, `src/inference/`, `src/api/`, `configs/model/`, `configs/api/`, `artifacts/calibration/`
- Student 3: `src/ui/`, `assets/`, `configs/ui/`

## Collaboration Rules
- Shared interface changes start in `docs/api_contract.md`.
- Shared project decisions are logged in `docs/decisions.md`.
- Released binaries belong in `artifacts/`; large experimental files stay out of Git.
