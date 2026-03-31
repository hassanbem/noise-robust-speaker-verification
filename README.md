# Noise-Robust Speaker Verification

A 3-student machine learning and audio systems project for building a speaker verification pipeline that stays reliable under noisy conditions.

## Team Roles
- Student 1 : data pipeline, manifests, evaluation, ablations
- Student 2: models, inference, calibration, API
- Student 3: UI, demo assets, Gradio integration

## Current Status
- Repository scaffold and collaboration rules are in place.
- The API contract is defined in `docs/api_contract.md`.
- The UI can be mocked against the contract before the backend is complete.

## Repository Structure
- `src/data`, `src/eval`: preprocessing, augmentation, manifests, metrics
- `src/models`, `src/inference`, `src/api`: model loading, scoring, calibration, FastAPI
- `src/ui`: Gradio app and frontend logic
- `data`: raw, processed, manifests, and external metadata
- `artifacts`: released model and calibration assets
- `results`: report-ready scores, figures, ablations, and tables
- `docs`, `reports/sections`: shared technical notes and report drafts

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/dev.txt
pytest
```

## Demo
- Backend target: `POST /verify`
- UI should consume the response contract from `docs/api_contract.md`
- Keep only tiny demo audio in `assets/demo_samples/`

## Student 1 + Student 2 Integration Workflow

### 1. Ensure manifest paths are portable
- Manifests under `data/manifests/` should store relative paths such as:
  - `data/processed/spk001/verify_clean_01.wav`
- Avoid machine-specific absolute paths like `C:\...` or `/home/...`.

### 2. Batch score validation trials
```bash
python -m src.inference.score_trials \
  --trials data/manifests/trials_val.csv \
  --output results/scores/val_scores.csv
```

### 3. Calibrate threshold on validation scores
```bash
python -m src.inference.calibrate_threshold \
  --scores results/scores/val_scores.csv \
  --output artifacts/calibration/threshold.json
```

### 4. Batch score test trials
```bash
python -m src.inference.score_trials \
  --trials data/manifests/trials_test.csv \
  --output results/scores/test_scores.csv
```

### 5. Evaluate scored test set (+ noise ablations when columns exist)
```bash
python -m src.eval.evaluate_scores \
  --scores results/scores/test_scores.csv \
  --output-dir results/eval
```

Outputs produced by evaluation:
- `results/eval/*_summary.json`
- `results/eval/*_summary.csv`
- `results/eval/*_roc.png`
- `results/eval/*_det.png`
- `results/ablation/noise_ablation.csv` (if `noise_type` and/or `snr_db` columns exist)
- `results/tables/noise_type_summary.csv` and `results/tables/snr_db_summary.csv` (when available)

## Workflow
- Keep `main` clean and protected when possible.
- Use one feature branch per student after this scaffold commit.
- Do not commit raw datasets, experiment dumps, or temporary outputs.
