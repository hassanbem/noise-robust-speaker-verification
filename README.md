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

## Workflow
- Keep `main` clean and protected when possible.
- Use one feature branch per student after this scaffold commit.
- Do not commit raw datasets, experiment dumps, or temporary outputs.
