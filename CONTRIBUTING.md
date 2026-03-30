# Contributing Rules

1. Never push directly to `main`.
2. One student owns one folder set.
3. Do not edit another student's folder without opening an issue and telling them.
4. Pull `main`, then merge `main` into your branch before starting new work.
5. Keep pull requests small and focused.
6. Do not commit raw datasets.
7. Do not commit big checkpoints unless they are final and tracked with Git LFS.
8. Each student uses only their own notebook folder.
9. Shared files (`README.md`, `.github/`, `requirements/`, `pyproject.toml`, `Makefile`, `docs/decisions.md`) are edited only by the maintainer.
10. API changes must first be written in `docs/api_contract.md`.

## Ownership
- Student 1 owns `src/data/`, `src/eval/`, `data/manifests/`, `results/ablation/`, `results/tables/`, and `reports/sections/02_data.md`.
- Student 2 owns `src/models/`, `src/inference/`, `src/api/`, `configs/model/`, `configs/api/`, `artifacts/calibration/`, and `reports/sections/03_model.md`.
- Student 3 owns `src/ui/`, `assets/`, `configs/ui/`, and `reports/sections/04_ui.md`.
- `@hassanbem` is the current maintainer for shared files until the team agrees otherwise.

## Daily Workflow
```powershell
git switch main
git pull origin main

git switch <your-branch>
git merge main
```

Then commit focused changes:

```powershell
git add .
git commit -m "feat(data): add manifest builder"
git push
```

## Practical Guardrails
- No shared notebook editing.
- No committing `mlruns/`, `wandb/`, or random binary dumps.
- No full VoxCeleb, MUSAN, or other large datasets in Git.
- Use Git LFS only for tiny demo audio and final released artifacts when needed.
