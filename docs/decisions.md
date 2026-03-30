# Technical Decisions

## 2026-03-30
- Use one shared repository with clear folder ownership.
- Keep raw and processed datasets out of Git.
- Split dependencies by workstream under `requirements/`.
- Treat `docs/api_contract.md` as the source of truth for UI and API integration.
- Use Git LFS only for tiny demo audio and final released model artifacts when necessary.
