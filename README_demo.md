# Demo Guide (Student 3)

Tell Student 3 this exactly:

Start the UI now using the API contract in docs/api_contract.md.
Do not wait for the real backend.
Build the Gradio interface using a fake backend function that returns the same JSON structure.

## Scope (UI only)
- UI files only: `src/ui/gradio_app.py`, `src/ui/mock_backend.py`
- No backend edits
- No model/inference/evaluation edits

## Run locally
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/ui.txt
python -m src.ui.gradio_app
```

## Demo flow
1. Enroll a speaker in **Enroll** tab.
2. Go to **Verify** tab and choose the enrolled speaker.
3. Upload/record test audio and click **Verify**.
4. Present score, threshold, decision, latency, and JSON contract output.

## Notes
- This app is intentionally wired to a fake backend contract first.
- Student 3 can swap the fake function with real API calls once backend stabilizes.
