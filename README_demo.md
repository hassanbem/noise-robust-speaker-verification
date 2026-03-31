# Demo Guide (Student 3)

Tell Student 3 this exactly:

Start the backend first, then run the UI against the real `/verify` API.
Use the contract in `docs/api_contract.md`.

## Scope (UI only)
- UI files only: `src/ui/gradio_app.py`, `src/ui/api_client.py`
- No backend edits
- No model/inference/evaluation edits

## Run locally
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/ui.txt
python -m pip install -r requirements/model.txt
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# in a new terminal
.\.venv\Scripts\Activate.ps1
python -m src.ui.gradio_app
```

## Demo flow
1. Enroll a speaker in **Enroll** tab.
2. Go to **Verify** tab and choose the enrolled speaker.
3. Upload/record test audio and click **Verify**.
4. Present score, threshold, decision, latency, and JSON contract output.

## Notes
- Verify API URL defaults to `http://127.0.0.1:8000/verify`.
- You can override it from the UI textbox if backend runs on another host/port.
