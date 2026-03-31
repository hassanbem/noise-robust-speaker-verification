# API Contract

## Goal
This contract defines how the backend and UI communicate.

---

## Verification flow

### Inputs
The backend receives:
- enrollment audio
- test audio
- enhancement mode

For `POST /verify`, use `multipart/form-data` fields:
- `enroll_audio` (file, required)
- `test_audio` (file, required)
- `enhancement` (boolean, optional, default `false`)

### Output
The backend returns a JSON object in this format:

```json
{
  "score": 0.73,
  "threshold": 0.61,
  "decision": true,
  "decision_label": "same speaker",
  "latency_ms": 118,
  "model_name": "speechbrain/spkrec-ecapa-voxceleb",
  "sample_rate": 16000,
  "enhancement": false,
  "threshold_mode": "fixed",
  "message": "verification completed successfully"
}
```

---

## Fields meaning

- `score`: cosine similarity score between enrollment and test audio
- `threshold`: threshold used for decision
- `decision`: boolean result
- `decision_label`: human-readable decision
- `latency_ms`: total inference time in milliseconds
  - integer milliseconds in responses
- `model_name`: model identifier
- `sample_rate`: processed sample rate
- `enhancement`: whether enhancement was applied
- `threshold_mode`: fixed / eer / far_1
- `message`: backend status text

---

## Threshold loading policy
Backend threshold source order:

1. `artifacts/calibration/threshold.json` (calibrated threshold, preferred)
2. `artifacts/calibration/fixed_threshold.json` (legacy fixed fallback)
3. internal fallback (`threshold=0.50`, `threshold_mode="fixed"`) if files are missing

This keeps `/verify` stable for UI usage while enabling calibrated deployment.

---

## Phase 3 Endpoints

### `GET /health`

Response:

```json
{
  "status": "ok",
  "message": "API is running"
}
```

### `POST /verify`

Behavior:
- receives enrollment and test audio files
- loads threshold using the policy above
- computes similarity score and decision
- returns the contract JSON shown above

Note:
- `/enroll` is intentionally not implemented yet in Phase 3.
