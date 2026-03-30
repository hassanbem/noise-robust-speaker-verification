# API Contract

## POST /verify

### Request
- `enroll_audio_path` or uploaded audio
- `test_audio_path` or uploaded audio
- `threshold_mode`: `"eer"` | `"far_1"`
- `enhancement`: `true` | `false`

### Response
```json
{
  "score": 0.73,
  "threshold": 0.61,
  "decision": true,
  "latency_ms": 118,
  "noise_type": "street",
  "snr_db": 5,
  "enhancement": false,
  "model_name": "speechbrain-ecapa"
}
```

### Notes
- Student 2 owns the backend implementation of this contract.
- Student 3 can build the UI against this response shape before the real model is ready.
- Local file paths are acceptable for development; uploaded files can replace them later.
