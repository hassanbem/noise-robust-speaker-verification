from __future__ import annotations

from pathlib import Path
import wave

from src.ui.mock_backend import fake_verify


REQUIRED_KEYS = {
    "score",
    "threshold",
    "decision",
    "decision_label",
    "latency_ms",
    "model_name",
    "sample_rate",
    "enhancement",
    "threshold_mode",
    "message",
}


def _make_silent_wav(path: Path, sample_rate: int = 16000, seconds: float = 0.1) -> None:
    nframes = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * nframes)


def test_fake_verify_returns_contract_keys(tmp_path: Path) -> None:
    enroll = tmp_path / "enroll.wav"
    test = tmp_path / "test.wav"
    _make_silent_wav(enroll)
    _make_silent_wav(test)

    response = fake_verify(str(enroll), str(test), threshold_mode="fixed", enhancement=False)

    assert REQUIRED_KEYS.issubset(response.keys())
    assert isinstance(response["score"], float)
    assert isinstance(response["threshold"], float)
    assert isinstance(response["decision"], bool)
    assert isinstance(response["latency_ms"], int)
    assert response["sample_rate"] == 16000


def test_fake_verify_handles_invalid_paths() -> None:
    response = fake_verify("missing_enroll.wav", "missing_test.wav")

    assert response["decision"] is False
    assert response["message"] == "invalid audio path(s)"
