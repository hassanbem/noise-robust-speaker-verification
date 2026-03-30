"""SpeechBrain ECAPA-TDNN wrapper for speaker verification."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml


@dataclass(frozen=True)
class VerifierConfig:
    """Runtime configuration for speaker verification."""

    model_name: str
    sample_rate: int
    device: str
    score_method: str
    normalize_audio: bool


def load_verifier_config(config_path: str | Path) -> VerifierConfig:
    """Load verifier settings from YAML."""
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return VerifierConfig(
        model_name=data["model_name"],
        sample_rate=int(data["sample_rate"]),
        device=str(data["device"]),
        score_method=str(data["score_method"]).lower(),
        normalize_audio=bool(data["normalize_audio"]),
    )


def _ensure_speechbrain_torchaudio_compat() -> None:
    """Patch torchaudio API expected by current SpeechBrain releases."""
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]


def _resolve_speechbrain_savedir() -> Path:
    """Resolve model cache dir outside the repo to avoid uvicorn reload loops."""
    override = os.getenv("SPEECHBRAIN_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path(gettempdir()) / "noise_robust_sv" / "speechbrain_ecapa").resolve()


class SpeechBrainVerifier:
    """Thin wrapper around SpeechBrain ECAPA embeddings + cosine scoring."""

    def __init__(self, config_path: str | Path = "configs/model/ecapa.yaml") -> None:
        self.config = load_verifier_config(config_path)
        if self.config.score_method != "cosine":
            raise ValueError(
                f"Unsupported score method: {self.config.score_method}. "
                "Only 'cosine' is implemented."
            )

        _ensure_speechbrain_torchaudio_compat()
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = torch.device(self.config.device)
        savedir = _resolve_speechbrain_savedir()
        savedir.mkdir(parents=True, exist_ok=True)
        self.model = EncoderClassifier.from_hparams(
            source=self.config.model_name,
            savedir=str(savedir),
            run_opts={"device": str(self.device)},
        )

    def read_audio(self, audio_path: str | Path) -> torch.Tensor:
        """Load waveform, convert to mono, resample, and optionally normalize."""
        waveform_np, sr = sf.read(str(audio_path), always_2d=True)
        waveform = torch.from_numpy(waveform_np.T).to(torch.float32)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.config.sample_rate
            )
            waveform = resampler(waveform)

        if self.config.normalize_audio:
            peak = torch.max(torch.abs(waveform))
            if peak > 0:
                waveform = waveform / peak

        return waveform

    def embed(self, audio_path: str | Path) -> torch.Tensor:
        """Return a 1D embedding tensor for one audio file."""
        waveform = self.read_audio(audio_path).to(self.device)
        with torch.inference_mode():
            embedding = self.model.encode_batch(waveform)

        return embedding.squeeze().detach().cpu().flatten()

    @staticmethod
    def cosine_score(enroll_embedding: torch.Tensor, test_embedding: torch.Tensor) -> float:
        """Compute cosine similarity score between two embeddings."""
        enroll = F.normalize(enroll_embedding.view(1, -1), dim=1)
        test = F.normalize(test_embedding.view(1, -1), dim=1)
        return float(torch.sum(enroll * test, dim=1).item())

    def score_embeddings(
        self, enroll_embedding: torch.Tensor, test_embedding: torch.Tensor
    ) -> float:
        """Score a pair of embeddings with the configured scoring method."""
        if self.config.score_method == "cosine":
            return self.cosine_score(enroll_embedding, test_embedding)
        raise ValueError(f"Unsupported score method: {self.config.score_method}")

    def score(self, enroll_audio_path: str | Path, test_audio_path: str | Path) -> float:
        """End-to-end score from two audio files."""
        enroll_embedding = self.embed(enroll_audio_path)
        test_embedding = self.embed(test_audio_path)
        return self.score_embeddings(enroll_embedding, test_embedding)
