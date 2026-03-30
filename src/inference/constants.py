"""Shared constants for inference and scoring scripts."""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_CONFIG_PATH = Path("configs/model/ecapa.yaml")
DEFAULT_CALIBRATION_PATH = Path("artifacts/calibration/fixed_threshold.json")

DEFAULT_FIXED_THRESHOLD = 0.50
FIXED_THRESHOLD_MODE = "fixed"

SCORE_DECIMALS = 6
