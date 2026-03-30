"""I/O helpers for inference scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.inference.constants import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_FIXED_THRESHOLD,
    FIXED_THRESHOLD_MODE,
)


def ensure_file_exists(path: Path, label: str) -> None:
    """Fail fast if a required file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file into a dictionary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def load_threshold_settings(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
) -> tuple[float, str]:
    """Load threshold and threshold_mode from calibration JSON.

    Falls back to fixed defaults when file is missing.
    """
    if not calibration_path.is_file():
        return DEFAULT_FIXED_THRESHOLD, FIXED_THRESHOLD_MODE

    payload = read_json(calibration_path)
    raw_threshold = payload.get("threshold", DEFAULT_FIXED_THRESHOLD)
    raw_mode = payload.get("threshold_mode", FIXED_THRESHOLD_MODE)

    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid threshold in calibration file {calibration_path}: {raw_threshold!r}"
        ) from exc

    return threshold, str(raw_mode)
