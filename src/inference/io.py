"""I/O helpers for inference scripts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from src.inference.constants import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_FIXED_THRESHOLD,
    DEFAULT_FIXED_THRESHOLD_PATH,
    FIXED_THRESHOLD_MODE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRIAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "enroll_path": (
        "enroll_path",
        "enrollment_path",
        "enroll_audio_path",
        "enrollment_audio_path",
        "enroll",
    ),
    "test_path": (
        "test_path",
        "verification_path",
        "verify_path",
        "test_audio_path",
        "test",
    ),
    "label": ("label", "target", "is_target", "same_speaker", "is_same"),
    "enroll_speaker_id": ("enroll_speaker_id", "speaker_id_enroll", "enroll_spk"),
    "test_speaker_id": ("test_speaker_id", "speaker_id_test", "test_spk"),
    "noise_type": ("noise_type", "noise"),
    "snr_db": ("snr_db", "snr"),
    "language": ("language", "lang"),
}
REQUIRED_TRIAL_COLUMNS = ("enroll_path", "test_path")

_WINDOWS_ABS_RE = re.compile(r"^[a-zA-Z]:[\\/]")


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


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _find_column(
    fieldnames: list[str], canonical_name: str, *, required: bool
) -> str | None:
    normalized_to_original = {_normalize_header(name): name for name in fieldnames}
    aliases = TRIAL_COLUMN_ALIASES.get(canonical_name, (canonical_name,))

    for alias in aliases:
        original = normalized_to_original.get(_normalize_header(alias))
        if original is not None:
            return original

    if required:
        alias_text = ", ".join(aliases)
        raise ValueError(
            f"Missing required trials column '{canonical_name}'. "
            f"Accepted aliases: {alias_text}"
        )
    return None


def resolve_trials_column_mapping(fieldnames: list[str]) -> dict[str, str]:
    """Return canonical->source mapping for trials CSV columns."""
    mapping: dict[str, str] = {}
    for canonical in REQUIRED_TRIAL_COLUMNS:
        source = _find_column(fieldnames, canonical, required=True)
        if source is None:
            raise ValueError(f"Missing required trials column: {canonical}")
        mapping[canonical] = source

    for canonical in TRIAL_COLUMN_ALIASES:
        if canonical in mapping:
            continue
        source = _find_column(fieldnames, canonical, required=False)
        if source is not None:
            mapping[canonical] = source

    return mapping


def _canonicalize_trial_row(row: dict[str, str], mapping: dict[str, str]) -> dict[str, str]:
    canonical_row = dict(row)
    for canonical, source in mapping.items():
        if canonical not in canonical_row:
            canonical_row[canonical] = row.get(source, "")
    return canonical_row


def load_trials_csv(
    trials_path: Path,
) -> tuple[list[dict[str, str]], list[str], dict[str, str]]:
    """Load trials CSV and normalize required/known column names."""
    ensure_file_exists(trials_path, "Trials CSV")
    with trials_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Trials CSV has no header: {trials_path}")
        original_columns = list(reader.fieldnames)
        mapping = resolve_trials_column_mapping(original_columns)
        rows = [_canonicalize_trial_row(dict(row), mapping) for row in reader]

    return rows, original_columns, mapping


def parse_binary_label(raw_label: Any) -> int | None:
    """Parse a label value to {0,1}. Returns None when unknown."""
    if raw_label is None:
        return None

    if isinstance(raw_label, bool):
        return int(raw_label)

    value = str(raw_label).strip().lower()
    if value == "":
        return None

    positive = {
        "1",
        "true",
        "yes",
        "y",
        "same",
        "same_speaker",
        "target",
        "genuine",
        "bonafide",
    }
    negative = {
        "0",
        "false",
        "no",
        "n",
        "different",
        "different_speaker",
        "nontarget",
        "non_target",
        "impostor",
        "spoof",
    }
    if value in positive:
        return 1
    if value in negative:
        return 0

    try:
        numeric = float(value)
    except ValueError:
        return None
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def _looks_like_windows_absolute(raw_path: str) -> bool:
    return bool(_WINDOWS_ABS_RE.match(raw_path.strip()))


def _extract_repo_relative(raw_path: str) -> Path | None:
    normalized = raw_path.strip().replace("\\", "/")
    lowered = normalized.lower()

    for anchor in ("data/", "assets/", "artifacts/", "results/"):
        if lowered.startswith(anchor):
            return Path(normalized)
        marker = f"/{anchor}"
        idx = lowered.find(marker)
        if idx != -1:
            return Path(normalized[idx + 1 :])
    return None


def resolve_audio_path(
    raw_path: str,
    *,
    trials_csv_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve trial audio paths from relative/Linux/Windows path formats."""
    cleaned = (raw_path or "").strip()
    if not cleaned:
        raise FileNotFoundError("Empty audio path in trials file.")

    candidates: list[Path] = []
    raw_as_path = Path(cleaned).expanduser()

    if raw_as_path.is_absolute():
        candidates.append(raw_as_path)
    elif not _looks_like_windows_absolute(cleaned):
        if trials_csv_path is not None:
            candidates.append((trials_csv_path.parent / raw_as_path).expanduser())
        candidates.append((project_root / raw_as_path).expanduser())

    repo_relative = _extract_repo_relative(cleaned)
    if repo_relative is not None:
        candidates.append((project_root / repo_relative).expanduser())
        if trials_csv_path is not None:
            candidates.append((trials_csv_path.parent / repo_relative).expanduser())

    dedup: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            dedup.append(candidate)

    for candidate in dedup:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n".join(f"- {path}" for path in dedup[:5]) or "- (no candidates)"
    raise FileNotFoundError(
        "Audio file not found for path "
        f"'{raw_path}'. Tried:\n{tried}\n"
        "If Student 1 manifests were generated on another machine, keep paths under "
        "the project tree (e.g. data/...)."
    )


def _extract_threshold_from_payload(payload: dict[str, Any], source_path: Path) -> tuple[float, str]:
    raw_mode = payload.get("threshold_mode", FIXED_THRESHOLD_MODE)
    mode = str(raw_mode).strip() or FIXED_THRESHOLD_MODE

    if "selected_threshold" in payload:
        raw_threshold = payload["selected_threshold"]
    elif "threshold" in payload:
        raw_threshold = payload["threshold"]
    elif mode == "far_1" and "threshold_far_1" in payload:
        raw_threshold = payload["threshold_far_1"]
    elif mode == "eer" and "threshold_eer" in payload:
        raw_threshold = payload["threshold_eer"]
    elif "threshold_eer" in payload:
        raw_threshold = payload["threshold_eer"]
        mode = "eer"
    elif "threshold_far_1" in payload:
        raw_threshold = payload["threshold_far_1"]
        mode = "far_1"
    else:
        raise ValueError(
            f"Threshold JSON missing threshold values in {source_path}. "
            "Expected one of: selected_threshold, threshold, threshold_eer, threshold_far_1."
        )

    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid threshold value in {source_path}: {raw_threshold!r}"
        ) from exc

    return threshold, mode


def load_threshold_settings(
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    fixed_threshold_path: Path = DEFAULT_FIXED_THRESHOLD_PATH,
) -> tuple[float, str]:
    """Load threshold with priority: calibrated JSON -> fixed JSON -> defaults."""
    if calibration_path.is_file():
        payload = read_json(calibration_path)
        return _extract_threshold_from_payload(payload, calibration_path)

    if fixed_threshold_path.is_file():
        payload = read_json(fixed_threshold_path)
        return _extract_threshold_from_payload(payload, fixed_threshold_path)

    return DEFAULT_FIXED_THRESHOLD, FIXED_THRESHOLD_MODE
