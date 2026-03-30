"""Utilities for API file handling."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile


async def save_upload_to_temp(upload: UploadFile) -> Path:
    """Persist an uploaded file to a temporary path and return it."""
    suffix = Path(upload.filename or "").suffix or ".wav"
    fd, temp_path = tempfile.mkstemp(prefix="sv_", suffix=suffix)
    os.close(fd)
    target_path = Path(temp_path)

    with target_path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    await upload.close()
    return target_path


def cleanup_temp_files(paths: Iterable[Path]) -> None:
    """Best-effort cleanup for temporary files."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
