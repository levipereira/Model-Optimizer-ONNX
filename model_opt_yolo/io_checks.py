"""Validate input paths before subprocess or heavy I/O (exist, type, readable)."""

from __future__ import annotations

import os
from pathlib import Path


def validate_readable_file(path: str | Path, *, label: str) -> str | None:
    """Return an error message if *path* is not a readable regular file, else ``None``."""
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError as exc:
        return f"{label}: invalid path {path!r}: {exc}"
    if not p.exists():
        return f"{label} not found: {path}"
    if not p.is_file():
        return f"{label} is not a file: {path}"
    try:
        with p.open("rb") as f:
            f.read(1)
    except OSError as exc:
        return f"{label}: cannot read {path}: {exc}"
    return None


def validate_existing_dir(path: str | Path, *, label: str) -> str | None:
    """Return an error message if *path* is not an existing readable directory, else ``None``."""
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError as exc:
        return f"{label}: invalid path {path!r}: {exc}"
    if not p.exists():
        return f"{label} not found: {path}"
    if not p.is_dir():
        return f"{label} is not a directory: {path}"
    if not os.access(p, os.R_OK):
        return f"{label}: directory not accessible: {path}"
    return None


_NPY_MAGIC = b"\x93NUMPY"


def validate_numpy_array_file(path: str | Path, *, label: str) -> str | None:
    """Like ``validate_readable_file`` plus NumPy ``.npy`` header magic (avoids loading large tensors)."""
    err = validate_readable_file(path, label=label)
    if err:
        return err
    p = Path(path).expanduser().resolve()
    try:
        with p.open("rb") as f:
            head = f.read(len(_NPY_MAGIC))
        if head != _NPY_MAGIC:
            return f"{label}: not a NumPy .npy file (invalid header): {path}"
    except OSError as exc:
        return f"{label}: cannot read {path}: {exc}"
    return None
