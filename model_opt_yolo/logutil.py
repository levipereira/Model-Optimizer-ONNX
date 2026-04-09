"""Shared logging setup for Model Optimizer YOLO tools.

Environment variables (optional):

- ``MODELOPT_YOLO_LOGLEVEL`` or ``LOGLEVEL``: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_ENV_KEYS = ("MODELOPT_YOLO_LOGLEVEL", "LOGLEVEL")


def _parse_level(name: str | None) -> int:
    if not name:
        return logging.INFO
    name = name.strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(
    name: str,
    *,
    log_file: str | Path | None = None,
    level: int | None = None,
    verbose: bool = False,
) -> logging.Logger:
    """Attach stderr and optional file handlers to logger *name*."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    if verbose:
        console_level = logging.DEBUG
    elif level is not None:
        console_level = level if isinstance(level, int) else _parse_level(str(level))
    else:
        console_level = logging.INFO
        for key in _ENV_KEYS:
            raw = os.environ.get(key)
            if raw:
                console_level = _parse_level(raw)
                break

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(console_level)
    sh.setFormatter(fmt)

    logger.setLevel(logging.DEBUG if log_file else console_level)
    logger.addHandler(sh)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def pop_logging_flags(argv: list[str]) -> tuple[list[str], str | None, bool]:
    """Remove ``-v`` / ``--verbose`` and ``--log-file`` from *argv*."""
    out: list[str] = []
    log_file: str | None = None
    verbose = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-v", "--verbose"):
            verbose = True
            i += 1
            continue
        if a == "--log-file" and i + 1 < len(argv):
            log_file = argv[i + 1]
            i += 2
            continue
        if a.startswith("--log-file="):
            log_file = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return out, log_file, verbose


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Write session log to this file (UTF-8). If omitted, a unique path under "
        "artifacts/.../logs/ is used. Console: INFO unless -v.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log DEBUG on stderr (and file if --log-file).",
    )
