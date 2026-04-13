"""Default artifact paths and run IDs under ``artifacts/``.

Uses ``MODELOPT_ARTIFACTS_ROOT`` when set (non-empty); otherwise
``<process current working directory>/artifacts``. The root directory is
created automatically (``mkdir -p``). A per-process timestamp is used so
repeated runs with the same hyperparameters do not overwrite outputs.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


def artifacts_root() -> Path:
    """Absolute path to the artifacts root; parents are created if missing."""
    raw = os.environ.get("MODELOPT_ARTIFACTS_ROOT", "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    else:
        root = (Path.cwd() / "artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


SESSION_ID_ENV_VAR = "SESSION_ID"


def effective_session_id(cli_session_id: str | None) -> str | None:
    """Resolve pipeline session id: non-empty ``--session-id`` wins, else :envvar:`SESSION_ID`.

    Used by ``build-trt``, ``eval-trt``, ``trt-bench``, ``report-runs``, and ``pipeline-e2e``
    so a shell ``export SESSION_ID=…`` applies to every command unless overridden on the CLI.
    """
    cli = (cli_session_id or "").strip()
    if cli:
        return cli
    return (os.environ.get(SESSION_ID_ENV_VAR, "").strip() or None)


def run_timestamp() -> str:
    """Local time ``YYYYMMDD-HHMMSS`` (unique per second per machine)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_component(s: str, max_len: int = 56) -> str:
    """Filesystem-friendly single path segment."""
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    t = t.strip("_") or "unnamed"
    return t[:max_len]


def default_calib_npy_path(
    *,
    images_dir_name: str,
    img_size: int,
    n_images: int,
    ts: str | None = None,
) -> Path:
    ts = ts or run_timestamp()
    name = "_".join(
        [
            "calib",
            safe_component(images_dir_name),
            f"sz{img_size}",
            f"n{n_images}",
            ts,
        ]
    )
    return artifacts_root() / "calibration" / f"{name}.npy"


def default_calib_prep_log(*, images_dir_name: str, ts: str | None = None) -> Path:
    ts = ts or run_timestamp()
    d = artifacts_root() / "calibration" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    name = "_".join(["calib_prep", safe_component(images_dir_name), ts])
    return d / f"{name}.log"


def quantized_logs_dir() -> Path:
    return artifacts_root() / "quantized" / "logs"


def trt_engine_dir() -> Path:
    """Directory for TensorRT ``.engine`` files (default ``build-trt`` output)."""
    return artifacts_root() / "trt_engine"


def trt_engine_logs_dir() -> Path:
    return artifacts_root() / "trt_engine" / "logs"


def default_build_trt_session_log(
    *, onnx_stem: str, ts: str | None = None, session_id: str | None = None
) -> Path:
    """Log file for ``model-opt-yolo build-trt`` when ``--log-file`` is omitted.

    With *session_id*, logs go under ``pipeline_e2e/sessions/<id>/trt_engine/logs/`` so
    ``report-runs --session-id <id>`` picks them up alongside manual runs.
    """
    ts = ts or run_timestamp()
    log_dir = (
        pipeline_e2e_session_trt_logs(session_id)
        if (session_id or "").strip()
        else trt_engine_logs_dir()
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(["build_trt", safe_component(onnx_stem), ts])
    return log_dir / f"{name}.log"


def default_trt_bench_session_log(
    *, engine_stem: str, ts: str | None = None, session_id: str | None = None
) -> Path:
    """Log file for ``model-opt-yolo trt-bench`` when ``--log-file`` is omitted."""
    ts = ts or run_timestamp()
    log_dir = (
        pipeline_e2e_session_trt_logs(session_id)
        if (session_id or "").strip()
        else trt_engine_logs_dir()
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(["trt_bench", safe_component(engine_stem), ts])
    return log_dir / f"{name}.log"


def default_quantize_session_log(
    *,
    onnx_stem: str,
    quantize_mode: str,
    calibration_method: str,
    ts: str | None = None,
) -> Path:
    ts = ts or run_timestamp()
    quantized_logs_dir().mkdir(parents=True, exist_ok=True)
    name = "_".join(
        [
            "quantize",
            safe_component(onnx_stem),
            f"qt{safe_component(quantize_mode, 12)}",
            safe_component(calibration_method, 16),
            ts,
        ]
    )
    return quantized_logs_dir() / f"{name}.log"


def default_quantize_session_log_batch(
    *,
    n_files: int,
    quantize_mode: str,
    calibration_method: str,
    ts: str | None = None,
) -> Path:
    ts = ts or run_timestamp()
    quantized_logs_dir().mkdir(parents=True, exist_ok=True)
    name = "_".join(
        [
            "quantize",
            f"batch{n_files}",
            f"qt{safe_component(quantize_mode, 12)}",
            safe_component(calibration_method, 16),
            ts,
        ]
    )
    return quantized_logs_dir() / f"{name}.log"


def predictions_logs_dir() -> Path:
    return artifacts_root() / "predictions" / "logs"


def pipeline_e2e_logs_dir() -> Path:
    """Legacy flat log dir; prefer :func:`pipeline_e2e_session_root`."""
    return artifacts_root() / "pipeline_e2e" / "logs"


def pipeline_e2e_session_root(session_id: str) -> Path:
    """Root folder for one ``pipeline-e2e`` run — scoped logs and report (no mixing with other sessions)."""
    p = artifacts_root() / "pipeline_e2e" / "sessions" / safe_component(session_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_trt_logs(session_id: str) -> Path:
    """Session directory for ``trt_bench_*.log`` / ``build_trt_*.log`` (``report-runs --trt-logs-dir``)."""
    p = pipeline_e2e_session_root(session_id) / "trt_engine" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_eval_logs(session_id: str) -> Path:
    """Session directory for ``eval_*.log`` (``report-runs --eval-logs-dir``)."""
    p = pipeline_e2e_session_root(session_id) / "predictions" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_quant_logs(session_id: str) -> Path:
    """Session directory for ``quantize_*.log`` from pipeline-e2e."""
    p = pipeline_e2e_session_root(session_id) / "quantized" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_calibration_dir(session_id: str) -> Path:
    """Session directory for ``calib*.npy`` from ``pipeline-e2e`` (and optional ``calibration/logs``)."""
    p = pipeline_e2e_session_root(session_id) / "calibration"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_calib_npy_path(
    *,
    session_id: str,
    images_dir_name: str,
    img_size: int,
    n_images: int,
    ts: str | None = None,
) -> Path:
    """Calibration tensor path under ``…/sessions/<id>/calibration/`` (not global ``artifacts/calibration``)."""
    ts = ts or run_timestamp()
    name = "_".join(
        [
            "calib",
            safe_component(images_dir_name),
            f"sz{img_size}",
            f"n{n_images}",
            ts,
        ]
    )
    d = pipeline_e2e_session_calibration_dir(session_id)
    return d / f"{name}.npy"


def pipeline_e2e_session_calib_logs_dir(session_id: str) -> Path:
    p = pipeline_e2e_session_calibration_dir(session_id) / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_calib_prep_log(
    *, session_id: str, images_dir_name: str, ts: str | None = None
) -> Path:
    ts = ts or run_timestamp()
    d = pipeline_e2e_session_calib_logs_dir(session_id)
    name = "_".join(["calib_prep", safe_component(images_dir_name), ts])
    return d / f"{name}.log"


def pipeline_e2e_session_quantized_dir(session_id: str) -> Path:
    """Quantized ONNX output directory for ``pipeline-e2e`` (under the session root)."""
    p = pipeline_e2e_session_root(session_id) / "quantized"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pipeline_e2e_session_trt_engine_dir(session_id: str) -> Path:
    """Directory for ``*.engine`` (and ``*.timing.cache``) from ``pipeline-e2e``; logs live in ``…/trt_engine/logs/``."""
    p = pipeline_e2e_session_root(session_id) / "trt_engine"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_pipeline_e2e_session_log(*, session_id: str) -> Path:
    """Main orchestrator log: ``artifacts/pipeline_e2e/sessions/<session_id>/pipeline.log``."""
    return pipeline_e2e_session_root(session_id) / "pipeline.log"


def trex_runs_root(session_id: str | None = None) -> Path:
    """Directory root for ``trex-analyze`` outputs (engine JSON, graphs, compare CSV)."""
    if (session_id or "").strip():
        p = pipeline_e2e_session_root(session_id.strip()) / "trex"
    else:
        p = artifacts_root() / "trex" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_trex_analyze_run_dir(
    *,
    onnx_stem: str,
    mode_slug: str,
    ts: str | None = None,
    session_id: str | None = None,
) -> Path:
    """Per-run folder: ``artifacts/trex/runs/<stem>_<mode>_<ts>/`` or under pipeline session."""
    ts = ts or run_timestamp()
    name = "_".join([safe_component(onnx_stem), safe_component(mode_slug, 24), ts])
    d = trex_runs_root(session_id) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_eval_session_log(
    *, engine_stem: str, ts: str | None = None, session_id: str | None = None
) -> Path:
    ts = ts or run_timestamp()
    log_dir = (
        pipeline_e2e_session_eval_logs(session_id)
        if (session_id or "").strip()
        else predictions_logs_dir()
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(["eval", safe_component(engine_stem), ts])
    return log_dir / f"{name}.log"


