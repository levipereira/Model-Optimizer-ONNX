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


def default_build_trt_session_log(*, onnx_stem: str, ts: str | None = None) -> Path:
    """Log file for ``model-opt-yolo build-trt`` when ``--log-file`` is omitted."""
    ts = ts or run_timestamp()
    trt_engine_logs_dir().mkdir(parents=True, exist_ok=True)
    name = "_".join(["build_trt", safe_component(onnx_stem), ts])
    return trt_engine_logs_dir() / f"{name}.log"


def default_trt_bench_session_log(*, engine_stem: str, ts: str | None = None) -> Path:
    """Log file for ``model-opt-yolo trt-bench`` when ``--log-file`` is omitted."""
    ts = ts or run_timestamp()
    trt_engine_logs_dir().mkdir(parents=True, exist_ok=True)
    name = "_".join(["trt_bench", safe_component(engine_stem), ts])
    return trt_engine_logs_dir() / f"{name}.log"


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


def default_pipeline_e2e_session_log(*, session_id: str) -> Path:
    """Main orchestrator log: ``artifacts/pipeline_e2e/sessions/<session_id>/pipeline.log``."""
    return pipeline_e2e_session_root(session_id) / "pipeline.log"


def default_eval_session_log(*, engine_stem: str, ts: str | None = None) -> Path:
    ts = ts or run_timestamp()
    predictions_logs_dir().mkdir(parents=True, exist_ok=True)
    name = "_".join(["eval", safe_component(engine_stem), ts])
    return predictions_logs_dir() / f"{name}.log"


