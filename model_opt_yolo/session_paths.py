"""Default artifact paths and run IDs under ``artifacts/``.

Uses ``MODELOPT_ARTIFACTS_ROOT`` when set (non-empty); otherwise
``<process current working directory>/artifacts``. The root directory is
created automatically (``mkdir -p``). A per-process timestamp is used so
repeated runs with the same hyperparameters do not overwrite outputs.

Run ID pattern (autotune example)::

    autotune_{model_stem}_qt{int8}_spr{30}_m_quick_img640x640_{20260409-183045}
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

# Default matches upstream autotune when --schemes_per_region is omitted
AUTOTUNE_DEFAULT_SCHEMES_NOTE = "DEF"


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


def argv_get(argv: list[str], long_flag: str, short_flag: str | None = None) -> str | None:
    """Return value for ``--long value``, ``--long=value``, or ``-s value`` style."""
    lf_eq = long_flag + "="
    for i, a in enumerate(argv):
        if a.startswith(lf_eq):
            return a.split("=", 1)[1]
        if a == long_flag and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            return argv[i + 1]
        if short_flag and a == short_flag and i + 1 < len(argv):
            v = argv[i + 1]
            if not v.startswith("-"):
                return v
    return None


def imagesize_label(h: int | None, w: int | None) -> str:
    if h is None or w is None:
        return "dyn"
    if h == w:
        return str(h)
    return f"{h}x{w}"


def build_autotune_run_id(
    model_stem: str,
    *,
    quant_type: str,
    schemes_per_region: str | None,
    mode: str | None,
    img_h: int | None,
    img_w: int | None,
    ts: str | None = None,
) -> str:
    """Single directory name segment (no slashes)."""
    ts = ts or run_timestamp()
    parts = [
        "autotune",
        safe_component(model_stem),
        f"qt{safe_component(quant_type, 12)}",
    ]
    if schemes_per_region is not None:
        parts.append(f"spr{safe_component(schemes_per_region, 8)}")
    else:
        parts.append(f"spr{AUTOTUNE_DEFAULT_SCHEMES_NOTE}")
    if mode and mode != "default":
        parts.append(f"m_{safe_component(mode, 24)}")
    parts.append(f"img{imagesize_label(img_h, img_w)}")
    parts.append(ts)
    return "_".join(parts)


def autotune_run_dir(run_id: str) -> Path:
    return artifacts_root() / "autotune" / run_id


def default_autotune_wrapper_log(run_dir: Path) -> Path:
    return run_dir / "wrapper.log"


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


def default_eval_session_log(*, engine_stem: str, ts: str | None = None) -> Path:
    ts = ts or run_timestamp()
    predictions_logs_dir().mkdir(parents=True, exist_ok=True)
    name = "_".join(["eval", safe_component(engine_stem), ts])
    return predictions_logs_dir() / f"{name}.log"


def get_output_dir_value(argv: list[str]) -> str | None:
    v = argv_get(argv, "--output_dir")
    if v is not None:
        return v
    return argv_get(argv, "-o")


def has_output_dir(argv: list[str]) -> bool:
    for i, a in enumerate(argv):
        if a == "--output_dir" or a.startswith("--output_dir="):
            return True
        if a == "-o":
            return True
    return False


def append_output_dir(argv: list[str], out_dir: Path) -> list[str]:
    """Append ``--output_dir`` if not already present."""
    if has_output_dir(argv):
        return argv
    return [*argv, "--output_dir", str(out_dir)]
