#!/usr/bin/env python3
"""Launch `python -m modelopt.onnx.quantization.autotune` when the CLI exists.

PyPI nvidia-modelopt 0.42.x ships only partial `autotune` helpers (no __main__.py).
The full Q/DQ autotuner is in upstream Model Optimizer — install from Git to use it.

Wrapper-only arguments (not passed to modelopt):

  --imagesize N | HxW   TensorRT profile for dynamic ONNX inputs (e.g. 640 or 640x640).
                        Required when the model input has symbolic/unknown spatial dims.
  --input_name NAME     Input tensor name for --minShapes/--optShapes/--maxShapes (default: images).
  --log-file PATH, -v   See model_opt_yolo.logutil (stripped before modelopt sees argv).

Unless ``--output_dir`` / ``-o`` is set, defaults to
``artifacts/autotune/autotune_<stem>_qt<...>_spr<...>_img<...>_<timestamp>/``.
Default wrapper log: ``wrapper.log`` in that directory (or ``autotune_session_<ts>.log`` if you set output_dir only).
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

from model_opt_yolo.logutil import pop_logging_flags, setup_logging
from model_opt_yolo.session_paths import (
    append_output_dir,
    autotune_run_dir,
    build_autotune_run_id,
    default_autotune_wrapper_log,
    get_output_dir_value,
    has_output_dir,
    argv_get,
    safe_component,
    run_timestamp,
)


def _autotune_cli_available() -> bool:
    return importlib.util.find_spec("modelopt.onnx.quantization.autotune.__main__") is not None


def _parse_imagesize(spec: str) -> tuple[int, int]:
    spec = spec.strip().lower().replace("×", "x")
    if "x" in spec:
        parts = spec.split("x", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid --imagesize {spec!r}, expected HxW or N (square)")
        return int(parts[0].strip()), int(parts[1].strip())
    n = int(spec)
    return n, n


def _strip_wrapper_argv(argv: list[str]) -> tuple[list[str], str | None, str]:
    """Remove --imagesize and --input_name; return (rest, imagesize, input_name)."""
    out: list[str] = []
    imagesize: str | None = None
    input_name = "images"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--imagesize" and i + 1 < len(argv):
            imagesize = argv[i + 1]
            i += 2
            continue
        if a.startswith("--imagesize="):
            imagesize = a.split("=", 1)[1]
            i += 1
            continue
        if a == "--input_name" and i + 1 < len(argv):
            input_name = argv[i + 1]
            i += 2
            continue
        if a.startswith("--input_name="):
            input_name = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return out, imagesize, input_name


def _extract_onnx_path(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a in ("--onnx_path", "-m") and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _graph_input_names_excluding_initializers(model) -> list[str]:
    init = {x.name for x in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in init]


def _input_tensor_is_dynamic(inp) -> bool:
    shape = inp.type.tensor_type.shape
    for d in shape.dim:
        if d.HasField("dim_param"):
            return True
        if not d.HasField("dim_value"):
            return True
    return False


def _needs_imagesize(onnx_path: str, input_name: str) -> bool:
    import onnx

    path = Path(onnx_path)
    if not path.is_file():
        return False
    model = onnx.load(str(path))
    names = _graph_input_names_excluding_initializers(model)
    if not names:
        return False
    targets = [i for i in model.graph.input if i.name in names and i.name == input_name]
    if not targets:
        targets = [i for i in model.graph.input if i.name in names]
    if not targets:
        return False
    return _input_tensor_is_dynamic(targets[0])


def _merge_trtexec_shapes(
    argv: list[str],
    input_name: str,
    h: int,
    w: int,
) -> list[str]:
    """Ensure --use_trtexec and --trtexec_benchmark_args include min/opt/maxShapes."""
    shape = f"1x3x{h}x{w}"
    shape_args = (
        f"--minShapes={input_name}:{shape} "
        f"--optShapes={input_name}:{shape} "
        f"--maxShapes={input_name}:{shape}"
    ).strip()

    out = list(argv)
    trt_idx: int | None = None
    for i, a in enumerate(out):
        if a == "--trtexec_benchmark_args" and i + 1 < len(out):
            trt_idx = i
            break
        if a.startswith("--trtexec_benchmark_args="):
            trt_idx = i
            break

    if trt_idx is None:
        out.extend(["--use_trtexec", "--trtexec_benchmark_args", f"{shape_args}"])
        return out

    if out[trt_idx].startswith("--trtexec_benchmark_args="):
        prefix, val = out[trt_idx].split("=", 1)
        merged = f"{shape_args} {val}".strip()
        out[trt_idx] = f"{prefix}={merged}"
    else:
        old = out[trt_idx + 1]
        merged = f"{shape_args} {old}".strip()
        out[trt_idx + 1] = merged

    if "--use_trtexec" not in out:
        out.insert(0, "--use_trtexec")

    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    argv, log_explicit, verbose = pop_logging_flags(argv)
    argv_rest, imagesize, input_name = _strip_wrapper_argv(argv)

    if "--help" in argv_rest or "-h" in argv_rest:
        setup_logging("autotune_onnx", log_file=log_explicit, verbose=verbose)
        cmd = [sys.executable, "-m", "modelopt.onnx.quantization.autotune", *argv_rest]
        return subprocess.call(cmd)

    if not _autotune_cli_available():
        setup_logging("autotune_onnx", log_file=log_explicit, verbose=verbose)
        log = logging.getLogger("autotune_onnx")
        log.error(
            "ONNX Q/DQ autotuner CLI is missing. Install Model Optimizer from GitHub (includes autotune): "
            'pip install -U "nvidia-modelopt[onnx] @ git+https://github.com/NVIDIA/Model-Optimizer.git@main"'
        )
        return 1

    onnx_path = _extract_onnx_path(argv_rest)
    model_stem = safe_component(Path(onnx_path).stem) if onnx_path else "model"

    img_h: int | None = None
    img_w: int | None = None
    if imagesize:
        try:
            img_h, img_w = _parse_imagesize(imagesize)
        except ValueError as e:
            setup_logging("autotune_onnx", log_file=log_explicit, verbose=verbose)
            logging.getLogger("autotune_onnx").error("Invalid --imagesize: %s", e)
            return 2

    if onnx_path and _needs_imagesize(onnx_path, input_name):
        if not imagesize:
            setup_logging("autotune_onnx", log_file=log_explicit, verbose=verbose)
            logging.getLogger("autotune_onnx").error(
                "This ONNX has a dynamic input tensor; pass --imagesize 640 or 640x640 "
                "(or --input_name if not `images`)."
            )
            return 2

    if imagesize and img_h is not None and img_w is not None:
        argv_rest = _merge_trtexec_shapes(argv_rest, input_name, img_h, img_w)

    ts = run_timestamp()
    quant = argv_get(argv_rest, "--quant_type") or "int8"
    schemes = argv_get(argv_rest, "--schemes_per_region") or argv_get(argv_rest, "-s")
    mode = argv_get(argv_rest, "--mode")
    run_id = build_autotune_run_id(
        model_stem,
        quant_type=quant,
        schemes_per_region=schemes,
        mode=mode,
        img_h=img_h,
        img_w=img_w,
        ts=ts,
    )
    run_dir = autotune_run_dir(run_id)
    out_val = get_output_dir_value(argv_rest)

    if not has_output_dir(argv_rest):
        argv_rest = append_output_dir(argv_rest, run_dir)

    if log_explicit is not None:
        log_path: str | None = log_explicit
    elif out_val is None:
        log_path = str(default_autotune_wrapper_log(run_dir))
    else:
        log_path = str(Path(out_val).resolve() / f"autotune_session_{ts}.log")

    setup_logging("autotune_onnx", log_file=log_path, verbose=verbose)
    log = logging.getLogger("autotune_onnx")

    if imagesize and img_h is not None and img_w is not None:
        log.info("TensorRT profile: input=%s shape 1x3x%dx%d (trtexec min/opt/max)", input_name, img_h, img_w)
    if out_val is None:
        log.info("Default output directory: %s", run_dir)

    cmd = [sys.executable, "-m", "modelopt.onnx.quantization.autotune", *argv_rest]
    log.info("Launching: %s", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
