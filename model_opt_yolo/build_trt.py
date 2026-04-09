#!/usr/bin/env python3
"""Run ``trtexec`` to build a TensorRT engine from ONNX (quantized or FP)."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from model_opt_yolo.logutil import add_logging_arguments, setup_logging
from model_opt_yolo.session_paths import default_build_trt_session_log, run_timestamp, trt_engine_dir


def _shape(batch: int, img_size: int) -> str:
    return f"{batch}x3x{img_size}x{img_size}"


def build_trtexec_argv(
    *,
    onnx_path: Path,
    engine_path: Path,
    timing_cache: Path,
    input_name: str,
    img_size: int,
    batch: int,
    mode: str,
    warm_up: int,
    duration: int,
    extra: list[str],
) -> list[str]:
    """Base ``trtexec`` argument list; *extra* is appended last for overrides."""
    onnx_s = str(onnx_path.resolve())
    eng_s = str(engine_path.resolve())
    cache_s = str(timing_cache.resolve())

    if mode == "strongly-typed":
        shp = _shape(batch, img_size)
        base = [
            "--onnx=" + onnx_s,
            "--saveEngine=" + eng_s,
            "--stronglyTyped",
            f"--minShapes={input_name}:{shp}",
            f"--optShapes={input_name}:{shp}",
            f"--maxShapes={input_name}:{shp}",
            "--timingCacheFile=" + cache_s,
        ]
    elif mode == "benchmark":
        min_shp = _shape(1, img_size)
        bm_shp = _shape(batch, img_size)
        base = [
            "--onnx=" + onnx_s,
            "--fp16",
            "--int8",
            "--saveEngine=" + eng_s,
            "--timingCacheFile=" + cache_s,
            f"--warmUp={warm_up}",
            f"--duration={duration}",
            "--useCudaGraph",
            "--useSpinWait",
            "--noDataTransfers",
            f"--minShapes={input_name}:{min_shp}",
            f"--optShapes={input_name}:{bm_shp}",
            f"--maxShapes={input_name}:{bm_shp}",
        ]
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return ["trtexec", *base, *extra]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TensorRT .engine from ONNX via trtexec. "
            "Default mode uses --stronglyTyped for Q/DQ models from Model Optimizer; "
            "benchmark mode adds --fp16/--int8 and latency flags for throughput measurement."
        )
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        metavar="PATH",
        help="Input .onnx file (e.g. quantized export).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        metavar="N",
        help="Square spatial size H=W for dynamic shape profile (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="B",
        help="Batch dimension for opt/max shapes (strongly-typed: min/opt/max all use B). "
        "For benchmark mode, min batch is fixed to 1.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="images",
        metavar="NAME",
        help="ONNX input tensor name used in --minShapes/--optShapes/--maxShapes (default: images).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("strongly-typed", "benchmark"),
        default="strongly-typed",
        help=(
            "strongly-typed: --stronglyTyped + same min/opt/max shape (default, matches Q/DQ ONNX). "
            "benchmark: --fp16 --int8 + warmup/duration/spin/CUDA graph + min batch 1."
        ),
    )
    parser.add_argument(
        "--warm-up",
        type=int,
        default=500,
        metavar="N",
        help="[benchmark] trtexec --warmUp (default: 500).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        metavar="SEC",
        help="[benchmark] trtexec --duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--engine-out",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output .engine path (default: <artifacts>/trt_engine/<onnx-stem>.engine; "
            "artifacts root is cwd/artifacts or MODELOPT_ARTIFACTS_ROOT)."
        ),
    )
    parser.add_argument(
        "--timing-cache",
        type=str,
        default=None,
        metavar="PATH",
        help="Timing cache path (default: <engine>.timing.cache).",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args after -- forwarded to trtexec (append; use to override or add flags).",
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    extra: list[str] = []
    if args.passthrough:
        pt = args.passthrough
        if pt and pt[0] == "--":
            pt = pt[1:]
        extra.extend(pt)

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 1

    stem = onnx_path.stem
    if args.engine_out:
        engine_path = Path(args.engine_out)
    else:
        engine_path = trt_engine_dir() / f"{stem}.engine"
    if args.timing_cache:
        timing_cache = Path(args.timing_cache)
    else:
        # Match shell: saveEngine=foo.engine -> foo.engine.timing.cache
        timing_cache = Path(str(engine_path) + ".timing.cache")

    ts = run_timestamp()
    log_path = args.log_file
    if log_path is None:
        log_path = str(default_build_trt_session_log(onnx_stem=stem, ts=ts))

    setup_logging("build_trt", log_file=log_path, verbose=args.verbose)
    log = logging.getLogger("build_trt")

    trtexec_bin = shutil.which("trtexec")
    if not trtexec_bin:
        log.error("trtexec not found in PATH. Use the TensorRT NGC container or install TensorRT tools.")
        return 127

    argv_exec = build_trtexec_argv(
        onnx_path=onnx_path,
        engine_path=engine_path,
        timing_cache=timing_cache,
        input_name=args.input_name,
        img_size=args.img_size,
        batch=args.batch,
        mode=args.mode,
        warm_up=args.warm_up,
        duration=args.duration,
        extra=extra,
    )
    # Use resolved trtexec path
    argv_exec[0] = trtexec_bin

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    timing_cache.parent.mkdir(parents=True, exist_ok=True)

    log.info("Running: %s", " ".join(argv_exec))
    log.info("Mode: %s", args.mode)
    rc = subprocess.call(argv_exec)
    if rc == 0:
        log.info("Engine: %s", engine_path.resolve())
        log.info("Timing cache: %s", timing_cache.resolve())
    else:
        log.error("trtexec exited with code %d", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
