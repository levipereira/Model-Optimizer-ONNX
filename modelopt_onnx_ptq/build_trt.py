#!/usr/bin/env python3
"""Run ``trtexec`` to build a TensorRT engine from ONNX (quantized or FP).

``--mode``: ``strongly-typed`` (default), ``best``, ``fp16``, or ``fp16-int8``.
For **quantized / PTQ** ONNX, ``strongly-typed`` (``--stronglyTyped``) matches Q/DQ
types; use ``best`` or ``fp16`` mainly for non-quantized exports. Benchmark with ``trt-bench``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from modelopt_onnx_ptq.io_checks import validate_readable_file
from modelopt_onnx_ptq.logutil import add_logging_arguments, setup_logging
from modelopt_onnx_ptq.session_paths import (
    default_build_trt_session_log,
    default_trt_engine_filename,
    effective_session_id,
    run_timestamp,
    trt_engine_dir,
)


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
    extra: list[str],
) -> list[str]:
    """Base ``trtexec`` argument list; *extra* is appended last for overrides."""
    onnx_s = str(onnx_path.resolve())
    eng_s = str(engine_path.resolve())
    cache_s = str(timing_cache.resolve())

    shp = _shape(batch, img_size)
    head = [
        "--onnx=" + onnx_s,
        "--saveEngine=" + eng_s,
    ]
    if mode == "strongly-typed":
        precision = ["--stronglyTyped"]
    elif mode == "best":
        precision = ["--best"]
    elif mode == "fp16":
        precision = ["--fp16"]
    elif mode == "fp16-int8":
        precision = ["--fp16", "--int8"]
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    tail = [
        f"--minShapes={input_name}:{shp}",
        f"--optShapes={input_name}:{shp}",
        f"--maxShapes={input_name}:{shp}",
        "--timingCacheFile=" + cache_s,
    ]
    base = [*head, *precision, *tail]

    return ["trtexec", *base, *extra]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TensorRT .engine from ONNX via trtexec. "
            "Modes: strongly-typed (default), best, fp16, fp16-int8 — default strongly-typed for PTQ/quantized ONNX. "
            "Benchmark: modelopt-onnx-ptq trt-bench --engine PATH. "
            "Append extra trtexec args after -- for overrides."
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
        help="Batch dimension for min/opt/max shapes (all modes use the same profile).",
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
        choices=("best", "strongly-typed", "fp16", "fp16-int8"),
        default="strongly-typed",
        help=(
            "strongly-typed (default): --stronglyTyped; use for quantized PTQ ONNX (Q/DQ). "
            "best: --best (often better for non-quantized FP graphs). "
            "fp16: --fp16 (typ. non-quantized ONNX). fp16-int8: --fp16 --int8. Same min/opt/max shape. "
            "Trailing args after -- are forwarded to trtexec."
        ),
    )
    parser.add_argument(
        "--engine-out",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output .engine path (default: <artifacts>/trt_engine/<onnx-stem>.b<batch>_i<img-size>.engine; "
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
        "--session-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Optional pipeline session id: when set (and --log-file is omitted), "
            "logs go under artifacts/pipeline_e2e/sessions/<id>/trt_engine/logs/. "
            "If omitted, the SESSION_ID environment variable is used when set. "
            "``report-runs --session-id`` or SESSION_ID picks up the same logs."
        ),
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

    err = validate_readable_file(args.onnx, label="ONNX model")
    if err:
        print(err, file=sys.stderr)
        return 1
    onnx_path = Path(args.onnx).expanduser().resolve()

    stem = onnx_path.stem
    if args.engine_out:
        engine_path = Path(args.engine_out)
    else:
        engine_path = trt_engine_dir() / default_trt_engine_filename(
            onnx_stem=stem, batch=args.batch, img_size=args.img_size
        )
    if args.timing_cache:
        timing_cache = Path(args.timing_cache)
    else:
        # Match shell: saveEngine=foo.engine -> foo.engine.timing.cache
        timing_cache = Path(str(engine_path) + ".timing.cache")

    ts = run_timestamp()
    log_path = args.log_file
    sid = effective_session_id(args.session_id)
    if log_path is None:
        log_path = str(default_build_trt_session_log(onnx_stem=stem, ts=ts, session_id=sid))

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
