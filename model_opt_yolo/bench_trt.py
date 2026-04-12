#!/usr/bin/env python3
"""Benchmark a TensorRT ``.engine`` with ``trtexec`` (load plan, no rebuild).

Follows NVIDIA guidance: ``--warmUp``, ``--iterations``, ``--duration``, plus
``--useCudaGraph``, ``--useSpinWait``, ``--noDataTransfers``. See
https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
("Performance Benchmarking using trtexec").
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from model_opt_yolo.io_checks import validate_readable_file
from model_opt_yolo.logutil import add_logging_arguments, setup_logging
from model_opt_yolo.session_paths import default_trt_bench_session_log, effective_session_id, run_timestamp

_TRT_BEST_PRACTICES = (
    "https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html"
)


def trt_bench_trtexec_argv(
    *,
    engine_path: Path,
    warm_up: int,
    iterations: int,
    duration: int,
    extra: list[str],
) -> list[str]:
    """``trtexec`` argv for ``--loadEngine`` performance run; *extra* appended last."""
    eng_s = str(engine_path.resolve())
    base: list[str] = [
        "--loadEngine=" + eng_s,
        f"--warmUp={warm_up}",
        f"--iterations={iterations}",
        f"--duration={duration}",
        "--useCudaGraph",
        "--useSpinWait",
        "--noDataTransfers",
    ]
    return ["trtexec", *base, *extra]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run trtexec performance benchmarking on an existing TensorRT .engine (--loadEngine). "
            "Input shapes are defined by the serialized plan (no --batch/--img-size). "
            f"See {_TRT_BEST_PRACTICES} (Performance Benchmarking using trtexec)."
        )
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        metavar="PATH",
        help="Existing .engine file (required).",
    )
    parser.add_argument(
        "--warm-up",
        type=int,
        default=500,
        metavar="MS",
        help="trtexec --warmUp: warmup time in milliseconds before measuring (default: 500).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        metavar="N",
        help="trtexec --iterations: minimum inference iterations (default: 100).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        metavar="SEC",
        help="trtexec --duration: wall-clock measurement window in seconds (default: 60).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Optional pipeline session id: when set (and --log-file is omitted), "
            "logs go under artifacts/pipeline_e2e/sessions/<id>/trt_engine/logs/. "
            "If omitted, SESSION_ID is used when set. Same as build-trt/eval-trt/report-runs."
        ),
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args after -- forwarded to trtexec.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    extra: list[str] = []
    if args.passthrough:
        pt = args.passthrough
        if pt and pt[0] == "--":
            pt = pt[1:]
        extra.extend(pt)

    err = validate_readable_file(args.engine, label="TensorRT engine")
    if err:
        print(err, file=sys.stderr)
        return 1
    engine_path = Path(args.engine).expanduser().resolve()

    stem = engine_path.stem
    ts = run_timestamp()
    log_path = args.log_file
    sid = effective_session_id(args.session_id)
    if log_path is None:
        log_path = str(default_trt_bench_session_log(engine_stem=stem, ts=ts, session_id=sid))

    setup_logging("trt_bench", log_file=log_path, verbose=args.verbose)
    log = logging.getLogger("trt_bench")

    log.info("TensorRT performance benchmarking (trtexec --loadEngine)")
    log.info("Reference: %s", _TRT_BEST_PRACTICES)
    log.info("Engine: %s", engine_path.resolve())

    trtexec_bin = shutil.which("trtexec")
    if not trtexec_bin:
        log.error("trtexec not found in PATH. Use the TensorRT NGC container or install TensorRT tools.")
        return 127

    argv_exec = trt_bench_trtexec_argv(
        engine_path=engine_path,
        warm_up=args.warm_up,
        iterations=args.iterations,
        duration=args.duration,
        extra=extra,
    )
    argv_exec[0] = trtexec_bin

    log.info(
        "warmUp=%sms iterations=%s duration=%ss",
        args.warm_up,
        args.iterations,
        args.duration,
    )
    log.info("Running: %s", " ".join(argv_exec))

    # Stream trtexec to the console and capture full output for the session log file.
    proc = subprocess.Popen(
        argv_exec,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            sys.stderr.write(line)
            chunks.append(line)
    rc = proc.wait()
    trt_out = "".join(chunks).rstrip()
    if trt_out:
        log.info("--- trtexec output ---\n%s", trt_out)
    if rc == 0:
        log.info("trtexec finished successfully.")
    else:
        log.error("trtexec exited with code %d", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
