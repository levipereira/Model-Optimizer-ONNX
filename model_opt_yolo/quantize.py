#!/usr/bin/env python3
"""Batch or single-file ONNX PTQ using NVIDIA Model Optimizer (CLI wrapper)."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

from model_opt_yolo.logutil import add_logging_arguments, setup_logging
from model_opt_yolo.session_paths import (
    artifacts_root,
    default_quantize_session_log,
    default_quantize_session_log_batch,
    run_timestamp,
)


def build_output_basename(
    stem: str,
    quantize_mode: str,
    calibration_method: str,
    suffix: str,
) -> str:
    """Unique name per mode+method so repeated runs do not overwrite."""
    return f"{stem}.{quantize_mode}.{calibration_method}{suffix}"


def run_quantize(
    onnx_path: str,
    calib_path: str,
    output_path: str,
    quantize_mode: str,
    calibration_method: str,
    high_precision_dtype: str,
    extra_args: list[str],
    log: logging.Logger,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "modelopt.onnx.quantization",
        f"--onnx_path={onnx_path}",
        f"--quantize_mode={quantize_mode}",
        f"--calibration_data_path={calib_path}",
        f"--calibration_method={calibration_method}",
        f"--output_path={output_path}",
        f"--high_precision_dtype={high_precision_dtype}",
        *extra_args,
    ]
    log.info("Running: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize one or more ONNX models with Model Optimizer PTQ.")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Single ONNX file.",
    )
    parser.add_argument(
        "--onnx_glob",
        type=str,
        default=None,
        help="Glob under models/ or custom path, e.g. 'models/*.onnx'.",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="Path to calib.npy (default from calib: under <artifacts root>/calibration/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for quantized ONNX files (default: <artifacts root>/quantized; see MODELOPT_ARTIFACTS_ROOT).",
    )
    parser.add_argument("--quantize_mode", type=str, default="int8", choices=("fp8", "int8", "int4"))
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="entropy",
        help="max|entropy for fp8/int8; awq_clip|rtn_dq for int4.",
    )
    parser.add_argument(
        "--high_precision_dtype",
        type=str,
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help=(
            "Dtype for non-quantized ops after PTQ. Default fp32 avoids Model Optimizer's "
            "post-pass FP16 autocast, which often fails shape_inference on YOLO/TRT exports "
            "(dynamic detection head). Use fp16 only if you know the graph supports it."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".quant.onnx",
        help="Output filename suffix (default: .quant.onnx).",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra args after -- passed to modelopt.onnx.quantization (e.g. -- --calibrate_per_node).",
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    output_dir = args.output_dir if args.output_dir is not None else str(artifacts_root() / "quantized")

    extra: list[str] = []
    if args.passthrough:
        pt = args.passthrough
        if pt and pt[0] == "--":
            pt = pt[1:]
        extra.extend(pt)

    if args.onnx_path and args.onnx_glob:
        print("Use either --onnx_path or --onnx_glob, not both.", file=sys.stderr)
        return 2
    if not args.onnx_path and not args.onnx_glob:
        args.onnx_glob = "models/*.onnx"

    paths: list[str] = []
    if args.onnx_path:
        paths = [args.onnx_path]
    else:
        paths = sorted(glob.glob(args.onnx_glob))
        if not paths:
            print(f"No files matched: {args.onnx_glob}", file=sys.stderr)
            return 1

    ts = run_timestamp()
    if args.log_file is None:
        if len(paths) == 1:
            log_path = str(
                default_quantize_session_log(
                    onnx_stem=Path(paths[0]).stem,
                    quantize_mode=args.quantize_mode,
                    calibration_method=args.calibration_method,
                    ts=ts,
                )
            )
        else:
            log_path = str(
                default_quantize_session_log_batch(
                    n_files=len(paths),
                    quantize_mode=args.quantize_mode,
                    calibration_method=args.calibration_method,
                    ts=ts,
                )
            )
    else:
        log_path = args.log_file

    setup_logging("quantize_onnx", log_file=log_path, verbose=args.verbose)
    log = logging.getLogger("quantize_onnx")

    os.makedirs(output_dir, exist_ok=True)
    log.info("Output directory: %s", output_dir)
    rc = 0
    for p in paths:
        stem = Path(p).stem
        name = build_output_basename(stem, args.quantize_mode, args.calibration_method, args.suffix)
        out = os.path.join(output_dir, name)
        log.debug("Quantizing %s -> %s", p, out)
        rc = run_quantize(
            onnx_path=p,
            calib_path=args.calibration_data,
            output_path=out,
            quantize_mode=args.quantize_mode,
            calibration_method=args.calibration_method,
            high_precision_dtype=args.high_precision_dtype,
            extra_args=extra,
            log=log,
        )
        if rc != 0:
            log.error("Failed: %s (exit %d)", p, rc)
            break
        log.info("Finished: %s", out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
