#!/usr/bin/env python3
"""End-to-end PTQ pipeline: calib → quantize (with optional autotune) → build-trt → eval-trt → trt-bench → report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from model_opt_yolo.io_checks import validate_readable_file
from model_opt_yolo.logutil import add_logging_arguments, setup_logging
from model_opt_yolo.session_paths import (
    artifacts_root,
    default_calib_npy_path,
    default_pipeline_e2e_session_log,
    pipeline_e2e_session_eval_logs,
    pipeline_e2e_session_quant_logs,
    pipeline_e2e_session_root,
    pipeline_e2e_session_trt_logs,
    run_timestamp,
    safe_component,
    trt_engine_dir,
)

# Full PTQ grid supported by ``quantize`` (fp8/int8: max|entropy; int4: awq_clip|rtn_dq).
QUANT_COMBOS_ALL: list[tuple[str, str]] = [
    ("int8", "entropy"),
    ("int8", "max"),
    ("fp8", "entropy"),
    ("fp8", "max"),
    ("int4", "awq_clip"),
    ("int4", "rtn_dq"),
]


def _quantize_output_path(
    *,
    input_onnx: Path,
    quantize_mode: str,
    calibration_method: str,
    output_dir: Path,
    suffix: str,
) -> Path:
    stem = input_onnx.stem
    name = f"{stem}.{quantize_mode}.{calibration_method}{suffix}"
    return output_dir / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full ONNX PTQ + TensorRT workflow: build calibration data, "
            "quantize with optional integrated autotune (one or all mode/method pairs), "
            "build engine, COCO eval, trtexec bench, then "
            "emit a Markdown report (same aggregation as ``report-runs``). "
            "Requires GPU TensorRT stack (typ. NGC container)."
        )
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        metavar="PATH",
        help="Input ONNX model (FP32 export).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/coco/val2017",
        metavar="DIR",
        help="Image folder for calibration tensors (default: data/coco/val2017).",
    )
    parser.add_argument(
        "--calibration-data-size",
        type=int,
        default=500,
        metavar="N",
        help="Number of images for calib.npy (default: 500).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        metavar="N",
        help="Square letterbox size H=W for calib / build-trt profile (default: 640).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/coco/annotations/instances_val2017.json",
        metavar="PATH",
        help="COCO instances JSON for eval-trt (default: val2017 annotations).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="onnx_trt",
        choices=("onnx_trt", "efficient_nms", "ultralytics", "deepstream_yolo"),
        help="eval-trt --output-format: must match your ONNX export (default: onnx_trt).",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="images",
        metavar="NAME",
        help=(
            "ONNX input tensor name for build-trt shape profiles (default: images). "
            "Must match the graph input (e.g. ``input`` for some DeepStream/YOLO exports)."
        ),
    )
    parser.add_argument(
        "--build-mode",
        type=str,
        default="strongly-typed",
        choices=("best", "strongly-typed", "fp16", "fp16-int8"),
        help="build-trt --mode (default: strongly-typed for PTQ ONNX).",
    )
    parser.add_argument(
        "--autotune",
        type=str,
        default=None,
        choices=("quick", "default", "extensive"),
        metavar="PRESET",
        help=(
            "Enable integrated Q/DQ autotune inside the quantize step. "
            "Presets: quick, default, extensive. Autotune runs after "
            "calibration/Q-DQ insertion and selectively removes Q/DQ nodes "
            "that hurt TensorRT latency. Omit to skip autotune."
        ),
    )
    parser.add_argument(
        "--quant-matrix",
        type=str,
        default="default",
        choices=("default", "all"),
        help=(
            "default: single PTQ run (--quantize-mode / --calibration-method, default int8+entropy). "
            "all: run all supported mode/method pairs (6 runs; long)."
        ),
    )
    parser.add_argument(
        "--quantize-mode",
        type=str,
        default=None,
        choices=("fp8", "int8", "int4"),
        help="When --quant-matrix default: quantization precision (default: int8).",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default=None,
        help=(
            "When --quant-matrix default: max|entropy (fp8/int8) or awq_clip|rtn_dq (int4). "
            "Default: entropy for fp8/int8, rtn_dq for int4."
        ),
    )
    parser.add_argument(
        "--high-precision-dtype",
        type=str,
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help="Passed to quantize (default: fp32).",
    )
    parser.add_argument(
        "--bench-duration",
        type=int,
        default=60,
        metavar="SEC",
        help="trt-bench --duration (default: 60).",
    )
    parser.add_argument(
        "--bench-warm-up",
        type=int,
        default=500,
        metavar="MS",
        help="trt-bench --warm-up (default: 500).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one quant/config step fails, log and continue with remaining combos.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip Markdown report at the end.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "report-runs -o path (default: "
            "artifacts/pipeline_e2e/sessions/<session_id>/e2e_report.md)."
        ),
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Unique id for this run (default: auto YYYYMMDD-HHMMSS). "
            "All step logs and the Markdown report go under artifacts/pipeline_e2e/sessions/<id>/ "
            "so report-runs does not merge logs from older runs."
        ),
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is None else argv)

    err = validate_readable_file(args.onnx, label="ONNX model")
    if err:
        print(err, file=sys.stderr)
        return 1

    onnx_path = Path(args.onnx).expanduser().resolve()
    session_id = (args.session_id or "").strip() or run_timestamp()
    session_root = pipeline_e2e_session_root(session_id)
    manifest = {
        "session_id": session_id,
        "artifacts_root": str(artifacts_root().resolve()),
        "onnx": str(onnx_path),
    }
    (session_root / "session.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log_path = args.log_file
    if log_path is None:
        log_path = str(default_pipeline_e2e_session_log(session_id=session_id))

    setup_logging("pipeline_e2e", log_file=log_path, verbose=args.verbose)
    log = logging.getLogger("pipeline_e2e")
    verbose_flag = ["-v"] if args.verbose else []

    log.info("Session id: %s", session_id)
    log.info("Session directory: %s", session_root)
    log.info("Artifacts root: %s", artifacts_root())
    log.info("Orchestrator log: %s", log_path)

    def _default_calib_method_for_mode(qm: str) -> str:
        return "rtn_dq" if qm == "int4" else "entropy"

    def _resolve_combos() -> list[tuple[str, str]]:
        if args.quant_matrix == "all":
            return list(QUANT_COMBOS_ALL)
        qm = args.quantize_mode or "int8"
        cm = args.calibration_method or _default_calib_method_for_mode(qm)
        return [(qm, cm)]

    combos = _resolve_combos()
    log.info(
        "Quantization plan: %s (%d combo(s))",
        ", ".join(f"{a}.{b}" for a, b in combos),
        len(combos),
    )

    # --- calib (fixed output path so later steps know the exact .npy) ---
    from model_opt_yolo.calib_prep import main as calib_main

    images_resolved = Path(args.images_dir).expanduser().resolve()
    calib_npy = default_calib_npy_path(
        images_dir_name=images_resolved.name,
        img_size=args.img_size,
        n_images=args.calibration_data_size,
        ts=session_id,
    )
    calib_argv = [
        "--images_dir",
        str(images_resolved),
        "--calibration_data_size",
        str(args.calibration_data_size),
        "--img_size",
        str(args.img_size),
        "--output_path",
        str(calib_npy),
        *verbose_flag,
    ]
    log.info("Step: calib → %s", calib_npy)
    rc = calib_main(calib_argv)
    if rc != 0:
        log.error("calib failed with exit %d", rc)
        return rc
    if not calib_npy.is_file():
        log.error("Calibration file missing after calib: %s", calib_npy)
        return 1
    log.info("Using calibration data: %s", calib_npy)

    # --- quantize (with optional integrated autotune) ---
    if args.autotune:
        log.info("Autotune enabled (preset: %s) — integrated into quantize step.", args.autotune)
    else:
        log.info("Autotune not requested; quantizing input ONNX directly.")

    quant_dir = artifacts_root() / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".quant.onnx"

    from model_opt_yolo.quantize import main as quantize_main
    from model_opt_yolo.build_trt import main as build_trt_main
    from model_opt_yolo.eval_trt import main as eval_trt_main
    from model_opt_yolo.bench_trt import main as bench_trt_main
    from model_opt_yolo.report_runs import main as report_main

    session_trt_logs = pipeline_e2e_session_trt_logs(session_id)
    session_eval_logs = pipeline_e2e_session_eval_logs(session_id)
    session_quant_logs = pipeline_e2e_session_quant_logs(session_id)

    any_fail = False
    for qm, cm in combos:
        tag = f"{qm}.{cm}"
        step_ts = run_timestamp()
        log.info("========== Quant combo: %s ==========", tag)
        q_out = _quantize_output_path(
            input_onnx=onnx_path,
            quantize_mode=qm,
            calibration_method=cm,
            output_dir=quant_dir,
            suffix=suffix,
        )
        q_log = session_quant_logs / f"quantize_{safe_component(q_out.stem)}_{step_ts}.log"
        q_argv = [
            "--onnx_path",
            str(onnx_path),
            "--calibration_data",
            str(calib_npy),
            "--quantize_mode",
            qm,
            "--calibration_method",
            cm,
            "--high_precision_dtype",
            args.high_precision_dtype,
            "--output_dir",
            str(quant_dir),
            "--log-file",
            str(q_log),
            *verbose_flag,
        ]
        if args.autotune:
            q_argv.extend(["--autotune", args.autotune])
        autotune_label = f" +autotune={args.autotune}" if args.autotune else ""
        log.info("Step: quantize%s → %s (log: %s)", autotune_label, q_out.name, q_log)
        rc = quantize_main(q_argv)
        if rc != 0:
            log.error("quantize failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc
        if not q_out.is_file():
            log.error("Expected quantized ONNX missing: %s", q_out)
            any_fail = True
            if args.continue_on_error:
                continue
            return 1

        eng_stem = q_out.stem
        build_log = session_trt_logs / f"build_trt_{safe_component(eng_stem)}_{step_ts}.log"
        b_argv = [
            "--onnx",
            str(q_out),
            "--img-size",
            str(args.img_size),
            "--input-name",
            args.input_name,
            "--mode",
            args.build_mode,
            "--log-file",
            str(build_log),
            *verbose_flag,
        ]
        log.info("Step: build-trt (%s.engine) log: %s", eng_stem, build_log)
        rc = build_trt_main(b_argv)
        if rc != 0:
            log.error("build-trt failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc

        engine_path = trt_engine_dir() / f"{eng_stem}.engine"
        if not engine_path.is_file():
            log.error("Engine not found after build: %s", engine_path)
            any_fail = True
            if args.continue_on_error:
                continue
            return 1

        eval_log = session_eval_logs / f"eval_{safe_component(eng_stem)}_{step_ts}.log"
        e_argv = [
            "--engine",
            str(engine_path),
            "--output-format",
            args.output_format,
            "--images",
            str(Path(args.images_dir).expanduser().resolve()),
            "--annotations",
            str(Path(args.annotations).expanduser().resolve()),
            "--img-size",
            str(args.img_size),
            "--log-file",
            str(eval_log),
            *verbose_flag,
        ]
        log.info("Step: eval-trt (log: %s)", eval_log)
        rc = eval_trt_main(e_argv)
        if rc != 0:
            log.error("eval-trt failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc

        bench_log = session_trt_logs / f"trt_bench_{safe_component(eng_stem)}_{step_ts}.log"
        tb_argv = [
            "--engine",
            str(engine_path),
            "--duration",
            str(args.bench_duration),
            "--warm-up",
            str(args.bench_warm_up),
            "--log-file",
            str(bench_log),
            *verbose_flag,
        ]
        log.info("Step: trt-bench (log: %s)", bench_log)
        rc = bench_trt_main(tb_argv)
        if rc != 0:
            log.error("trt-bench failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc

    if any_fail and args.continue_on_error:
        log.warning("One or more steps failed (--continue-on-error); check logs above.")

    if args.no_report:
        log.info("Step: report skipped (--no-report).")
        return 1 if any_fail else 0

    rep = args.report_output
    if rep is None:
        rep = str(session_root / "e2e_report.md")
    Path(rep).parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Step: report-runs (session-scoped dirs: trt=%s eval=%s) → %s",
        session_trt_logs,
        session_eval_logs,
        rep,
    )
    rc = report_main(
        [
            "--trt-logs-dir",
            str(session_trt_logs.resolve()),
            "--eval-logs-dir",
            str(session_eval_logs.resolve()),
            "-o",
            rep,
        ]
    )
    if rc != 0:
        log.error("report-runs failed with exit %d", rc)
        return rc
    log.info("Done. Report: %s", rep)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
