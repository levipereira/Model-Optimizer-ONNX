#!/usr/bin/env python3
"""End-to-end PTQ pipeline: calib → optional FP16 baseline on original ONNX → quantize → build-trt → eval-trt → trt-bench → report."""

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
    default_pipeline_e2e_session_log,
    effective_session_id,
    pipeline_e2e_session_calib_npy_path,
    pipeline_e2e_session_calib_prep_log,
    pipeline_e2e_session_eval_logs,
    pipeline_e2e_session_quant_logs,
    pipeline_e2e_session_quantized_dir,
    pipeline_e2e_session_root,
    pipeline_e2e_session_trt_engine_dir,
    pipeline_e2e_session_trt_logs,
    run_timestamp,
    safe_component,
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

_METHODS_FOR_MODE: dict[str, tuple[str, ...]] = {
    "int8": ("entropy", "max"),
    "fp8": ("entropy", "max"),
    "int4": ("awq_clip", "rtn_dq"),
}


def parse_quant_matrix_spec(spec: str) -> list[tuple[str, str]]:
    """Expand ``--quant-matrix`` into an ordered, de-duplicated list of (mode, calibration_method).

    * ``all`` — full grid (same as ``int8.all,fp8.all,int4.all``).
    * ``<mode>.all`` — both calibration methods for ``int8``, ``fp8``, or ``int4``.
    * ``<mode>.<method>`` — one combo (e.g. ``int8.entropy``, ``int4.rtn_dq``).
    * Comma-separated — union of the above (e.g. ``int8.all,fp8.entropy``).
    """
    raw = spec.strip()
    if not raw:
        raise ValueError("--quant-matrix must not be empty")
    if raw == "all":
        return list(QUANT_COMBOS_ALL)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--quant-matrix must not be empty")
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for part in parts:
        if "." not in part:
            raise ValueError(
                f"invalid quant-matrix fragment {part!r}; use mode.method, mode.all, or the keyword all"
            )
        mode, rest = part.split(".", 1)
        mode_l = mode.strip().lower()
        rest_l = rest.strip().lower()
        if mode_l not in _METHODS_FOR_MODE:
            raise ValueError(
                f"unknown mode {mode!r} in {part!r}; expected int8, fp8, or int4"
            )
        valid = _METHODS_FOR_MODE[mode_l]
        if rest_l == "all":
            chunk = [(mode_l, m) for m in valid]
        else:
            if rest_l not in valid:
                raise ValueError(
                    f"unknown method {rest!r} for mode {mode_l}; "
                    f"expected one of {list(valid)} or all"
                )
            chunk = [(mode_l, rest_l)]
        for pair in chunk:
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out


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
            "optionally TensorRT FP16 on the original ONNX (baseline), then "
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
        nargs="?",
        const="quick",
        type=str,
        default=None,
        choices=("quick", "default", "extensive"),
        metavar="PRESET",
        help=(
            "Enable integrated Q/DQ autotune inside the quantize step. "
            "Presets: quick (default if flag is given without a value), default, extensive. "
            "Autotune runs after calibration/Q-DQ insertion and selectively removes Q/DQ nodes "
            "that hurt TensorRT latency. Omit for no autotune. "
            "FP8 subprocesses never receive --autotune (Model Optimizer limitation); "
            "int8/int4 steps get --autotune when this flag is set."
        ),
    )
    parser.add_argument(
        "--quant-matrix",
        type=str,
        default="int8.entropy",
        metavar="SPEC",
        help=(
            "Which PTQ combos to run. Keyword ``all`` = full 6-run grid. "
            "Otherwise use ``mode.method``, ``mode.all``, or comma-separated unions "
            "(e.g. int8.entropy, fp8.all, int8.all,fp8.entropy). "
            "Modes: int8, fp8, int4. Methods: int8/fp8 → entropy|max; int4 → awq_clip|rtn_dq. "
            "Default: int8.entropy (single run)."
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
        "--quantize-profile",
        type=str,
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            "Optional YAML profile passed to each quantize step as --profile "
            "(include/exclude op types, nodes, etc.). See model_opt_yolo/profiles/."
        ),
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
        "--no-fp16-baseline",
        action="store_true",
        help=(
            "Skip TensorRT FP16 baseline on the original ONNX (--mode fp16): "
            "build-trt → eval-trt → trt-bench before the PTQ loop. "
            "The baseline compares full-precision export vs quantized engines in report-runs."
        ),
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
            "Unique id for this run (default: SESSION_ID env var if set, else auto YYYYMMDD-HHMMSS). "
            "All pipeline-e2e outputs (calibration .npy, quantized ONNX, engines, eval JSON, trt/eval logs, "
            "report) go under artifacts/pipeline_e2e/sessions/<id>/ — not the global artifacts/ paths used "
            "when running calib/quantize/build-trt manually. CLI overrides SESSION_ID."
        ),
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        combos = parse_quant_matrix_spec(args.quant_matrix)
    except ValueError as exc:
        print(f"Invalid --quant-matrix: {exc}", file=sys.stderr)
        return 2

    err = validate_readable_file(args.onnx, label="ONNX model")
    if err:
        print(err, file=sys.stderr)
        return 1

    onnx_path = Path(args.onnx).expanduser().resolve()
    session_id = effective_session_id(args.session_id) or run_timestamp()
    session_root = pipeline_e2e_session_root(session_id)
    manifest = {
        "session_id": session_id,
        "artifacts_root": str(artifacts_root().resolve()),
        "onnx": str(onnx_path),
        "session_root": str(session_root.resolve()),
        "e2e_calibration_dir": str((session_root / "calibration").resolve()),
        "e2e_quantized_dir": str((session_root / "quantized").resolve()),
        "e2e_trt_engine_dir": str((session_root / "trt_engine").resolve()),
        "e2e_predictions_dir": str((session_root / "predictions").resolve()),
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

    plan_has_fp8 = any(qm == "fp8" for qm, _ in combos)
    log.info(
        "Quantization plan: %s (%d combo(s))",
        ", ".join(f"{a}.{b}" for a, b in combos),
        len(combos),
    )

    if args.autotune and plan_has_fp8:
        log.warning(
            "With --autotune: int8 and int4 quantize steps will pass --autotune=%s. "
            "FP8 steps run standard PTQ only (no --autotune on the quantize subprocess).",
            args.autotune,
        )

    # --- calib (outputs under session dir, not global artifacts/calibration) ---
    from model_opt_yolo.calib_prep import main as calib_main

    images_resolved = Path(args.images_dir).expanduser().resolve()
    calib_npy = pipeline_e2e_session_calib_npy_path(
        session_id=session_id,
        images_dir_name=images_resolved.name,
        img_size=args.img_size,
        n_images=args.calibration_data_size,
        ts=session_id,
    )
    calib_prep_ts = run_timestamp()
    calib_prep_log = pipeline_e2e_session_calib_prep_log(
        session_id=session_id,
        images_dir_name=images_resolved.name,
        ts=calib_prep_ts,
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
        "--log-file",
        str(calib_prep_log),
        *verbose_flag,
    ]
    log.info("Step: calib → %s (log → %s)", calib_npy, calib_prep_log)
    rc = calib_main(calib_argv)
    if rc != 0:
        log.error("calib failed with exit %d", rc)
        return rc
    if not calib_npy.is_file():
        log.error("Calibration file missing after calib: %s", calib_npy)
        return 1
    log.info("Using calibration data: %s", calib_npy)

    # --- quantize (optional --autotune; fp8 never receives --autotune) ---
    if args.autotune:
        if plan_has_fp8:
            log.info(
                "Autotune preset %s — passed to int8/int4 quantize; fp8 quantize runs without --autotune (standard PTQ).",
                args.autotune,
            )
        else:
            log.info("Autotune enabled (preset: %s) — integrated into quantize step.", args.autotune)
    else:
        log.info("Autotune not requested; quantizing input ONNX directly.")

    quant_dir = pipeline_e2e_session_quantized_dir(session_id)
    suffix = ".quant.onnx"

    from model_opt_yolo.quantize import main as quantize_main
    from model_opt_yolo.build_trt import main as build_trt_main
    from model_opt_yolo.eval_trt import main as eval_trt_main
    from model_opt_yolo.bench_trt import main as bench_trt_main
    from model_opt_yolo.report_runs import main as report_main

    session_trt_logs = pipeline_e2e_session_trt_logs(session_id)
    session_trt_engine_dir = pipeline_e2e_session_trt_engine_dir(session_id)
    session_eval_logs = pipeline_e2e_session_eval_logs(session_id)
    session_quant_logs = pipeline_e2e_session_quant_logs(session_id)

    any_fail = False

    # --- FP16 baseline: original ONNX + trtexec --fp16 (same session logs as PTQ runs) ---
    if not args.no_fp16_baseline:
        fp16_ts = run_timestamp()
        fp16_stem = f"{onnx_path.stem}.fp16"
        engine_fp16 = session_trt_engine_dir / f"{fp16_stem}.engine"
        build_log_fp16 = session_trt_logs / f"build_trt_{safe_component(fp16_stem)}_{fp16_ts}.log"
        log.info("========== FP16 baseline (original ONNX, --mode fp16) ==========")
        b_fp16 = [
            "--onnx",
            str(onnx_path),
            "--img-size",
            str(args.img_size),
            "--input-name",
            args.input_name,
            "--mode",
            "fp16",
            "--engine-out",
            str(engine_fp16),
            "--log-file",
            str(build_log_fp16),
            *verbose_flag,
        ]
        log.info(
            "Starting build-trt FP16 baseline (engine → %s; session log → %s)",
            engine_fp16.resolve(),
            build_log_fp16.resolve(),
        )
        rc = build_trt_main(b_fp16)
        if rc != 0:
            log.error("build-trt FP16 baseline failed (exit %d)", rc)
            any_fail = True
            if not args.continue_on_error:
                return rc
        elif not engine_fp16.is_file():
            log.error("FP16 baseline engine missing after build: %s", engine_fp16)
            any_fail = True
            if not args.continue_on_error:
                return 1
        else:
            log.info(
                "FP16 build-trt OK: %s (%d B)",
                engine_fp16.resolve(),
                engine_fp16.stat().st_size,
            )
            eval_log_fp16 = session_eval_logs / f"eval_{safe_component(fp16_stem)}_{fp16_ts}.log"
            pred_json_fp16 = session_root / "predictions" / f"{fp16_stem}_predictions.json"
            e_fp16 = [
                "--engine",
                str(engine_fp16),
                "--output-format",
                args.output_format,
                "--images",
                str(images_resolved),
                "--annotations",
                str(Path(args.annotations).expanduser().resolve()),
                "--img-size",
                str(args.img_size),
                "--save-json",
                str(pred_json_fp16),
                "--log-file",
                str(eval_log_fp16),
                *verbose_flag,
            ]
            log.info("Starting eval-trt FP16 baseline (log → %s)", eval_log_fp16.resolve())
            rc = eval_trt_main(e_fp16)
            if rc != 0:
                log.error("eval-trt FP16 baseline failed (exit %d)", rc)
                any_fail = True
                if not args.continue_on_error:
                    return rc
            else:
                log.info("FP16 eval-trt OK (log → %s)", eval_log_fp16.resolve())
                bench_log_fp16 = session_trt_logs / f"trt_bench_{safe_component(fp16_stem)}_{fp16_ts}.log"
                tb_fp16 = [
                    "--engine",
                    str(engine_fp16),
                    "--duration",
                    str(args.bench_duration),
                    "--warm-up",
                    str(args.bench_warm_up),
                    "--log-file",
                    str(bench_log_fp16),
                    *verbose_flag,
                ]
                log.info("Starting trt-bench FP16 baseline (log → %s)", bench_log_fp16.resolve())
                rc = bench_trt_main(tb_fp16)
                if rc != 0:
                    log.error("trt-bench FP16 baseline failed (exit %d)", rc)
                    any_fail = True
                    if not args.continue_on_error:
                        return rc
                else:
                    log.info("FP16 trt-bench OK (log → %s)", bench_log_fp16.resolve())

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
        pass_autotune_flag = bool(args.autotune) and qm != "fp8"
        if pass_autotune_flag:
            q_argv.extend(["--autotune", args.autotune])
        if args.quantize_profile:
            q_argv.extend(["--profile", args.quantize_profile])
        if args.autotune:
            if qm == "fp8":
                autotune_label = " (fp8: standard PTQ, no --autotune)"
            else:
                autotune_label = f" +autotune={args.autotune}"
        else:
            autotune_label = ""
        log.info(
            "Starting quantize%s: ONNX out → %s; quantize log → %s",
            autotune_label,
            q_out.resolve(),
            q_log.resolve(),
        )
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
        log.info(
            "quantize OK: %s (%d B); session log exists=%s",
            q_out.resolve(),
            q_out.stat().st_size,
            q_log.is_file(),
        )

        eng_stem = q_out.stem
        build_log = session_trt_logs / f"build_trt_{safe_component(eng_stem)}_{step_ts}.log"
        engine_path = session_trt_engine_dir / f"{eng_stem}.engine"
        b_argv = [
            "--onnx",
            str(q_out),
            "--img-size",
            str(args.img_size),
            "--input-name",
            args.input_name,
            "--mode",
            args.build_mode,
            "--engine-out",
            str(engine_path),
            "--log-file",
            str(build_log),
            *verbose_flag,
        ]
        log.info(
            "Starting build-trt: engine → %s; session log → %s",
            engine_path.resolve(),
            build_log.resolve(),
        )
        rc = build_trt_main(b_argv)
        if rc != 0:
            log.error("build-trt failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc

        if not engine_path.is_file():
            log.error("Engine not found after build: %s", engine_path)
            any_fail = True
            if args.continue_on_error:
                continue
            return 1
        log.info(
            "build-trt OK: %s (%d B)",
            engine_path.resolve(),
            engine_path.stat().st_size,
        )

        eval_log = session_eval_logs / f"eval_{safe_component(eng_stem)}_{step_ts}.log"
        pred_json = session_root / "predictions" / f"{eng_stem}_predictions.json"
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
            "--save-json",
            str(pred_json),
            "--log-file",
            str(eval_log),
            *verbose_flag,
        ]
        log.info("Starting eval-trt (log → %s)", eval_log.resolve())
        rc = eval_trt_main(e_argv)
        if rc != 0:
            log.error("eval-trt failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc
        log.info("eval-trt OK (log → %s)", eval_log.resolve())

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
        log.info("Starting trt-bench (log → %s)", bench_log.resolve())
        rc = bench_trt_main(tb_argv)
        if rc != 0:
            log.error("trt-bench failed for %s (exit %d)", tag, rc)
            any_fail = True
            if args.continue_on_error:
                continue
            return rc
        log.info("trt-bench OK (log → %s)", bench_log.resolve())

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
            "--session-id",
            session_id,
            "--merge-global-logs",
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
