"""Entry point for the ``modelopt-onnx-ptq`` console script."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_usage(to_stdout=True)
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd in ("download-coco", "download_coco"):
        from modelopt_onnx_ptq.download_coco import main as run

        return run(rest)
    if cmd == "calib":
        from modelopt_onnx_ptq.calib_prep import main as run

        return run(rest)
    if cmd == "quantize":
        from modelopt_onnx_ptq.quantize import main as run

        return run(rest)
    if cmd in ("eval-trt", "eval_trt"):
        from modelopt_onnx_ptq.eval_trt import main as run

        return run(rest)
    if cmd in ("build-trt", "build_trt"):
        from modelopt_onnx_ptq.build_trt import main as run

        return run(rest)
    if cmd in ("trt-bench", "trt_bench"):
        from modelopt_onnx_ptq.bench_trt import main as run

        return run(rest)
    if cmd in ("report-runs", "report_runs"):
        from modelopt_onnx_ptq.report_runs import main as run

        return run(rest)
    if cmd in ("pipeline-e2e", "pipeline_e2e", "e2e"):
        from modelopt_onnx_ptq.pipeline_e2e import main as run

        return run(rest)
    if cmd in ("trex-analyze", "trex_analyze"):
        from modelopt_onnx_ptq.trex_analyze import main as run

        return run(rest)

    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    _print_usage(to_stdout=False)
    return 2


def _print_usage(*, to_stdout: bool) -> None:
    out = sys.stdout if to_stdout else sys.stderr
    print(
        """modelopt-onnx-ptq — ONNX PTQ / TensorRT helpers for exported ONNX models

Usage:
  modelopt-onnx-ptq <command> [options]

Commands:
  download-coco  Download COCO val2017 + annotations into data/coco (or --output-dir)
  calib       Build calibration .npy from image folders
  quantize    NVIDIA Model Optimizer ONNX PTQ (--autotune quick|default|extensive for Q/DQ tuning)
  build-trt   TensorRT engine from ONNX (--mode strongly-typed|best|fp16|fp16-int8)
  trt-bench   trtexec throughput/latency on an existing .engine (--loadEngine; no rebuild)
  eval-trt    COCO mAP on TRT engines (EfficientNMS, Ultralytics, or DeepStream-Yolo output)
  report-runs Scan trt_bench / eval logs → Markdown report (tables + charts); --session-id uses pipeline_e2e/sessions/…
  pipeline-e2e  End-to-end: calib → FP16 baseline → quantize [+autotune] → build-trt → eval-trt → trt-bench → report
  trex-analyze  trtexec build+profile; at most one of --compare | --graph | --report | none (needs trex)

Examples:
  modelopt-onnx-ptq download-coco --output-dir data/coco
  modelopt-onnx-ptq calib --images_dir data/coco/val2017 --calibration_data_size 500 --img_size 640
  modelopt-onnx-ptq quantize --calibration_data artifacts/calibration/calib_....npy \\
      --onnx_path models/yolo.onnx
  modelopt-onnx-ptq quantize --calibration_data artifacts/calibration/calib_....npy \\
      --onnx_path models/yolo.onnx --autotune default
  modelopt-onnx-ptq quantize --calibration_data artifacts/calibration/calib_....npy \\
      --onnx_path models/yolo.onnx --profile matmul_fp_exclude
  modelopt-onnx-ptq build-trt --onnx artifacts/quantized/model.int8.entropy.quant.onnx --img-size 640 --batch 1
  modelopt-onnx-ptq trt-bench --engine artifacts/trt_engine/model.int8.entropy.quant.b1_i640.engine
  modelopt-onnx-ptq eval-trt --output-format onnx_trt --engine artifacts/trt_engine/model.int8.entropy.quant.b1_i640.engine

See also: modelopt-onnx-ptq <command> --help
""",
        file=out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
