"""Entry point for the ``model-opt-yolo`` console script."""

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
        from model_opt_yolo.download_coco import main as run

        return run(rest)
    if cmd == "calib":
        from model_opt_yolo.calib_prep import main as run

        return run(rest)
    if cmd == "quantize":
        from model_opt_yolo.quantize import main as run

        return run(rest)
    if cmd in ("eval-trt", "eval_trt"):
        from model_opt_yolo.eval_trt import main as run

        return run(rest)
    if cmd in ("build-trt", "build_trt"):
        from model_opt_yolo.build_trt import main as run

        return run(rest)
    if cmd in ("trt-bench", "trt_bench"):
        from model_opt_yolo.bench_trt import main as run

        return run(rest)
    if cmd in ("report-runs", "report_runs"):
        from model_opt_yolo.report_runs import main as run

        return run(rest)
    if cmd in ("pipeline-e2e", "pipeline_e2e", "e2e"):
        from model_opt_yolo.pipeline_e2e import main as run

        return run(rest)

    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    _print_usage(to_stdout=False)
    return 2


def _print_usage(*, to_stdout: bool) -> None:
    out = sys.stdout if to_stdout else sys.stderr
    print(
        """model-opt-yolo — ONNX PTQ / TensorRT helpers for YOLO-style models

Usage:
  model-opt-yolo <command> [options]

Commands:
  download-coco  Download COCO val2017 + annotations into data/coco (or --output-dir)
  calib       Build calibration .npy from image folders
  quantize    NVIDIA Model Optimizer ONNX PTQ (--autotune quick|default|extensive for Q/DQ tuning)
  build-trt   TensorRT engine from ONNX (--mode strongly-typed|best|fp16|fp16-int8)
  trt-bench   trtexec throughput/latency on an existing .engine (--loadEngine; no rebuild)
  eval-trt    COCO mAP on TRT engines (EfficientNMS, Ultralytics, or DeepStream-Yolo output)
  report-runs Scan trt_bench / eval logs → Markdown report (tables + Mermaid charts)
  pipeline-e2e  End-to-end: calib → quantize [+autotune] → build-trt → eval-trt → trt-bench → report

Examples:
  model-opt-yolo download-coco --output-dir data/coco
  model-opt-yolo calib --images_dir data/coco/val2017 --calibration_data_size 500 --img_size 640
  model-opt-yolo quantize --calibration_data artifacts/calibration/calib_....npy \\
      --onnx_path models/yolo.onnx
  model-opt-yolo quantize --calibration_data artifacts/calibration/calib_....npy \\
      --onnx_path models/yolo.onnx --autotune default
  model-opt-yolo build-trt --onnx artifacts/quantized/model.int8.entropy.quant.onnx
  model-opt-yolo trt-bench --engine artifacts/trt_engine/model.int8.entropy.quant.engine
  model-opt-yolo eval-trt --output-format onnx_trt --engine artifacts/trt_engine/model.int8.entropy.quant.engine

See also: model-opt-yolo <command> --help
""",
        file=out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
