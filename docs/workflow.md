# Workflow

## Overview

**Autotune** is optional; **COCO (or equivalent images) + `calib.npy`** are required for the standard PTQ path.

```mermaid
flowchart TD
  A[PyTorch weights — .pt] --> B[Export to ONNX<br/>--dynamic · --simplify · opset ≥ 18]
  B --> C[models/*.onnx]
  C --> H[download-coco]
  H --> I[model-opt-yolo calib]
  I --> J[artifacts/calibration/*.npy]
  J --> K{Autotune Q/DQ<br/>optional}
  K -->|yes| E[model-opt-yolo autotune]
  E --> F[artifacts/autotune/.../<br/>optimized_final.onnx]
  F --> Q[model-opt-yolo quantize]
  K -->|no| Q
  Q --> L[artifacts/quantized/*.quant.onnx]
  L --> M[model-opt-yolo build-trt]
  M --> N[artifacts/trt_engine/*.engine]
  N --> O[model-opt-yolo eval-trt<br/>COCO mAP]
```

| Step | Action |
|------|--------|
| 1 | Export your detector to **ONNX** → place under `models/` |
| 2 | **COCO val** — images + annotations for calib and eval (`model-opt-yolo download-coco`), or your own dataset layout |
| 3 | **Calibration** — build `calib.npy` from images (`model-opt-yolo calib`) |
| 4 | *(Optional)* **Autotune** — Q/DQ placement for TensorRT (`model-opt-yolo autotune`) |
| 5 | **PTQ** — quantize using calibration data (`model-opt-yolo quantize`; ONNX is `models/*.onnx` or `optimized_final.onnx` if you autotuned) |
| 6 | **Engine** — `model-opt-yolo build-trt --onnx …` (default `--mode best`; see [CLI reference](cli-reference.md#model-opt-yolo-build-trt) for YOLO vs `strongly-typed`) |
| 7 | **Eval** — COCO mAP (`model-opt-yolo eval-trt --output-format …`) — set **`--output-format`**: **`onnx_trt`** (four tensors; [levipereira/ultralytics](https://github.com/levipereira/ultralytics) `onnx_trt`), **`ultralytics`**, or **`deepstream_yolo`** (`efficient_nms` is an alias for `onnx_trt`) — see [CLI reference](cli-reference.md#model-opt-yolo-eval-trt) |

---

## Autotune vs PTQ

- **Autotune** searches **where** to insert Q/DQ nodes using **TensorRT** timing. It does **not** replace full calibration.
- **Quantize** runs Model Optimizer **PTQ** with your `calib.npy` and produces the quantized ONNX.

Typical order: **download-coco** → **calib** → **autotune (optional)** → **quantize** (using `optimized_final.onnx` if you autotuned) → **engine** → **eval**.

---

## Preprocessing alignment

Calibration preprocessing (`calib`) must match how the ONNX was exported: **input size**, **letterbox vs resize**, **RGB vs BGR**, **normalization** (e.g. ÷255). Defaults follow common Ultralytics-style exports (RGB, NCHW, letterbox).

---

## Further reading

- [Model Optimizer ONNX PTQ example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)
- [Artifacts & logging](artifacts-and-logging.md) for output paths and log files
