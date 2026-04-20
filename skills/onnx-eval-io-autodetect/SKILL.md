---
name: onnx-eval-io-autodetect
description: >-
  eval-trt and pipeline-e2e ONNX/TensorRT I/O: --output-format auto, ultralytics_e2e
  vs ultralytics_raw vs deepstream_yolo, --output-tensor override, --input-name for
  build-trt shapes, unsupported four-tensor outputs, and planned --input-tensor rename.
  Use when the user asks about eval layout, COCO mAP decoding, DeepStream-Yolo exports,
  or aligning engines with ONNX graphs.
---

# ONNX I/O for `eval-trt` and `pipeline-e2e`

## Current behavior (this repository)

- **`eval-trt`** requires **`--output-format`** with choices including **`auto`**. With **`auto`**, pass **`--onnx`** (recommended) so **`modelopt_onnx_ptq/onnx_eval_layout.py`** classifies the graph; if **`--onnx`** is omitted, inference uses **TensorRT engine** output bindings when possible.
- **`auto`** resolves to one of:
  - **`ultralytics_e2e`** — single **`[B, N, 6]`** tensor, NMS **in** the graph (typical name **`output0`**; alias **`ultralytics`** → **`ultralytics_e2e`**).
  - **`ultralytics_raw`** — **`[B, 4+nc, num_anchors]`** (NMS **in eval**).
  - **`deepstream_yolo`** — **`[B, num_anchors, 6]`** DeepStream-style (typical name **`output`**), per-class NMS in eval.
- **`--output-tensor`** overrides which bound output to treat as the detection tensor when names are ambiguous.
- **`pipeline-e2e`** defaults **`--output-format`** to **`auto`** and passes **`--onnx`** through to **`eval-trt`** (original ONNX for FP16 baseline; quantized ONNX for PTQ rows).
- **`--input-name`** (on **`pipeline-e2e`**) sets the ONNX input name for **build-trt** shape profiles. If omitted, **build-trt** infers a single-input graph; else a documented default (see **`pipeline-e2e --help`**).
- **Four-tensor** graphs (**`num_dets`**, **`det_*`**, …) are **unsupported** for bbox mAP — re-export to a **single-tensor** detection layout or use a different eval path.

## DeepStream-Yolo `utils/export_*.py` (reference)

The reference tree **`.garbage_data/source/DeepStream-Yolo/utils/`** (many exporters) follows one **detection** pattern in most **PyTorch** scripts:

| Convention | Value |
|------------|--------|
| ONNX **input** name | Often **`input`** (`input_names=["input"]`) |
| ONNX **output** name | Often **`output`** |
| Output tensor | One float **`[batch, N, 6]`** — **xyxy**, **score**, **class** |

**Exceptions:** PP-YOLOE uses input **`image`**; some models use **`DeepStreamOutputDual`**; DETR-style wrappers still often end as **`input`** / **`output`** with **`[B, Q, 6]`**.

## Roadmap (not necessarily implemented)

| Item | Status |
|------|--------|
| **`--output-format auto`** + ONNX/engine inference | **Done** (`onnx_eval_layout.py`) |
| Rename **`--input-name`** → **`--input-tensor`** (parallel to **`--output-tensor`**) | Planned |
| Default **`--output-format`** to **`auto`** everywhere | **`pipeline-e2e`** defaults to **`auto`**; **`eval-trt`** still requires an explicit choice (can be **`auto`**) |

## Supported detection shapes (summary)

| Layout | Typical shape | Notes |
|--------|----------------|--------|
| E2E / Ultralytics post-NMS | **`(B, N, 6)`** | NMS in graph; **`ultralytics_e2e`**. |
| DeepStream single tensor | **`(B, A, 6)`** | NMS in eval; **`deepstream_yolo`**. |
| Ultralytics raw head | **`(B, 4+nc, anchors)`** | Decode + NMS in eval; **`ultralytics_raw`**. |

**Segmentation** or **raw heads** without a matching decoder → **unsupported** for this COCO bbox workflow; fail with a clear message (see code).

## Agent checklist (maintenance)

- [x] **`--output-format auto`** on **`eval-trt`** + **`pipeline-e2e`** with **`onnx_eval_layout`**.
- [ ] Rename **`--input-name`** → **`--input-tensor`** where profile shapes are set; optional ONNX default (**`infer_default_input_tensor_name_from_onnx`** exists).
- [ ] Keep **`docs/cli-reference.md`** / **README** aligned when flags change.

## Related code

- **`modelopt_onnx_ptq/onnx_eval_layout.py`** — `infer_eval_output_format_from_onnx`, `infer_eval_output_format_from_trt_outputs`, helpers.
- **`modelopt_onnx_ptq/eval_trt.py`** — decoders, **`--output-format`**, **`--onnx`**, **`--output-tensor`**.
- **`modelopt_onnx_ptq/pipeline_e2e.py`** — forwards eval and build flags.
