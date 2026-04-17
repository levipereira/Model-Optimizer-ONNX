---
name: onnx-eval-io-autodetect
description: >-
  Specifies ONNX/TensorRT I/O handling for modelopt-onnx-ptq eval-trt and pipeline-e2e:
  remove --output-format in favor of automatic detection from the ONNX graph, rename
  --input-name to --input-tensor, optional --output-tensor, onnx-based discovery when
  omitted, supported vs unsupported output shapes, and when to error. Use when the user
  asks to auto-detect eval output layout, drop output-format, rename input-name, inspect
  ONNX inputs/outputs, or align eval-trt with YOLO/RT-DETR export shapes.
---

# ONNX I/O auto-detection for `eval-trt` (and `pipeline-e2e`)

## DeepStream-Yolo `utils/export_*.py` (analysed)

The reference tree **`.garbage_data/source/DeepStream-Yolo/utils/`** (26 exporters) follows one **detection** pattern in almost every **PyTorch** script:

| Convention | Value |
|------------|--------|
| ONNX **input** name | **`input`** (`input_names=["input"]`) |
| ONNX **output** name | **`output`** (`output_names=["output"]`) |
| Output tensor | **One** float tensor **`[batch, N, 6]`** — per row: **xyxy** (in network input / letterbox space for YOLO heads), **score**, **class id** |

**`DeepStreamOutput`** (typical YOLO head): `transpose` raw **`[B, 4+nc, A]`** → **`[B, A, 4+nc]`**, take **`[:,:,:4]`** boxes, **`torch.max`** over class logits for score + label → **`[B, A, 6]`**. NMS is **not** inside the ONNX; the DeepStream parser / **`eval_trt`** path **`deepstream_yolo`** applies **per-class NMS** on those candidates.

**Same `[B,N,6]` rank** appears in:

- DETR-style wrappers (**RT-DETR** PyTorch/Ultralytics, **D-FINE**, **RF-DETR**, etc.): boxes are converted to **xyxy** in **pixel** space via a fixed matrix × `img_size` before concat — still **one** tensor named **`output`**, so **name-based** auto-detect maps to **`deepstream_yolo`** decoding. Coordinate space may differ from plain YOLO; treat accuracy as export-dependent.

**Exceptions:**

- **`export_ppyoloe.py` (Paddle)** — input tensor name **`image`**, not `input`; still ends with a single detection tensor after **`DeepStreamOutput`** (dict-based head inside Paddle).
- **`export_yoloV9.py`** — dual-head models use **`DeepStreamOutputDual`** (still one **`output`**).
- **`export_rtdetr_pytorch.py` / `export_dfine.py`** — inner model returns a **dict**; the wrapper still exports **`input`** / **`output`** with **`[B,Q,6]`**.

So for **bbox ONNX from these scripts**, auto-detection should treat **`output` + rank-3 + last dim 6** as **DeepStream-Yolo–compatible** (`deepstream_yolo` in **`eval-trt`**).

## Implemented in repo

- **`modelopt_onnx_ptq/onnx_eval_layout.py`** — `infer_eval_output_format_from_onnx`, `infer_eval_output_format_from_trt_outputs`, `infer_default_input_tensor_name_from_onnx`.
- **`eval-trt`**: **`--output-format auto`** (optional **`--onnx PATH`**; if omitted, uses engine bindings). Maps **`output`** / large **N** / **`output0`** as documented below.
- **`pipeline-e2e`**: **`--output-format auto`** passes **`--onnx`** (original ONNX for FP16 baseline; quantized ONNX for PTQ rows).

## Scope

- **`eval-trt`** is where output layout matters for **COCO mAP** (decoding + NMS semantics).
- **`pipeline-e2e`** forwards the same knobs to **`eval-trt`** for each engine row; it does not implement separate decoding logic.

## CLI changes (target behavior)

1. **`--output-format auto`** (implemented) infers **`ultralytics`** or **`deepstream_yolo`** only. Four-tensor **`num_dets`/det_*** graphs are **not supported** by **`eval-trt`** at all — re-export with a single **`[B,N,6]`** tensor.
2. **Rename `--input-name` → `--input-tensor`** (Python `dest` may be `input_tensor`). Used for **TensorRT profile shapes** (`build-trt`, `trex-analyze`, and any step that passes `--minShapes` / `--optShapes` / `--maxShapes`). Keep naming parallel to **`--output-tensor`**.
3. **Keep `--output-tensor OUTPUT_TENSOR`** as an optional override when the graph has **multiple** detection-related outputs or names are ambiguous.
4. If **`--input-tensor`** or **`--output-tensor`** is **omitted**, load the **ONNX** and **infer** names (and ranks) from the graph. Prefer **`onnx`** (already a project dependency) — see **Detection procedure** below.

## Detection procedure (ONNX)

Use the **ONNX** file paired with the engine (original or quantized ONNX used for `trtexec` / `build-trt`).

Suggested inspection pattern (agents may wrap this in a small helper inside the package):

```python
import onnx
from onnx import shape_inference

path = "models/your.onnx"
model = onnx.load(path)
# Optional but recommended when value_info is sparse:
try:
    model = shape_inference.infer_shapes(model)
except Exception:
    pass

for inp in model.graph.input:
    # Skip weights bundled as inputs if your exporter does that
    print("INPUT", inp.name, _dims(inp))

for out in model.graph.output:
    print("OUTPUT", out.name, _dims(out))
```

Where `_dims` reads `TensorShapeProto` dimensions (`dim_value` / `dim_param`). A one-liner for quick checks is also fine:

```bash
python -c "import onnx; m=onnx.load('models/your.onnx'); \
  [print('in', i.name) for i in m.graph.input]; \
  [print('out', o.name) for o in m.graph.output]"
```

**Input tensor:** if there is a **single** graph input, use its name as the default **`input_tensor`**. If there are multiple inputs, require **`--input-tensor`** or apply a documented heuristic (e.g. exclude obvious constant/initializer-only inputs — only if implemented explicitly).

**Output tensor / layout:**

- Four-tensor **`num_dets`/det_*** graphs: **unsupported** (error from **`onnx_eval_layout`** / **`eval-trt`**).
- If there is a **single** primary detection output with shape **`(B, N, 6)`**, **`postprocess_single_tensor_xyxy`** applies. **Heuristic (implemented):** output name **`output`** → **`deepstream_yolo`** (DeepStream-Yolo utils); **`output0`** → **`ultralytics`**; else if middle dimension **≥ 512** (static) → **`deepstream_yolo`**, else **`ultralytics`**. If ambiguous, user can still pass explicit **`--output-format`**.

If **`--output-tensor`** is omitted:

- **Single-tensor layouts:** pick the tensor whose shape matches a **supported** pattern (below); if multiple outputs exist (e.g. segmentation), see **Segmentation**.

## Supported output shapes (detection / eval)

When auto-detecting, **accept** and map to existing decoders:

| Layout | Typical shape | Decoder notes |
|--------|----------------|----------------|
| **Single tensor post-NMS / e2e** | **`(B, max_det, 6)`** e.g. `(1, 300, 6)` | `x1,y1,x2,y2,score,class` in letterbox space; **no** extra NMS in eval (`apply_nms=False`). |
| **Single tensor DeepStream-style** | **`(B, num_anchors, 6)`** (same rank as above) | Same six fields but **per-class NMS in eval** (`apply_nms=True`). |

**RT-DETR raw head:** **`(B, num_queries, 4 + nc)`** without a DeepStream wrapper is **not** the same as **`[B,N,6]`** — **unsupported** unless a decoder exists. DeepStream-wrapped RT-DETR exports use **`[B,Q,6]`** like other scripts.

**Raw YOLO head (no NMS in graph):** **`(B, 4 + nc, A)`** — requires **different** postprocess (decode anchors + optional NMS). If not implemented, **unsupported** with an explicit error.

## Unsupported / error behavior

- **Segmentation** exports often have **two** outputs, e.g. **`(B, 4+nc+nm, A)`** + **`(B, nm, H', W')`**. This workflow targets **bbox COCO mAP**; if the ONNX clearly looks like segmentation (mask protos + coefficients), **exit with a short message** that the shape/layout is **not supported** by **`eval-trt`**.
- Any output whose **rank/shape** does not match the **supported** table (after optional shape inference) → **fail** with: **which tensor**, **observed shape**, and **what is supported** (link or bullet list).

## Agent checklist when implementing

- [x] **`--output-format auto`** on **`eval-trt`** + **`pipeline-e2e`** with ONNX/engine inference (**`onnx_eval_layout`**).
- [ ] Rename **`--input-name`** → **`--input-tensor`**; optional default from ONNX (**`infer_default_input_tensor_name_from_onnx`**).
- [ ] Optional: default **`--output-format`** to **`auto`** and treat explicit enums as legacy.
- [ ] Update **`docs/cli-reference.md`** / **README** for **`auto`** and DeepStream-Yolo alignment.

## Related code (current)

- Decoding: `modelopt_onnx_ptq/eval_trt.py` — `postprocess_single_tensor_xyxy`, `run_eval`.
- Pipeline forwarding: `modelopt_onnx_ptq/pipeline_e2e.py`.
