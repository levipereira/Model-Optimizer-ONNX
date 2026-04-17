---
name: ptq-trt-performance
description: >-
  Measures YOLO ONNX PTQ + TensorRT performance (mAP, QPS, latency) and finds a good
  quantization profile or mode/method. Covers pipeline-e2e --quant-matrix, manual
  quantize/build-trt/eval-trt/trt-bench, YAML profiles, report-runs, how to derive
  backbone/neck/head Conv splits from an ONNX for include_nodes regexes, and common pitfalls.
  Use when the user asks to benchmark PTQ, compare int8/fp8/int4, measure throughput,
  find the best profile, optimize latency, regional quantization, or evaluate quantization settings for
  modelopt-onnx-ptq.
---

# PTQ + TensorRT performance (modelopt-onnx-ptq)

**ONNX I/O / eval layout:** For removing **`--output-format`**, renaming **`--input-name`** → **`--input-tensor`**, ONNX defaults for **`input_tensor`** / **`output-tensor`**, and supported shapes, see **[onnx-eval-io-autodetect](onnx-eval-io-autodetect/SKILL.md)**.

## Goal

Produce **comparable** numbers: same calibration tensor shapes, same `build-trt` input tensor name / image size, same `eval-trt` dataset and **matching detection decode** (today: `--output-format`; planned: ONNX-driven auto-detection per **onnx-eval-io-autodetect**), then compare **mAP** (accuracy) and **trt-bench** (throughput / GPU latency).

## Prerequisites

- ONNX under `models/` (or pass absolute path).
- COCO **val2017** images + `instances_val2017.json` (or adjust `--images-dir` / `--annotations`).
- Match **`--input-tensor`** (or legacy **`--input-name`**) to the graph (e.g. Ultralytics often `images`; some exports `input`). If omitted, infer from ONNX (**onnx-eval-io-autodetect**).
- Match **output layout** for **`eval-trt`** — use **`--output-tensor`** when needed; ONNX auto-detection replaces a required **`--output-format`** once implemented (**onnx-eval-io-autodetect**).
- **FP8** PTQ needs a GPU with **compute capability ≥ 8.9** (e.g. RTX 4090).

## Path A — Full mode/method grid (`pipeline-e2e`)

Use **`modelopt-onnx-ptq pipeline-e2e`** when the user wants to compare **int8 / fp8 / int4** and calibration methods without hand-writing six commands.

1. Choose **`--quant-matrix`**:
   - **`all`** → six runs: `int8.{entropy,max}`, `fp8.{entropy,max}`, `int4.{awq_clip,rtn_dq}`.
   - **`int8.all`** → two int8 runs; combine with commas as needed (see `docs/workflow.md`).
2. Optional: **`--quantize-profile <name>`** — same YAML profile for **every** quantize step (e.g. a YOLO26n backbone whitelist under `modelopt_onnx_ptq/profiles/`).
3. **`--high-precision-dtype`** defaults to **fp16** in **`quantize`** / **`pipeline-e2e`**; use **fp32** only if PTQ shape inference fails. Omit **`--autotune`** if the user wants profile-only tuning.
4. Set **`--input-name`** / **`--input-tensor`** (when renamed) to match the graph, **`--build-mode`** (default `strongly-typed` for PTQ ONNX). For **eval**, default **`--output-format auto`** on **`pipeline-e2e`** forwards **`--onnx`** and infers **`ultralytics`** vs **`deepstream_yolo`** (**onnx-eval-io-autodetect**). Four-tensor detection outputs are not supported.
5. Use **`--session-id my_run`** so all logs sit under `artifacts/pipeline_e2e/sessions/<id>/`.
6. If a combo fails, **`--continue-on-error`** keeps the rest.
7. **`--no-fp16-baseline`** skips the extra FP16 engine on the **original** ONNX (saves one full eval if only PTQ rows matter).

**Report:** `pipeline-e2e` ends with **`report-runs`** using **`--merge-global-logs`**, which can mix in **older** bench/eval logs from global `artifacts/`. For a **session-only** table, re-run:

```bash
modelopt-onnx-ptq report-runs --session-id <id> -o artifacts/pipeline_e2e/sessions/<id>/report_session_only.md
```

(omit `--merge-global-logs` — not passed, so only session logs are used.)

**Interpret “best”:** the generated Markdown picks best rows by **mAP**, **QPS**, **mean GPU ms**, and a **combined** √(mAP×QPS/1000) score. Trade-offs are normal: e.g. **int4** may win mAP or lose QPS vs **fp8**; choose by product requirement.

## Path B — Manual A/B (`quantize` → `build-trt` → `eval-trt` → `trt-bench`)

1. **`calib`** → `calib.npy` (fixed `--img_size`, same preprocessing as deployment).
2. **`quantize`** with **`--profile`** / **`--suffix`** so variants do not overwrite (e.g. `.v2.quant.onnx`).
3. **`build-trt --onnx … --mode strongly-typed`** (typical for PTQ ONNX with Q/DQ).
4. **`eval-trt`** on the `.engine` → mAP.
5. **`trt-bench --engine …`** → QPS / latency (run **sequentially** if comparing many engines; parallel load can distort QPS).

Repeat for each profile or flag set.

## Backbone / neck / head splits (`include_nodes` whitelist)

Regional PTQ uses **`include_nodes`** regexes so only chosen **Conv** nodes get Q/DQ. **Boundaries are export-specific** — always derive them from **your** ONNX; do not assume another checkpoint uses the same `node_conv2d_*` numbering.

### 1) Enumerate Conv nodes from the graph

```python
import onnx
m = onnx.load("models/your_model.onnx")
convs = [n.name for n in m.graph.node if n.op_type == "Conv"]
# Typical YOLO ONNX: stem "node_conv2d" + "node_conv2d_1" … "node_conv2d_N"
```

Confirm **count** and **ordering** (e.g. 102 Convs = stem + `_1`…`_101`). If names differ, build patterns from the actual list.

### 2) Map indices to backbone / neck / head (manual + graph)

- Open the ONNX in **Netron** (or similar) and locate **where the backbone ends** (e.g. last stride before FPN/PAN **Resize**/concat region) and **where detection heads** start.
- Record **cut indices** on the `node_conv2d_*` sequence for **that** export. Example split for a **YOLO26n** ONNX in this repo (102 Convs: stem + `_1`…`_101`):
  - **Backbone:** stem + `node_conv2d_1` … `node_conv2d_39`
  - **Neck:** `node_conv2d_40` … `node_conv2d_75` (FPN/PAN-heavy band — approximate by layer index)
  - **Head:** `node_conv2d_76` … `node_conv2d_101`

These indices are **not universal**; re-validate if the ONNX is re-exported or renamed.

### 3) Build and validate regexes

- **Single contiguous range** (e.g. backbone only, or backbone+neck through `_75`): one pattern matching stem + `_1`…`_K` — see the shipped **YOLO26n** “perf” profiles under `modelopt_onnx_ptq/profiles/` (backbone-only and backbone+neck variants).
- **Two disjoint ranges** (e.g. **backbone + head**, skipping neck): **two** `include_nodes` entries (OR of patterns); same repo ships a **backbone+head** variant (backbone `_1`…`_39` + head `_76`…`_101`).
- **All Convs:** one pattern for stem + `_1`…`_101` — **backbone+neck+head** / full Conv whitelist variant in the same folder.

After editing YAML, **validate in Python**: `re.match(pattern, name)` for every Conv name in the list — ensure exactly the intended nodes match (no off-by-one, no duplicate matches).

### 4) Quantize and measure

Use **`quantize --profile <your.yaml>`** (default high precision is **fp16**), unique **`--suffix`**, then Path B (**build-trt** → **eval-trt** → **trt-bench**). Compare mAP and QPS across **backbone-only** vs **backbone+neck** vs **backbone+head** vs **full** Convs.

### 5) If TensorRT fails on some splits

Some graphs fail **`build-trt --mode strongly-typed`** on mixed int8/FP16 boundaries (e.g. Conv+SiLU fusion). Retry **`--mode best`** for that quantized ONNX and note it in comparisons.

**Shipped templates:** `modelopt_onnx_ptq/profiles/` — search for **YOLO26n** performance profiles (`*perf*.yaml`); copy and adjust indices/regexes for other exports.

## Path C — Iterative YAML profile tuning

When latency regresses or **Reformat** dominates in profiling:

1. Run **`trex-analyze`** (and optionally **`--compare`** vs FP16 plan) — see `docs/quantization-performance-workflow.md`.
2. Adjust **`include_nodes`** / **`exclude_nodes`** / **`exclude_op_types`** in a **copy** of a shipped profile under `modelopt_onnx_ptq/profiles/`.
3. Re-run Path B (or a smaller **`--quant-matrix`** slice).

**Autotune:** optional **`quantize --autotune`**; int4 path does not use integrated autotune meaningfully — see `docs/workflow.md`.

## TensorRT build failures on PTQ ONNX

If **`build-trt --mode strongly-typed`** fails with an internal error on a fused layer (e.g. Conv + SiLU), retry **`--mode best`** for that ONNX. Document the mode used when comparing engines.

## Related docs

- `docs/quantization-performance-workflow.md` — iterative loop, profiles, hpfp16 notes.
- `docs/workflow.md` — **`--quant-matrix`** grammar, **`pipeline-e2e`** flow.
- `docs/cli-reference.md` — flags for **`quantize`**, **`build-trt`**, **`eval-trt`**, **`pipeline-e2e`**, **`report-runs`**.
