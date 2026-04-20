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

**ONNX I/O / eval layout:** **`--output-format auto`** (with **`--onnx`**) is implemented; optional **`--output-tensor`** override; planned rename **`--input-name`** → **`--input-tensor`** on pipeline/build steps — see **[onnx-eval-io-autodetect](../onnx-eval-io-autodetect/SKILL.md)**.

## Goal

Produce **comparable** numbers: same calibration tensor shapes, same **`build-trt`** profile (**`--input-name`** / img-size / batch on **`pipeline-e2e`**), same **`eval-trt`** dataset, and **matching detection decode**. Use **`--output-format auto`** plus **`--onnx`** so layout matches the graph (or pass explicit **`ultralytics_e2e`**, **`ultralytics_raw`**, **`deepstream_yolo`**). Then compare **mAP** (**eval-trt**) and **QPS / latency** (**trt-bench**).

## Prerequisites

- ONNX under `models/` (or pass absolute path).
- COCO **val2017** images + `instances_val2017.json` (or adjust **`--images-dir`** / **`--annotations`**).
- **`pipeline-e2e`**: set **`--input-name`** when the ONNX input name is not the single-input default (else **build-trt** infers it; default fallback name **`images`** if ambiguous — see **`pipeline-e2e --help`**).
- **`eval-trt`**: pass **`--output-format auto`** and **`--onnx`** (recommended), or a fixed format; optional **`--output-tensor`** if multiple outputs — see **onnx-eval-io-autodetect**.
- **FP8** PTQ needs a GPU with **compute capability ≥ 8.9** (e.g. RTX 4090).

## Path A — Full mode/method grid (`pipeline-e2e`)

Use **`modelopt-onnx-ptq pipeline-e2e`** when the user wants to compare **int8 / fp8 / int4** and calibration methods without hand-writing six commands.

1. Choose **`--quant-matrix`**:
   - **`all`** → six runs: `int8.{entropy,max}`, `fp8.{entropy,max}`, `int4.{awq_clip,rtn_dq}`.
   - **`int8.all`** → two int8 runs; combine with commas as needed (see `docs/workflow.md`).
2. Optional: **`--quantize-profile <name>`** — same YAML profile for **every** quantize step (e.g. a YOLO26n backbone whitelist under `modelopt_onnx_ptq/profiles/`).
3. **`--high-precision-dtype`** defaults to **fp16** in **`quantize`** / **`pipeline-e2e`**; use **fp32** only if PTQ shape inference fails. Omit **`--autotune`** if the user wants profile-only tuning.
4. Set **`--input-name`** when needed for **build-trt** profile shapes, and **`--build-mode`** (default **`strongly-typed`** for PTQ ONNX). **`pipeline-e2e`** defaults **`--output-format`** to **`auto`** and passes **`--onnx`** into **eval-trt** so **`onnx_eval_layout`** can pick **`ultralytics_e2e`**, **`ultralytics_raw`**, or **`deepstream_yolo`**. Four-tensor **`num_dets`** / **`det_*`** outputs are not supported.
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

When latency regresses or **Reformat** dominates in profiling, pick **one** profiling path (they can be combined across iterations: e.g. CLI for exports, then notebooks for drill-down).

### TREx profiling — three valid methods

**1) Project CLI (automates `trtexec` + JSON + optional graph/compare)**  

```text
modelopt-onnx-ptq trex-analyze …
```

At most **one** of **`--compare`**, **`--graph`**, **`--report`**, or none (layer/timing JSON only). Typical: compare quantized vs FP16 ONNX, or emit a plan graph. Outputs under **`artifacts/trex/runs/…`** (or **`artifacts/pipeline_e2e/sessions/<id>/trex/…`** with **`--session-id`**). Details: **`docs/cli-reference.md`**, **`docs/quantization-performance-workflow.md`**.

**2) TREx virtualenv + Jupyter (`tutorial.ipynb` / `compare_engines.ipynb`)**  

TREx (**`trex`**) lives in a **dedicated venv**, separate from **`modelopt-onnx-ptq`** (avoids **pandas** / **numpy** clashes). Default layout in the Docker image:

```bash
source /workspace/TREx/tools/experimental/trt-engine-explorer/env_trex/bin/activate
# Optional: export TREX_VENV=/workspace/TREx/tools/experimental/trt-engine-explorer/env_trex
cd /workspace/TREx/tools/experimental/trt-engine-explorer/notebooks
jupyter lab   # or: jupyter notebook
```

Upstream notebooks (same paths relative to **`notebooks/`**):

| Notebook | Role |
|----------|------|
| **`tutorial.ipynb`** | **Section 1** — runs **`../utils/process_engine.py`** to **build** the engine from ONNX, **profile** it, and emit **`.graph.json`**, **`.profile.json`**, timing JSON, and an **SVG** plan. Use the **initial `trtexec` parameters** below so results match **`build-trt --mode strongly-typed`** + **`images`** @ 640². |
| **`compare_engines.ipynb`** | Loads **two** **`EnginePlan`** instances (e.g. FP16/FP32 vs INT8 PTQ). Set **`engine_name_1`** / **`engine_name_2`** to **`.engine`** paths whose JSON sidecars sit beside them (typically **`../tests/inputs/`** after **`tutorial.ipynb`**, or copy/symlink engines from **`modelopt-onnx-ptq/artifacts/trt_engine/`** / **`…/pipeline_e2e/sessions/…/trt_engine/`**). |

**Initial `process_engine.py` parameters (YOLO PTQ ONNX — matches repo `build-trt`)**  

Run **from `notebooks/`** so **`../utils/process_engine.py`** resolves. Positional args: **`ONNX`**, **`outdir`**, then the **`trtexec`** passthrough list (**`nargs='*'`** in **`utils/process_engine.py`**). Put **`--`** **after** **`stronglyTyped`** and **before** **`minShapes=…`** (same as **`tutorial.ipynb`**). That **`--`** is the **bypass for `trtexec`**: it tells **`argparse`** to stop treating what follows as **`process_engine.py`** flags so **`minShapes=…`**, **`optShapes=…`**, **`maxShapes=…`** are collected as **`trtexec`** tokens and forwarded into the **`trtexec`** subprocess (the script then turns each token into a **`trtexec`** flag via **`append_trtexec_args`**). The bare **`--`** itself is **not** passed to **`trtexec`**.

```bash
# Example: quantized ONNX from a pipeline_e2e session; outputs under ../tests/inputs/
python3 ../utils/process_engine.py \
  /workspace/modelopt-onnx-ptq/artifacts/pipeline_e2e/sessions/yolo26n_no_e2e/quantized/yolo26n_no_e2e.int8.entropy.quant.onnx \
  ../tests/inputs \
  stronglyTyped -- minShapes=images:1x3x640x640 optShapes=images:1x3x640x640 maxShapes=images:1x3x640x640
```

**Jupyter** cell (matches **`tutorial.ipynb`** — note **`--`** after **`stronglyTyped`**):

```text
!python3 ../utils/process_engine.py /workspace/modelopt-onnx-ptq/artifacts/pipeline_e2e/sessions/yolo26n_no_e2e/quantized/yolo26n_no_e2e.int8.entropy.quant.onnx ../tests/inputs stronglyTyped  -- minShapes=images:1x3x640x640 optShapes=images:1x3x640x640 maxShapes=images:1x3x640x640
```

This produces **`trtexec`** lines of the form **`--stronglyTyped --minShapes=images:1x3x640x640 --optShapes=… --maxShapes=…`** for both **build** and **profile** (as in **`tutorial.ipynb`** output). If your ONNX input tensor is not **`images`**, replace **`images:`** with the actual name (e.g. **`input:1x3x640x640`** for many DeepStream exports).

**3) TREx CLI / scripts without Jupyter**  

With the same venv **activated**, you can run **`process_engine.py`** from a terminal (command above) or other flows under:

```text
/workspace/TREx/tools/experimental/trt-engine-explorer/
```

See upstream **`bin/`**, **`utils/`**, and **README** for CSV/HTML utilities. **`process_engine.py`** is the same entry **`tutorial.ipynb`** section 1 uses.

**Note:** **`trex-analyze`** prepends **`$TREX_VENV`**’s **`site-packages`** so **`import trex`** works from the image Python; if **`trex`** is still missing, it **re-executes** with the venv interpreter — see **`docs/docker-reference.md`** (TREx section).

### Iteration loop

1. **Profile** using **(1)** and/or **(2)** and/or **(3)** above until you know which layers or regions to change.
2. Adjust **`include_nodes`** / **`exclude_nodes`** / **`exclude_op_types`** in a **copy** of a shipped profile under `modelopt_onnx_ptq/profiles/`.
3. Re-run Path B (or a smaller **`--quant-matrix`** slice).

**Autotune:** optional **`quantize --autotune`**; int4 path does not use integrated autotune meaningfully — see `docs/workflow.md`.

## TensorRT build failures on PTQ ONNX

If **`build-trt --mode strongly-typed`** fails with an internal error on a fused layer (e.g. Conv + SiLU), retry **`--mode best`** for that ONNX. Document the mode used when comparing engines.

## Related docs

- `docs/quantization-performance-workflow.md` — iterative loop, profiles, hpfp16 notes.
- `docs/docker-reference.md` — **TREx** (`TREX_VENV`, **`env_trex`**, **`install.sh --venv --full`**).
- `docs/workflow.md` — **`--quant-matrix`** grammar, **`pipeline-e2e`** flow.
- `docs/cli-reference.md` — flags for **`quantize`**, **`build-trt`**, **`eval-trt`**, **`pipeline-e2e`**, **`report-runs`**, **`trex-analyze`**.
