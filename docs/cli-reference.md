# CLI reference — `model-opt-yolo`

After `pip install -e .`, the entry point **`model-opt-yolo`** dispatches subcommands:

```bash
model-opt-yolo --help
```

| Subcommand | Purpose |
|------------|---------|
| `download-coco` | Download COCO val2017 images + annotations (`instances_val2017.json`) |
| `calib` | Build a NumPy calibration tensor (`.npy`) from a folder of images |
| `quantize` | Wrapper around `python -m modelopt.onnx.quantization` (PTQ, optional `--autotune`) |
| `build-trt` | Run `trtexec` to build a `.engine` from ONNX (`--mode`: `strongly-typed`, `best`, `fp16`, `fp16-int8`) |
| `trt-bench` | `trtexec` throughput/latency on an **existing** `.engine` (`--loadEngine`; logs under `artifacts/trt_engine/logs/`) |
| `eval-trt` | COCO mAP on TensorRT engines — **`--output-format`** chooses `onnx_trt` (four tensors), Ultralytics single-tensor, or DeepStream-Yolo |
| `report-runs` | Aggregate `trt-bench` / `eval-trt` logs into a Markdown report |
| `pipeline-e2e` | Full run: calib → FP16 baseline → quantize → build-trt → eval-trt → trt-bench → report (optional `--autotune`, `--quant-matrix`) |
| `trex-analyze` | `trtexec` build + profile → layer/timing JSON, optional TREx plan graph (SVG/PNG); optional **`--compare-onnx`** for two different ONNX files + layer CSV (needs **`trex`** / Docker TREx) |

---

## `model-opt-yolo download-coco`

Fetches **COCO val2017** images and **2017 train/val annotations** (so you get `instances_val2017.json` for mAP). Default output layout matches the rest of this repo: `data/coco/val2017/`, `data/coco/annotations/`.

| Argument | Description |
|----------|-------------|
| `--output-dir` | Root directory (default: `data/coco`) |
| `--log-file`, `-v` | Logging |

Skips a download if the target folder/file already exists (same behavior as the former shell script). Uses **`wget -c`** when `wget` is on `PATH`; otherwise streams via **urllib** with a progress bar.

---

## `model-opt-yolo calib`

Builds `calib.npy` for Model Optimizer calibration.

**Common arguments**

| Argument | Description |
|----------|-------------|
| `--images_dir` | Directory of images (e.g. COCO `val2017`) |
| `--calibration_data_size` | Number of images (≥500 recommended for CNN-style models) |
| `--img_size` | Square side length (must match model export) |
| `--no-letterbox` | Disable letterbox; resize only |
| `--bgr` | Keep BGR order (default: RGB) |
| `--fp16` | Save tensor as float16 |
| `--output_path` | Override output path (default: under `artifacts/calibration/` with session name) |
| `--log-file`, `-v` | Logging (see [Artifacts](artifacts-and-logging.md)) |

---

## `model-opt-yolo quantize`

Runs ONNX PTQ via Model Optimizer.

**Common arguments**

| Argument | Description |
|----------|-------------|
| `--calibration_data` | Path to `calib.npy` (**required**) |
| `--onnx_path` | Single ONNX file |
| `--onnx_glob` | Glob (e.g. `models/*.onnx`) — mutually exclusive with `--onnx_path` |
| `--output_dir` | Output directory (default: `<artifacts root>/quantized`; root is `cwd/artifacts` or `MODELOPT_ARTIFACTS_ROOT`) |
| `--quantize_mode` | `fp8`, `int8`, `int4` |
| `--calibration_method` | e.g. `entropy`, `max` (mode-dependent) |
| `--high_precision_dtype` | Default **`fp32`** in this project (avoids many YOLO `infer_shapes` issues); `fp16` optional |
| `--autotune` | Q/DQ placement tuning via TensorRT timing (`quick` \| `default` \| `extensive`). Use with **int8**; **fp8** + autotune often fails on YOLO-style graphs (operator coverage). **int4** ignores autotune. Needs GPU + TensorRT. |
| `--suffix` | Output suffix (default `.quant.onnx`) |

**FP8 hardware:** `--quantize_mode fp8` requires a **CUDA GPU with compute capability ≥ 8.9** (FP8 tensor cores). Examples: **Ada Lovelace** (RTX 4090, 4080, 4070, …) — CC 8.9; **Hopper** (H100, H200) — CC 9.0; **Blackwell** (B200, RTX 5090, 5080, …) — CC 10.0+.

### Pass-through (`--` …)

Everything after a lone `--` is appended to the same command as `python -m modelopt.onnx.quantization`. The wrapper **always** passes these flags (do **not** repeat them after `--`, or you will duplicate arguments):

| Already set by `model-opt-yolo quantize` | Source in this CLI |
|------------------------------------------|--------------------|
| `--onnx_path` | Per input file |
| `--quantize_mode` | `--quantize_mode` |
| `--calibration_data_path` | `--calibration_data` |
| `--calibration_method` | `--calibration_method` |
| `--output_path` | Derived from `--output_dir`, stem, mode, method, `--suffix` |
| `--high_precision_dtype` | `--high_precision_dtype` |

Example:

```bash
# Quantize with integrated autotune
model-opt-yolo quantize --calibration_data ... --onnx_path ... --autotune default

# Pass extra flags through to modelopt
model-opt-yolo quantize --calibration_data ... --onnx_path ... -- --calibrate_per_node --simplify
```

For the authoritative list on your install, run:

```bash
python -m modelopt.onnx.quantization --help
```

The following are **additional** `modelopt.onnx.quantization` options you can pass through (grouped for readability). Names and behavior match NVIDIA Model Optimizer; minor differences may appear across versions.

#### Calibration and shapes

| Flag | Description |
|------|-------------|
| `--trust_calibration_data` | Trust calibration data (allows pickle deserialization where applicable). |
| `--calibration_cache_path` | Pre-computed activation scaling factors (calibration cache). |
| `--calibration_shapes` | Static shapes for calibration if inputs have non-batch dynamic dims (e.g. `input0:1x3x256x256,input1:1x3x128x128`). |
| `--calibration_eps` | Execution provider order for calibration (`trt`, `cuda:x`, `dml:x`, `cpu`, …). |
| `--override_shapes` | Override model inputs to static shapes (same shape-spec style as above). |

#### Scope: which ops / nodes to quantize

| Flag | Description |
|------|-------------|
| `--op_types_to_quantize` | Space-separated ONNX op types to quantize. |
| `--op_types_to_exclude` | Space-separated op types to exclude from quantization. |
| `--op_types_to_exclude_fp16` | Op types to exclude from FP16/BF16 conversion (when `--high_precision_dtype` is fp16/bf16). |
| `--nodes_to_quantize` | Node names to quantize (regex supported). |
| `--nodes_to_exclude` | Node names to exclude (regex supported). |

#### I/O, plugins, logging, files

| Flag | Description |
|------|-------------|
| `--use_external_data_format` | Write large weights to `.onnx_data` when needed. |
| `--keep_intermediate_files` | Keep intermediate files from conversion/calibration. |
| `--log_level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` (case variants accepted). |
| `--log_file` | Log file for the quantization process (separate from `model-opt-yolo`’s `--log-file`). |
| `--trt_plugins` | Paths to custom TensorRT `.so` plugins (enables TensorRT EP). |
| `--trt_plugins_precision` | Per custom-op precision spec (`op:fp16`, or detailed in/out lists — see `--help`). |

#### Precision and graph behavior

| Flag | Description |
|------|-------------|
| `--mha_accumulation_dtype` | MHA accumulation dtype when relevant (e.g. with `fp8`). |
| `--disable_mha_qdq` | Do not insert Q/DQ on MatMuls in MHA patterns. |
| `--dq_only` | Weight-only style: DQ nodes with quantized weights, Q nodes removed. |
| `--use_zero_point` | Zero-point quantization (e.g. awq_lite). |
| `--passes` | Extra optimization passes (e.g. `concat_elimination`). |
| `--simplify` | Simplify ONNX before quantization. |
| `--calibrate_per_node` | Calibrate per node (lower memory on large models). |
| `--direct_io_types` | Lower I/O dtypes in the quantized model where supported. |
| `--opset` | Target ONNX opset for the quantized model. |

#### Built-in autotune (inside PTQ module)

When `--autotune` is set on upstream, it tunes Q/DQ placement for TensorRT. These flags are only meaningful together with `--autotune`:

| Flag | Description |
|------|-------------|
| `--autotune` | Optional preset: `quick`, `default`, or `extensive`. |
| `--autotune_output_dir` | Directory for autotune artifacts (state, logs). |
| `--autotune_schemes_per_region` | Q/DQ schemes to try per region. |
| `--autotune_pattern_cache` | YAML pattern cache for warm-start. |
| `--autotune_qdq_baseline` | Pre-quantized ONNX to import Q/DQ patterns. |
| `--autotune_state_file` | Resume/crash-recovery state file (default under output dir). |
| `--autotune_node_filter_list` | File with wildcard patterns; regions without matches are skipped. |
| `--autotune_verbose` | Verbose autotuner logging. |
| `--autotune_use_trtexec` | Benchmark with `trtexec` instead of TensorRT Python API. |
| `--autotune_timing_cache` | TensorRT timing cache path. |
| `--autotune_warmup_runs` | Warmup iterations before timing. |
| `--autotune_timing_runs` | Timed runs for latency. |
| `--autotune_trtexec_args` | Extra `trtexec` arguments as one quoted string. |

> **Note:** The `--autotune*` flags above are the **in-module** PTQ autotune integrated into `python -m modelopt.onnx.quantization`, passed through by `model-opt-yolo quantize --autotune`.

---

## `model-opt-yolo build-trt`

Runs **`trtexec`** to compile an ONNX file into a TensorRT **`.engine`**. Requires `trtexec` on `PATH` (TensorRT / NGC container).

**Common arguments**

| Argument | Description |
|----------|-------------|
| `--onnx` | Input `.onnx` path (**required**) |
| `--img-size` | Square `H=W` for shape profile (default `640`) |
| `--batch` | Batch size for shapes (see modes below; default `1`) |
| `--input-name` | Input tensor name for `--minShapes` / `--optShapes` / `--maxShapes` (default `images`) |
| `--mode` | `strongly-typed` (default), `best`, `fp16`, or `fp16-int8` |
| `--engine-out` | Output `.engine` path (default: `<artifacts>/trt_engine/<onnx-stem>.engine`) |
| `--timing-cache` | Timing cache file (default: `<engine>.timing.cache`) |
| `--session-id` | Optional id: default logs go under `artifacts/pipeline_e2e/sessions/<id>/trt_engine/logs/` (for `report-runs`; ignored if `--log-file` is set). If omitted, **`SESSION_ID`** is used when set. |
| `--log-file`, `-v` | Logging (default log under `<artifacts>/trt_engine/logs/build_trt_*.log`, or session dir with `--session-id`) |

### Modes

All modes share the same dynamic-shape profile (**`minShapes` / `optShapes` / `maxShapes`**: **batch × 3 × H × W**). Throughput on an already-built plan is measured with **`trt-bench`**, not `build-trt`.

**`strongly-typed`** (default) — `trtexec` **`--stronglyTyped`**

Builds a **strongly typed** network: TensorRT follows the ONNX graph’s tensor types and Q/DQ layout **without** the same cross-layer precision exploration as **`--best`**. For **Model Optimizer PTQ** artifacts (INT8/FP8/INT4 Q/DQ graphs), this is the **recommended default** so the engine respects the quantized types.

**`best`** — `trtexec` **`--best`**

Lets TensorRT consider multiple precisions and tactics to minimize latency. Often a good choice for **non-quantized** FP ONNX when you care about **speed**; for PTQ ONNX, prefer **`strongly-typed`** unless you have a reason to experiment with **`--best`**.

**`fp16`** — `trtexec` **`--fp16`**

For **FP32 / non-quantized** ONNX (no Q/DQ): enables FP16 where the builder can use it. Not the right knob for interpreting a full INT8 Q/DQ graph—that path is different.

**`fp16-int8`** — `trtexec` **`--fp16`** + **`--int8`**

Classic TensorRT combination for **FP** ONNX when you want both FP16 and INT8 kernels in the search space. For **YOLO-style** models from a **non-quantized** checkpoint, this is often a **good practical choice** (with TensorRT’s INT8 requirements, calibration, and operator support as documented by NVIDIA). It is **not** a drop-in substitute for Model Optimizer PTQ artifacts; those are a separate pipeline.

| `--mode` | Flags | Summary |
|----------|-------|---------|
| **`strongly-typed`** (default) | **`--stronglyTyped`** | Strict ONNX/Q/DQ types; default for PTQ/quantized ONNX. |
| **`best`** | **`--best`** | Broad tactic / precision search; common for non-quantized FP graphs. |
| **`fp16`** | **`--fp16`** | Non-quantized ONNX → FP16 where applicable. |
| **`fp16-int8`** | **`--fp16`** **`--int8`** | FP + INT8 search space; often recommended for YOLO from FP ONNX (see TensorRT docs). |

### Pass-through (`--` …)

Arguments after `--` are appended to the **`trtexec`** command (after the built-in flags). Use this to **add** flags or **override** behavior (later tokens win depending on `trtexec`).

```bash
model-opt-yolo build-trt --onnx model.fp.onnx --img-size 640 --batch 1 -- --verbose
model-opt-yolo build-trt --onnx model.onnx --mode best --img-size 640
model-opt-yolo build-trt --onnx model.onnx --mode fp16-int8 --img-size 640
```

---

## `model-opt-yolo trex-analyze`

End-to-end **TensorRT Engine Explorer (TREx)** workflow for **one ONNX** and a **`build-trt`-style `--mode`**: runs **`trtexec`** twice — **build** (with **`--profilingVerbosity=detailed`**, **`--exportLayerInfo`**, **`--dumpLayerInfo`**) and **profile** (**`--loadEngine`**, **`--exportTimes`**, **`--exportProfile`**, **`--exportLayerInfo`**, CUDA graph + separate profile run, matching the upstream TREx `process_engine` pattern). Writes everything under **`artifacts/trex/runs/<stem>_<mode>_<timestamp>/`** (or **`artifacts/pipeline_e2e/sessions/<id>/trex/…`** with **`--session-id`**).

**Comparison** is for **two different ONNX graphs** (e.g. FP16 export vs PTQ `*.quant.onnx`, or two quantized variants). Pass **`--compare-onnx PATH`**; optional **`--compare-onnx-mode`** sets the second builder mode (defaults to **`--mode`**). Outputs use **`primary/`** and **`compare/`** subfolders plus **`compare_layers__*.csv`**. Comparing the **same** ONNX with two **`trtexec`** modes only is not the intended workflow — use two exports instead.

Requires **`trex`**. On the Docker image, TREx is in **`env_trex`** (not the same Python as **`quantize`** / **`build-trt`**); **`trex-analyze`** first adds **`$TREX_VENV/.../site-packages`** to **`sys.path`**, then **re-executes** with **`$TREX_VENV/bin/python`** only if **trex** is still missing (override **`TREX_VENV`**, disable re-exec with **`MODELOPT_TREX_NO_REEXEC=1`**). Locally, use a **dedicated venv** for **`trt-engine-explorer`** or install **`model-opt-yolo`** into that venv too. **Graphviz** (`dot`) must be on **`PATH`** for **`--graph-format`**.

**Common arguments**

| Argument | Description |
|----------|-------------|
| `--onnx` | Primary `.onnx` (**required**) |
| `--mode` | Same as **`build-trt`** for **`--onnx`**: `strongly-typed`, `best`, `fp16`, `fp16-int8` |
| `--compare-onnx` | Optional second `.onnx` (must differ from **`--onnx`** after path resolution) |
| `--compare-onnx-mode` | Builder mode for **`--compare-onnx`** (default: same as **`--mode`**) |
| `--img-size`, `--batch`, `--input-name` | Shape profile (defaults match **`build-trt`**) |
| `--graph-format` | `svg` (default), `png`, or `pdf` for the TREx **`DotGraph`** plan diagram; **`--no-graph`** skips |
| `--output-dir` | Force run directory (default auto under **`artifacts/trex/runs/`**) |
| `--session-id` | Session-scoped output under **`pipeline_e2e/sessions/<id>/trex/`** |
| `--log-file`, `-v` | Logging (default: **`trex_analyze.log`** inside the run directory) |

### Pass-through (`--` …)

Extra **`trtexec`** tokens after **`--`** are appended to **both** the build and profile commands. **`trex-analyze`** does not set **`--memPoolSize`**; TensorRT defaults apply unless you pass e.g. **`-- --memPoolSize=workspace:8192MiB`**.

```bash
model-opt-yolo trex-analyze --onnx models/yolo.onnx --mode strongly-typed --img-size 640
model-opt-yolo trex-analyze --onnx models/yolo_fp16.onnx --mode fp16 \\
  --compare-onnx artifacts/quantized/yolo.int8.entropy.quant.onnx --compare-onnx-mode strongly-typed --img-size 640
```

---

## `model-opt-yolo trt-bench`

Runs **`trtexec`** with **`--loadEngine`** on an **existing** TensorRT **`.engine`** (no ONNX rebuild). Shapes come from the serialized plan — there is no **`--batch`** / **`--img-size`** here. Follows NVIDIA’s [TensorRT performance best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) (warmup, iterations, duration; plus CUDA graph, spin wait, no transfers for isolated GPU timing).

**Common arguments**

| Argument | Description |
|----------|-------------|
| `--engine` | Input **`.engine`** path (**required**) |
| `--warm-up` | `trtexec --warmUp` in **milliseconds** (default **500**) |
| `--iterations` | `trtexec --iterations`: minimum inference iterations (default **100**) |
| `--duration` | `trtexec --duration` in **seconds** (default **60**) |
| `--session-id` | Optional id: default logs under `artifacts/pipeline_e2e/sessions/<id>/trt_engine/logs/` (for `report-runs`; ignored if `--log-file` is set). If omitted, **`SESSION_ID`** is used when set. |
| `--log-file`, `-v` | Logging (default: `<artifacts>/trt_engine/logs/trt_bench_<engine-stem>_<timestamp>.log`, or session dir with `--session-id`) |

Built-in `trtexec` flags include **`--useCudaGraph`**, **`--useSpinWait`**, **`--noDataTransfers`**.

```bash
model-opt-yolo trt-bench --engine artifacts/trt_engine/model.int8.entropy.quant.engine
model-opt-yolo trt-bench --engine model.engine --warm-up 500 --iterations 100 --duration 60 -- --avgRuns=20
```

---

## `model-opt-yolo eval-trt`

Runs **TensorRT** inference on **COCO val2017** and computes **bbox mAP** with **pycocotools**. Preprocessing uses the same **letterbox** + **÷255** convention as `calib` (aligned with common Ultralytics-style exports).

You must set **`--output-format`** to match how your `.engine` exposes detections. The three modes correspond to common export stacks:

| `--output-format` | References | TensorRT I/O (typical) |
|-------------------|------------|-------------------------|
| **`onnx_trt`** | **[levipereira/ultralytics](https://github.com/levipereira/ultralytics)** — `format=onnx_trt` / `onnx_trt.py`; four fixed detection outputs (see upstream README). The graph is **not** defined solely by “EfficientNMS”: e‑to‑e heads omit the TRT NMS plugin; YOLOv8-style paths may insert EfficientNMS_TRT — **`eval-trt`** always decodes the same four tensor names. | **Input:** `[B, 3, H, W]`. **Outputs:** `num_dets` `[B, 1]`, `det_boxes` `[B, K, 4]`, `det_scores` `[B, K]`, `det_classes` `[B, K]`. Boxes **xyxy** in **letterboxed** input pixel space. |
| **`efficient_nms`** | *Alias* for **`onnx_trt`** (same behavior; legacy flag name). | Same as **`onnx_trt`**. |
| **`ultralytics`** | **[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)** — export to ONNX/TensorRT with NMS in the graph. | **Single** output tensor (e.g. `output0`) `[B, N, 6]` (e.g. `N = 300`). Rows: **xyxy, score, class**; NMS already applied. |
| **`deepstream_yolo`** | **[marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)** — same layout as the **`nvdsparsebbox_Yolo`** custom parser (xyxy + score + class per anchor). | **Single** output (e.g. `output`) `[B, num_anchors, 6]` (e.g. **8400** anchors at 640×640). **Pre-clustering**; this tool applies **per-class NMS** (`--iou-thres`) then rescales boxes. |

**How evaluation works (all modes):** (1) load image, letterbox to engine `H×W`, run inference with **batch 1**; (2) decode tensors according to `--output-format`; (3) map boxes from letterboxed coordinates to **original image** size; (4) map training **class indices** to **COCO category IDs** (80-class COCO mapping); (5) write predictions JSON and run **COCOeval**.

**Dynamic batch:** `B` may be dynamic in the profile; this command always uses **`B = 1`** per image.

| Argument | Description |
|----------|-------------|
| `--engine` | Path to `.engine` (**required**) |
| `--output-format` | `onnx_trt` \| `efficient_nms` (alias) \| `ultralytics` \| `deepstream_yolo` (**required**) |
| `--output-tensor` | Output tensor name for `ultralytics` / `deepstream_yolo` if the engine has **multiple** outputs and none of the default names (`output0`, `output`, `output1`) is correct |
| `--iou-thres` | IoU threshold for **per-class NMS** in `deepstream_yolo` only (default `0.45`) |
| `--images` | COCO images directory (default `data/coco/val2017`) |
| `--annotations` | `instances_val2017.json` |
| `--img-size` | Hint (overridden by engine input shape if different) |
| `--conf-thres` | Confidence threshold (default `0.001`) |
| `--save-json` | Predictions JSON path (default under `artifacts/predictions/`) |
| `--session-id` | Optional id: default logs under `artifacts/pipeline_e2e/sessions/<id>/predictions/logs/` (for `report-runs`; ignored if `--log-file` is set). If omitted, **`SESSION_ID`** is used when set. |
| `--log-file`, `-v` | Logging |

Examples:

```bash
# Four-tensor layout (levipereira/ultralytics onnx_trt)
model-opt-yolo eval-trt --output-format onnx_trt --engine model.engine \
  --images data/coco/val2017 --annotations data/coco/annotations/instances_val2017.json

# Ultralytics single tensor (explicit name if needed)
model-opt-yolo eval-trt --output-format ultralytics --output-tensor output0 --engine model.engine \
  --images data/coco/val2017 --annotations data/coco/annotations/instances_val2017.json

# DeepStream-Yolo layout (NMS in eval)
model-opt-yolo eval-trt --output-format deepstream_yolo --output-tensor output --engine model.engine \
  --images data/coco/val2017 --annotations data/coco/annotations/instances_val2017.json
```

---

## `model-opt-yolo pipeline-e2e`

Orchestrates **calib** → **FP16 baseline** on the original ONNX (`build-trt --mode fp16` → **eval-trt** → **trt-bench**) → **quantize** → **build-trt** → **eval-trt** → **trt-bench** (per combo) → **report-runs** under a session id (logs under `artifacts/pipeline_e2e/sessions/<id>/`). The FP16 run uses engine stem `<onnx-stem>.fp16` so the report can compare **FP16 TensorRT** vs **quantized** engines.

| Argument | Notes |
|----------|--------|
| `--onnx` | Input FP32 ONNX (**required**) |
| `--session-id` | Session directory name under `artifacts/pipeline_e2e/sessions/`. Default: **`SESSION_ID`** env var if set, else timestamp. CLI overrides **`SESSION_ID`**. |
| `--quant-matrix` | **SPEC** string: default **`int8.entropy`**. Keyword **`all`** = full 6-combo grid. **`mode.all`** = both methods for `int8`, `fp8`, or `int4`. **`mode.method`** = one run. Comma-separated = union (e.g. `int8.all,fp8.entropy`). Details: [Workflow](workflow.md). **FP8** combos need a GPU with **CC ≥ 8.9** (see **FP8 hardware** under [`quantize`](#model-opt-yolo-quantize)). |
| `--autotune` | Same presets as `quantize`. **int8** steps receive `--autotune` when set; **fp8** steps always run **standard PTQ** (no `--autotune` passed). **`int4`** receives the flag but Model Optimizer **ignores** integrated autotune for int4. |
| `--continue-on-error` | Continue after a failed combo |
| `--no-fp16-baseline` | Skip the FP16 baseline on the original ONNX (only run PTQ combos) |
| `--no-report` | Skip final Markdown report |

See `model-opt-yolo pipeline-e2e --help` for image paths, `--input-name`, bench duration, etc.

---

## `model-opt-yolo report-runs`

Scans log directories and writes a Markdown report with tables plus **PNG** bar charts (throughput and mAP) next to the `.md` file (`<stem>_throughput_qps.png`, `<stem>_map5095.png`). Used standalone or at the end of `pipeline-e2e`.

The report includes an **Environment & versions** section (current machine): **model-opt-yolo** and **NVIDIA Model Optimizer** pip versions, **TensorRT** (Python bindings and trtexec log line when present), **PyTorch** / **PyTorch CUDA**, **CUDA** as reported by **`nvidia-smi`**, and per-GPU **name**, **driver**, **memory**, **compute capability**, and **SM count** when the driver exposes it.

| Argument | Description |
|----------|-------------|
| `--session-id` | Shortcut: set `--trt-logs-dir` and `--eval-logs-dir` to `artifacts/pipeline_e2e/sessions/<id>/trt_engine/logs` and `…/predictions/logs` (unless you override them). With `-o` omitted, writes `artifacts/pipeline_e2e/sessions/<id>/e2e_report.md`. Same layout as `pipeline-e2e`. If omitted, **`SESSION_ID`** in the environment is used. **Use this** (or explicit session log paths) so the report sees `pipeline-e2e` outputs — the default without `--session-id` is the **global** flat `artifacts/trt_engine/logs`, not the session folder. |
| `--merge-global-logs` | Also read global `<artifacts>/trt_engine/logs` and `<artifacts>/predictions/logs` and merge with the primary dirs (newest timestamp per config). `pipeline-e2e` enables this when invoking `report-runs`. |
| `--trt-logs-dir` | Folder with `trt_bench_*.log` (default without `--session-id`: `<artifacts>/trt_engine/logs` — often **not** where `pipeline-e2e` writes) |
| `--eval-logs-dir` | Folder with `eval_*.log` (default without `--session-id`: `<artifacts>/predictions/logs`) |
| `-o`, `--output` | Output `.md` path (default: `artifacts/reports/trt_eval_report_<timestamp>.md`, or `…/sessions/<id>/e2e_report.md` when **`--session-id`** or **`SESSION_ID`** selects a session) |

---

## Environment variables (logging)

- `MODELOPT_YOLO_LOGLEVEL` or `LOGLEVEL` — `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `MODELOPT_ARTIFACTS_ROOT` — root for artifacts (default: `<cwd>/artifacts`, created if missing)
- `SESSION_ID` — default session id for `pipeline-e2e`, `build-trt`, `eval-trt`, `trt-bench`, and `report-runs` when `--session-id` is not passed (CLI wins over the variable)
