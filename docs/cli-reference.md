# CLI reference — `model-opt-yolo`

After `pip install -e .`, the entry point **`model-opt-yolo`** dispatches subcommands:

```bash
model-opt-yolo --help
```

| Subcommand | Purpose |
|------------|---------|
| `download-coco` | Download COCO val2017 images + annotations (`instances_val2017.json`) |
| `calib` | Build a NumPy calibration tensor (`.npy`) from a folder of images |
| `quantize` | Wrapper around `python -m modelopt.onnx.quantization` (PTQ) |
| `autotune` | Wrapper around `python -m modelopt.onnx.quantization.autotune` (Q/DQ placement) |
| `build-trt` | Run `trtexec` to build a `.engine` from ONNX (`--mode`: `best`, `strongly-typed`, `fp16`, `fp16-int8`) |
| `trt-bench` | `trtexec` throughput/latency on an **existing** `.engine` (`--loadEngine`; logs under `artifacts/trt_engine/logs/`) |
| `eval-trt` | COCO mAP on TensorRT engines — **`--output-format`** chooses `onnx_trt` (four tensors), Ultralytics single-tensor, or DeepStream-Yolo |

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
| `--suffix` | Output suffix (default `.quant.onnx`) |

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

> **Note:** This repo also provides **`model-opt-yolo autotune`** for the separate `modelopt.onnx.quantization.autotune` CLI. The `--autotune*` flags above are the **in-module** PTQ autotune integrated into `python -m modelopt.onnx.quantization`, which is different from that subcommand.

---

## `model-opt-yolo autotune`

Requires **Model Optimizer from Git** (full autotune CLI). The Docker image installs from GitHub by default.

**Wrapper-only options** (consumed before upstream sees them):

| Option | Description |
|--------|-------------|
| `--imagesize` | `N` or `HxW` — TensorRT profile for **dynamic** spatial dims |
| `--input_name` | ONNX input tensor name for TRT shapes (default `images`) |
| `--log-file`, `-v` | Logging |

If the ONNX input is dynamic and `--imagesize` is omitted, the wrapper exits with an error.

Default output directory pattern: `artifacts/autotune/autotune_<stem>_qt<...>_spr<...>_img<...>_<timestamp>/` when `--output_dir` is not set.

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
| `--mode` | `best` (default), `strongly-typed`, `fp16`, or `fp16-int8` |
| `--engine-out` | Output `.engine` path (default: `<artifacts>/trt_engine/<onnx-stem>.engine`) |
| `--timing-cache` | Timing cache file (default: `<engine>.timing.cache`) |
| `--log-file`, `-v` | Logging (default log under `<artifacts>/trt_engine/logs/build_trt_*.log`) |

### Modes

All modes share the same dynamic-shape profile (**`minShapes` / `optShapes` / `maxShapes`**: **batch × 3 × H × W**). Throughput on an already-built plan is measured with **`trt-bench`**, not `build-trt`.

**`best`** (default) — `trtexec` **`--best`**

Lets TensorRT consider multiple precisions and tactics to minimize latency. This is usually a strong default when you care about **speed** and can outperform a strict graph import because the builder has more freedom to place FP16 (and other) kernels where they help.

**`strongly-typed`** — `trtexec` **`--stronglyTyped`**

Builds a **strongly typed** network: TensorRT follows the ONNX graph’s tensor types and Q/DQ layout **without** the same cross-layer precision exploration as **`--best`**. Think of it as honoring the exported graph “as typed,” not as “apply every global optimization.”

That strictness is **not YOLO-specific**, but **YOLO-style detectors** often suffer here: many ops must remain **FP32** in the exported graph while you still want TensorRT to run hot paths in **FP16** elsewhere. **`best`** typically handles that tradeoff better. **`--mode strongly-typed` is not recommended as the default for YOLO** when **throughput** is the goal—it can leave performance on the table. Use it when you **intentionally** need maximum fidelity to the ONNX types (debugging, compliance, or graphs where strong typing is required).

**`fp16`** — `trtexec` **`--fp16`**

For **FP32 / non-quantized** ONNX (no Q/DQ): enables FP16 where the builder can use it. Not the right knob for interpreting a full INT8 Q/DQ graph—that path is different.

**`fp16-int8`** — `trtexec` **`--fp16`** + **`--int8`**

Classic TensorRT combination for **FP** ONNX when you want both FP16 and INT8 kernels in the search space. For **YOLO-style** models from a **non-quantized** checkpoint, this is often a **good practical choice** (with TensorRT’s INT8 requirements, calibration, and operator support as documented by NVIDIA). It is **not** a drop-in substitute for Model Optimizer PTQ artifacts; those are a separate pipeline.

| `--mode` | Flags | Summary |
|----------|-------|---------|
| **`best`** (default) | **`--best`** | Broad tactic / precision search; good default when optimizing latency. |
| **`strongly-typed`** | **`--stronglyTyped`** | Strict ONNX types; avoid as default for YOLO if throughput matters. |
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

## `model-opt-yolo trt-bench`

Runs **`trtexec`** with **`--loadEngine`** on an **existing** TensorRT **`.engine`** (no ONNX rebuild). Shapes come from the serialized plan — there is no **`--batch`** / **`--img-size`** here. Follows NVIDIA’s [TensorRT performance best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) (warmup, iterations, duration; plus CUDA graph, spin wait, no transfers for isolated GPU timing).

**Common arguments**

| Argument | Description |
|----------|-------------|
| `--engine` | Input **`.engine`** path (**required**) |
| `--warm-up` | `trtexec --warmUp` in **milliseconds** (default **500**) |
| `--iterations` | `trtexec --iterations`: minimum inference iterations (default **100**) |
| `--duration` | `trtexec --duration` in **seconds** (default **60**) |
| `--log-file`, `-v` | Logging (default: `<artifacts>/trt_engine/logs/trt_bench_<engine-stem>_<timestamp>.log`) |

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

## Environment variables (logging)

- `MODELOPT_YOLO_LOGLEVEL` or `LOGLEVEL` — `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `MODELOPT_ARTIFACTS_ROOT` — root for artifacts (default: `<cwd>/artifacts`, created if missing)
