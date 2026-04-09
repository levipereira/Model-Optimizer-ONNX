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
| `build-trt` | Run `trtexec` to build a `.engine` from ONNX (strongly-typed or benchmark mode) |
| `eval-trt` | COCO mAP for end-to-end TRT engines (`num_dets`, `det_boxes`, `det_scores`, `det_classes`) |

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
| `--mode` | `strongly-typed` (default) or `benchmark` |
| `--engine-out` | Output `.engine` path (default: same basename as ONNX, `.engine`) |
| `--timing-cache` | Timing cache file (default: `<engine>.timing.cache`) |
| `--log-file`, `-v` | Logging (default log under `<artifacts>/quantized/logs/build_trt_*.log`) |

### Modes

| `--mode` | Behavior |
|----------|----------|
| **`strongly-typed`** | For Q/DQ ONNX from Model Optimizer: `--stronglyTyped`, and `min/opt/max` shapes all use **batch × 3 × H × W** (default for PTQ exports). |
| **`benchmark`** | For throughput / latency measurement: adds **`--fp16`** and **`--int8`**, **`--warmUp`**, **`--duration`**, **`--useCudaGraph`**, **`--useSpinWait`**, **`--noDataTransfers`**. **`minShapes`** use batch **1**; **`opt`/`max`** use **`--batch`**. |

### Pass-through (`--` …)

Arguments after `--` are appended to the **`trtexec`** command (after the built-in flags). Use this to **add** flags or **override** behavior (later tokens win depending on `trtexec`).

```bash
model-opt-yolo build-trt --onnx model.quant.onnx --img-size 640 --batch 1 -- --verbose
model-opt-yolo build-trt --onnx model.onnx --mode benchmark --batch 4 -- --workspace=8192
```

---

## `model-opt-yolo eval-trt`

Evaluates an **end-to-end** TensorRT engine on COCO (post-processed tensors only). Expected bindings:

| Role | Tensor | Shape |
|------|--------|--------|
| Input | `images` | `[B, 3, H, W]` |
| Output | `num_dets` | `[B, 1]` |
| Output | `det_boxes` | `[B, K, 4]` |
| Output | `det_scores` | `[B, K]` |
| Output | `det_classes` | `[B, K]` |

**Dynamic batch:** `B` can be a dynamic dimension in the engine profile. This command runs **batch size 1** per image for mAP. Raw pre-NMS exports are not supported yet.

| Argument | Description |
|----------|-------------|
| `--engine` | Path to `.engine` file |
| `--images` | COCO images directory |
| `--annotations` | `instances_val2017.json` |
| `--img-size` | Hint (may be overridden by engine) |
| `--conf-thres` | Confidence threshold |
| `--save-json` | Predictions JSON path |

---

## Environment variables (logging)

- `MODELOPT_YOLO_LOGLEVEL` or `LOGLEVEL` — `DEBUG`, `INFO`, `WARNING`, `ERROR`
- `MODELOPT_ARTIFACTS_ROOT` — root for artifacts (default: `<cwd>/artifacts`, created if missing)
