---
name: modelopt-onnx-ptq-dev
description: >-
  modelopt-onnx-ptq repository: layout, coding rules, ONNX PTQ (NVIDIA Model Optimizer),
  Docker/CUDA, full CLI (download-coco through trex-analyze), pipeline-e2e, YAML profiles,
  eval-trt I/O (onnx-eval-io-autodetect), and troubleshooting pointers. Use for refactors,
  new CLI flags, or aligning docs/skills with code.
---

# Model-Optimizer-ONNX — development skill (umbrella)

**Agent Skills** for this project live under **`skills/`** (published on GitHub; portable to Claude Code, Cursor, or other agents). See **[`skills/README.md`](../README.md)**.

**Bundled domain skills:**

| Path | Contents |
|------|----------|
| [`skills/onnx-ptq/SKILL.md`](../onnx-ptq/SKILL.md) | PTQ workflow (env → download-coco → calib → quantize → validate → build-trt), CLI index, mode/method tables, autotune. |
| [`skills/onnx-ptq/reference.md`](../onnx-ptq/reference.md) | `quantize()` signature, `python -m modelopt.onnx.quantization` flags, opset, EP resolution. |
| [`skills/ptq-trt-performance/SKILL.md`](../ptq-trt-performance/SKILL.md) | mAP vs QPS, `pipeline-e2e`, `report-runs`, backbone/neck/head `include_nodes`, `trex-analyze`. |
| [`skills/onnx-eval-io-autodetect/SKILL.md`](../onnx-eval-io-autodetect/SKILL.md) | `eval-trt` **`--output-format auto`**, Ultralytics vs DeepStream layouts, **`--input-name`**, roadmap. |
| [`skills/modelopt-troubleshooting/SKILL.md`](../modelopt-troubleshooting/SKILL.md) | libcublas, EP errors, autotune Concat, `QuantizeLinear` / trtexec parse errors, OOM. |

**Privacy:** Do not embed customer hostnames, internal IPs, or credentials in skills or issues.

---

## Part A — Project rules (maintainer conventions)

The sections below collect **maintainer conventions** for this repository. If a topic overlaps a domain skill, prefer the **skill** for command examples and this part for policy and naming.

### A.1 Project overview

ONNX post-training quantization (PTQ) pipeline for object-detection and vision ONNX models using **NVIDIA Model Optimizer** (`nvidia-modelopt[onnx]`), calibrated with COCO val2017 data, targeting TensorRT deployment.

**Upstream pin:** [`docker/Dockerfile`](../../docker/Dockerfile) installs **`nvidia-modelopt[onnx]==0.43.0`** from NVIDIA PyPI (`MODELOPT_VERSION`; override at `docker build --build-arg MODELOPT_VERSION=…`).

**Repository layout**

| Path | Purpose |
|------|---------|
| `docker/` | Container image definition; **TREx** under `/workspace/TREx` in **`env_trex`** (`install.sh --venv --full`), separate from `modelopt-onnx-ptq` — see `docs/docker-reference.md` |
| `modelopt_onnx_ptq/`, `pyproject.toml` | Installable package (`pip install .`); CLI **`modelopt-onnx-ptq`** — see **CLI commands** below |
| `skills/` | Portable Agent Skills for coding agents (this umbrella + domain skills) |
| `models/` | Input ONNX weights (user-provided) |
| `data/coco/` | Datasets (`val2017/`, `annotations/`); large files gitignored |
| `artifacts/calibration/` | Calibration tensors (e.g. `calib_coco.npy`); gitignored |
| `artifacts/quantized/` | Quantized ONNX |
| `artifacts/trt_engine/` | TensorRT `.engine` and timing cache from `build-trt` (default output path) |
| `artifacts/predictions/` | COCO eval JSON exports; gitignored |

Optional reference trees under `.garbage_data/source/` are not part of the maintained workflow.

### CLI commands (`modelopt-onnx-ptq`)

Source of truth: `modelopt_onnx_ptq/cli.py` and per-module `--help`.

| Command | Purpose |
|---------|---------|
| `download-coco` | COCO val2017 + annotations → `data/coco` (or `--output-dir`) |
| `calib` | Calibration `.npy` from image folders |
| `quantize` | Model Optimizer PTQ; `--autotune` with presets `quick`, `default`, `extensive` (int8/fp8) |
| `build-trt` | TensorRT engine from ONNX; `--mode` strongly-typed, best, fp16, fp16-int8 |
| `trt-bench` | Throughput/latency on an existing `.engine` |
| `eval-trt` | COCO mAP; `--output-format auto\|…` and optional `--onnx` |
| `report-runs` | Markdown report from logs; `--session-id` → `artifacts/pipeline_e2e/sessions/…` |
| `pipeline-e2e` | End-to-end: calib → FP16 baseline → quantize → build-trt → eval-trt → trt-bench → report |
| `trex-analyze` | TREx / trtexec profiling (needs **`TREX_VENV`**) |

Aliases in code: `download_coco`, `eval_trt`, `build_trt`, `trt_bench`, `report_runs`, `pipeline_e2e`, `trex_analyze`, `e2e`.

**Pipeline (correct order)**

```
ONNX FP32 → calib → quantize [--autotune] → build-trt → eval-trt → trt-bench
```

- **Autotune is a flag inside `quantize`**, not a separate step. It runs inside `modelopt.onnx.quantization` after calibration/Q-DQ insertion, selectively removing Q/DQ nodes that hurt TensorRT latency.
- **Autotune only supports int8 and fp8.** For int4, `modelopt` ignores the `--autotune` flag (the int4 code path does not invoke the autotuner).
- There is **no standalone `autotune` command** — it was removed.

**Key conventions**

- All quantization goes through `modelopt.onnx.quantization.quantize()` (or the CLI `python -m modelopt.onnx.quantization`).
- Autotune is passed as `--autotune=<quick|default|extensive>` to the same quantization CLI/API.
- Calibration data is always a NumPy `.npy` array matching the ONNX model's input shape/preprocessing.
- The canonical environment is the Docker container (`docker/Dockerfile`); local runs require matching CUDA/cuDNN/TensorRT versions.
- Scripts use `argparse` and delegate to `subprocess.call` or direct Python API imports.
- Python >= 3.10, CUDA 12.x or 13.x. ORT version must match CUDA major version.

**Naming**

- Quantized outputs: `<model_stem>.<mode>.<method><suffix>` (e.g. `detector.int8.entropy.quant.onnx`).
- Default calibration path: `artifacts/calibration/calib_coco.npy`.

**Do not**

- Commit ONNX weights, COCO images, or `.npy` calibration files.
- Hardcode GPU device IDs; use `--calibration_eps` to control EP selection.
- Mix CUDA toolkit major versions between ORT and system libs.
- Create a standalone autotune command or step — autotune is always `quantize --autotune`.

---

### A.2 Code organization

**File size**

- Avoid source files **longer than ~500 lines**. When a module grows past that, **split** it into smaller files with clear responsibilities.
- Goal: readability, review, and maintenance — not a hard cap for exceptional cases (e.g. generated data), but the norm should be **short modules**.

**Directory layout**

- Organize code **by domain or feature** (folders that reflect responsibilities), instead of piling everything into one package-level file.
- Put shared helpers in dedicated modules (e.g. `*_utils.py`, `.../utils/`) when it makes sense, rather than duplicating logic.

**Reusable functions**

- **Extract** repeated or generic logic into **reusable** functions (or small classes) with stable names and signatures.
- Prefer pure functions or localized side effects when possible so CLI, pipelines, and scripts can reuse the same building blocks.

When adding new code or refactoring oversized files, apply these guidelines pragmatically — without drive-by large refactors unless requested.

---

### A.3 Documentation and naming

**User-facing docs must follow code changes**

Whenever you **change behavior**, **CLI flags**, **defaults**, **paths**, or **workflows** in this repository, update the **same change** in user-facing documentation so it stays accurate.

- **Primary locations:** `README.md`, `docs/*.md` (especially `docs/cli-reference.md`), and any strings shown by `modelopt-onnx-ptq --help` / subcommand help in `modelopt_onnx_ptq/`.
- **Scope:** If the change is user-visible (how to run a command, what a flag does, output layout, Docker/TREx notes), the docs must reflect it in the same PR or edit — do not leave stale examples or flag lists.
- **Comments in code** still follow the rules below (minimal); this section is about **markdown docs and CLI help**, not inline comment volume.

**Comments — less is more**

- **Do NOT** add comments that narrate what the code already says. The code should speak for itself.
- Only comment **why** something non-obvious is done — never **what** it does.
- No tutorial-style explanations. Assume the reader is a competent developer.
- One-line docstrings on public functions are welcome; verbose `Args:`/`Returns:` blocks only when the signature is genuinely ambiguous.
- Skip docstrings entirely on trivial or self-explanatory functions.

```python
# bad — narrates the obvious
img = img / 255.0  # normalize image to 0-1 range

# bad — teaches a lesson
# We use Path here because pathlib provides a more Pythonic way to handle
# filesystem paths compared to os.path, allowing operator overloading for
# path concatenation and better cross-platform compatibility.
output = base_dir / "model.onnx"

# good — explains a non-obvious choice
# TRT requires even spatial dims; pad asymmetrically to preserve alignment
padded = np.pad(tensor, ((0,0), (0,0), (0,1), (0,1)))

# good — no comment needed, code is clear
output = base_dir / "model.onnx"
```

This applies to **all** file types: `.py`, `Dockerfile`, `.sh`, `.yaml`, etc.

**Naming**

| Element | Convention | Example |
|---------|------------|---------|
| Functions / methods | `snake_case` | `build_engine()` |
| Classes | `PascalCase` | `CalibrationRunner` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_IMG_SIZE = 640` |
| Private helpers | `_leading_underscore` | `_parse_row()` |
| Module files | `snake_case.py` | `trt_bench.py` |

**Constants**

- Extract magic numbers and repeated string literals into **named constants** at module level.
- Group related constants near the top of the file, below imports.

**Module docstring**

- A brief one-line module docstring is enough. Skip it if the filename already makes the purpose obvious.

---

### A.4 Python style

**Type hints**

- Annotate **all** public function signatures (args + return type).
- Use built-in generics (`list[str]`, `dict[str, int]`, `tuple[int, ...]`) — no `typing.List` etc. (Python >= 3.10).
- Use `X | None` instead of `Optional[X]`.
- For complex types, define a `TypeAlias` at module level.

**Paths**

- Use `pathlib.Path` for filesystem operations — not `os.path.join` or string concatenation.
- Accept `str | Path` at public API boundaries; convert early with `Path(x)`.

**Imports**

- **Order**: stdlib → third-party → local (one blank line between groups).
- No wildcard imports (`from x import *`).
- Prefer absolute imports; relative only inside the same sub-package.

**Strings**

- Prefer **f-strings** over `.format()` or `%` formatting.
- Use raw strings (`r"..."`) for regex patterns.

**Miscellaneous**

- Prefer `enumerate()` over manual index counters.
- Use unpacking (`a, b = pair`) instead of index access when the structure is known.
- Prefer list/dict/set comprehensions over `map()`/`filter()` with lambdas for readability.

---

### A.5 Error handling and logging

**Logging over print**

- Use the `logging` module for operational output — not `print()`.
- Create a module-level logger: `logger = logging.getLogger(__name__)`.
- Reserve `print()` for CLI user-facing output only (e.g. final summary tables).

**Exceptions**

- Raise **specific** exception types (`ValueError`, `FileNotFoundError`, `RuntimeError`) — never bare `Exception`.
- Include actionable context in the message (what was expected vs. what was found).
- Catch the **narrowest** exception possible; avoid bare `except:` or `except Exception:` unless re-raising.

**Resource management**

- Use **context managers** (`with`) for files, database connections, temporary directories, and subprocess pipes.
- Prefer `tempfile.TemporaryDirectory()` over manual `mkdir` + `shutil.rmtree`.

**Subprocess calls**

- Check return codes: use `subprocess.run(..., check=True)` or inspect `returncode` explicitly.
- Capture and log `stderr` on failure for debuggability.

---

### A.6 ONNX quantization with Model Optimizer

See also **[`skills/onnx-ptq/SKILL.md`](../onnx-ptq/SKILL.md)** and **[`skills/onnx-ptq/reference.md`](../onnx-ptq/reference.md)**.

**Primary API**

```python
from modelopt.onnx.quantization import quantize

quantize(
    onnx_path="model.onnx",
    quantize_mode="int8",            # int8 | fp8 | int4
    calibration_data=np_array,       # np.ndarray or dict[str, np.ndarray]
    calibration_method="entropy",    # int8/fp8: entropy|max; int4: awq_clip|rtn_dq
    calibration_eps=["cpu", "cuda:0", "trt"],
    output_path="model.quant.onnx",
    autotune=True,                   # optional; only effective for int8/fp8
)
```

**CLI equivalent**

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=model.onnx \
  --quantize_mode=int8 \
  --calibration_data_path=calib.npy \
  --calibration_method=entropy \
  --output_path=model.quant.onnx \
  --autotune=default
```

**Note:** Upstream **`python -m modelopt.onnx.quantization`** uses **`--calibration_data_path`**. This repo’s wrapper **`modelopt-onnx-ptq quantize`** uses **`--calibration_data`** (passthrough to the same pipeline).

**Dynamic heads / tricky exports:** **`modelopt-onnx-ptq quantize`** defaults **`--high_precision_dtype`** to **`fp16`**. If `infer_shapes` fails, use **`fp32`**.

**Autotune (Q/DQ placement optimization)** — integrated in `quantize()`, not a separate command. Internal steps: preprocess → (optional) autotune regions → calibration → insert Q/DQ → export.

**Mode support**

| Mode | Autotune supported | Notes |
|------|--------------------|-------|
| `int8` | Yes | Full autotune with TRT benchmarking |
| `fp8` | Yes | Full autotune with TRT benchmarking |
| `int4` | **No** | `quantize_int4()` ignores autotune entirely |

**Presets** (`--autotune`): `quick` | `default` | `extensive` — schemes per region vs speed vs quality (see onnx-quantization rule in repo).

**Execution providers (`calibration_eps`)**

| Token | ORT Provider | Notes |
|-------|-------------|-------|
| `cpu` | CPUExecutionProvider | Always available |
| `cuda:0` | CUDAExecutionProvider | Requires cuDNN in LD_LIBRARY_PATH |
| `trt` | TensorrtExecutionProvider | Requires `tensorrt>=10.0` + cuDNN + libcublas matching ORT's CUDA version |
| `dml:0` | DmlExecutionProvider | Windows only |

Default order: `["cpu", "cuda:0", "trt"]`. If TRT/CUDA fail, ORT falls back to CPU.

**Calibration data**

- Type: `np.ndarray` (single input) or `dict[str, np.ndarray]` (multi-input).
- Shape must match model's input spec (batch dim is split into iterations automatically).
- Recommended: >= 500 images for CNN/detection models, ~64 for INT4 weight-only.
- Preprocessing **must** match ONNX export (size, RGB/BGR, letterbox, normalization).

**Common parameters**

- `--calibrate_per_node`: Reduces VRAM during calibration (int8/fp8 only).
- `--use_external_data_format`: Required for models > 2GB.
- `--high_precision_dtype`: fp16 (default in `modelopt-onnx-ptq quantize`) | fp32 | bf16 for non-quantized ops.
- `--simplify`: Runs onnxsim before quantization.
- `--trt_plugins`: Space-separated `.so` paths for custom TRT ops.

**Error patterns**

- `libcublas.so.12 not found` → CUDA version mismatch between ORT and system.
- `Failed to import tensorrt` → Install `tensorrt>=10.0` or remove `trt` from EPs.
- `cuDNN not found in LD_LIBRARY_PATH` → Export `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`.
- Model > 2GB without `--use_external_data_format` → ValueError with hint.

---

### A.7 Docker and environment

**Base image**

```dockerfile
FROM nvcr.io/nvidia/tensorrt:26.02-py3
```

Alternative images: NGC PyTorch (`nvcr.io/nvidia/pytorch:*`), NeMo, or TRT-LLM (`nvcr.io/nvidia/tensorrt-llm/release:*`).

**Required environment variables**

```dockerfile
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
```

For TensorRT images, also set:

```bash
export PIP_CONSTRAINT=""
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/include:/usr/lib/x86_64-linux-gnu"
```

**CUDA / ORT version alignment**

- `onnxruntime-gpu` on PyPI is built for a specific CUDA major version (typically 12).
- System CUDA toolkit must match. If system has CUDA 13 only, either: install `cuda-cublas-12-*` compat packages, or use nightly ORT with CUDA 13 support, or run inside the recommended Docker image.
- Verify with: `ldd $(python -c "import onnxruntime; print(onnxruntime.__file__.replace('__init__.py','capi/libonnxruntime_providers_cuda.so'))")`

**Install pattern (see `docker/Dockerfile` for exact pins)**

Model Optimizer is often installed from **GitHub `main`** (PyPI wheel may lack full ONNX autotune CLI). ORT GPU may be reinstalled from the CUDA 13 nightly index.

**Docker run**

Bind-mount persistent dirs (`models/`, `data/`, `artifacts/`). Optional dev mount: `-v "$(pwd)":/workspace/modelopt-onnx-ptq` from a clone.

---

### A.8 Calibration data preparation

**Letterbox preprocessing (default for common detection exports)**

1. Load image as RGB (PIL/OpenCV)
2. Letterbox resize to `img_size` (default 640), maintaining aspect ratio with gray padding (114,114,114)
3. Convert to float32, divide by 255.0
4. Transpose to NCHW: `(N, 3, H, W)`
5. Stack into single NumPy array and save as `.npy`

**Preprocessing must match export**

| Parameter | Default (this CLI) | Adjust if different |
|-----------|-------------|---------------------|
| Color space | RGB | `--bgr` flag |
| Resize method | Letterbox | `--no-letterbox` for stretch |
| Input size | 640x640 | `--img_size` |
| Normalization | /255.0 | Custom if model expects different |
| Dtype | float32 | `--fp16` for FP16 activations |

**Output format**

- Single-input: `np.ndarray` `(N, C, H, W)` → `.npy`.
- Multi-input: `dict[str, np.ndarray]` → `.npz`.
- modelopt CLI uses `--calibration_data_path`; Python API uses `calibration_data=`.

---

### A.9 PTQ / TensorRT performance measurement

Read **[`skills/ptq-trt-performance/SKILL.md`](../ptq-trt-performance/SKILL.md)** first for pitfalls (session-only reports, TRT build modes) and the **backbone / neck / head** whitelist workflow (enumerate Convs → Netron cuts → regex `include_nodes` → validate → quantize/bench). User-facing docs: **`docs/quantization-performance-workflow.md`**, **`docs/workflow.md`** (`--quant-matrix`).

**Automated grid**

1. **`pipeline-e2e --onnx models/…onnx`** — required.
2. **`--output-format`** — default **`auto`** (recommended); set **`--input-name`** if **build-trt** cannot infer the ONNX input tensor name.
3. **`--quant-matrix all`** (or `int8.all`, `fp8.entropy`, …) — selects PTQ combos.
4. Optional **`--quantize-profile`** — YAML applied to every `quantize` (e.g. a profile under `modelopt_onnx_ptq/profiles/`).
5. Optional **`--high-precision-dtype fp16`** — high-precision strips in FP16 after PTQ.
6. Omit **`--autotune`** if tuning via profiles only.
7. **`--session-id <id>`** — keeps all logs under `artifacts/pipeline_e2e/sessions/<id>/`.
8. Open **`report_<id>.md`** in the session folder; if it mixed global runs, re-run **`report-runs --session-id <id> -o …/report_session_only.md`** without merge flags.

**Manual profile tuning**

1. **`calib`** → fixed `calib.npy`.
2. **`quantize --profile … --suffix …`** — variants must not overwrite.
3. **`build-trt --mode strongly-typed`** (or `best` if build fails).
4. **`eval-trt`** → mAP; **`trt-bench`** → QPS/latency.
5. Optional **`trex-analyze`** → edit profile → repeat.

Do not commit calibration `.npy`, engines, or large binaries.

---

## Part B — Where to look next

| Need | Open |
|------|------|
| Step-by-step PTQ + pipeline-e2e one-liners | [`skills/onnx-ptq/SKILL.md`](../onnx-ptq/SKILL.md) |
| `quantize()` / CLI flag table / opset / AutoCast | [`skills/onnx-ptq/reference.md`](../onnx-ptq/reference.md) |
| Benchmarking, `report-runs`, backbone-neck-head splits | [`skills/ptq-trt-performance/SKILL.md`](../ptq-trt-performance/SKILL.md) |
| `eval-trt` layouts, `--output-format auto` | [`skills/onnx-eval-io-autodetect/SKILL.md`](../onnx-eval-io-autodetect/SKILL.md) |
| Environment and runtime errors | [`skills/modelopt-troubleshooting/SKILL.md`](../modelopt-troubleshooting/SKILL.md) |
| CLI flags for this repo | `docs/cli-reference.md` |

---

## When to use this umbrella vs a domain skill

- **Umbrella (this file):** Conventions, layout, rules, and pointers — editing **`modelopt_onnx_ptq/`** or docs.
- **Domain skills:** Running commands, profiling, or debugging quantization/TRT end-to-end.
