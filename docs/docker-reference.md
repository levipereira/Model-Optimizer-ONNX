# Docker reference

## Base image

```dockerfile
FROM nvcr.io/nvidia/tensorrt:26.02-py3
```

- **TensorRT** and **CUDA** versions follow the NGC tag **26.02-py3**.
- Documentation: [NVIDIA TensorRT container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt).

---

## Environment

Set inside the Dockerfile for Model Optimizer compatibility with TensorRT images:

- `CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/`
- `LD_LIBRARY_PATH` includes `CUDNN_LIB_DIR`

---

## Build arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `MODELOPT_GIT_REF` | `main` | Git ref for `nvidia-modelopt[onnx] @ git+https://github.com/NVIDIA/Model-Optimizer.git@...` |
| `ORT_CUDA13_INDEX` | Azure DevOps **ort-cuda-13-nightly** PyPI simple URL | Source for `onnxruntime-gpu` rebuild |
| `TENSORRT_TREX_GIT_REF` | `release/10.15` | Git branch (or tag) for the shallow [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) clone used by TREx (TensorRT Engine Explorer) |

---

## TREx for model profiling

The image includes an optional **TensorRT engine profiling** stack, separate from the **`model-opt-yolo`** Python environment:

| Path | Contents |
|------|----------|
| **`/workspace/TREx`** | Shallow git clone of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) at **`TENSORRT_TREX_GIT_REF`** (default **`release/10.15`**). |
| **`/workspace/TREx/env`** | Python **venv** with [TREx](https://github.com/NVIDIA/TensorRT/tree/release/10.15/tools/experimental/trt-engine-explorer) (*trt-engine-explorer*) installed in editable mode with **`[notebook]`** extras (Jupyter, plotting, and related dependencies per upstream `setup.py`). |
| **`/workspace/TREx/tools/experimental/trt-engine-explorer/`** | TREx package root: notebooks (`notebooks/`), `utils/` (e.g. engine processing helpers), and upstream `README.md`. |

**Purpose:** analyze TensorRT **`.engine`** plans and profiling artifacts (layers, timing, comparisons). This matches the experimental **TensorRT Engine Explorer** workflow described in NVIDIA’s documentation and blog posts; it is **not** required for ONNX PTQ, calibration, **`quantize`**, or **`build-trt`** inside this project.

**Usage (inside the container):**

```bash
source /workspace/TREx/env/bin/activate
trex --help
```

Do **not** prepend **`/workspace/TREx/env/bin`** to the global `PATH` in the image: the default shell should keep using the base environment where **`model-opt-yolo`** is installed. Activate the TREx venv only when you run TREx or its notebooks.

**Version note:** the NGC base image ships a fixed **TensorRT** runtime (see [Base image](#base-image)). The **`release/10.15`** tree provides **source** and **TREx** tooling aligned with that product line; minor differences from the container’s preinstalled TensorRT Python wheel are possible. For engine analysis, use **`trtexec`** and artifacts produced in the same container when troubleshooting.

**Build-time dependencies:** the Dockerfile installs **`graphviz`** (apt) for TREx graph backends (`pydot` / Graphviz) and **`python3-venv`** so **`python3 -m venv`** can create **`/workspace/TREx/env`** reliably.

---

## Build sequence (summary)

1. Install **git** (for pip VCS installs).
2. Install **`nvidia-modelopt[onnx]`** from GitHub at `MODELOPT_GIT_REF`.
3. **`pip install /workspace/model-opt-yolo`** — installs **`model-opt-yolo`** from copied `pyproject.toml` and `model_opt_yolo/`.
4. Reinstall **`onnxruntime-gpu`** from the CUDA 13 nightly index with **`--no-deps`** so wheels match the container toolkit.

---

## Files copied into the image

- `pyproject.toml`, `README.md`, `LICENSE`, `NOTICE` → `/workspace/model-opt-yolo/`
- `model_opt_yolo/` package → `/workspace/model-opt-yolo/model_opt_yolo/`

`WORKDIR` is **`/workspace/model-opt-yolo`** so the shell and relative paths (`models/`, `data/`, `artifacts/`) align with that tree.

Rebuild the image after changing package code or dependencies.

---

## Data persistence (volume mapping)

`model-opt-yolo` is installed **in the image** (`pip install` during `docker build`). You do **not** have to mount the Git repository to run the CLI.

**Recommended run** — bind only three host directories so weights, datasets, and generated files survive after `exit`:

```bash
export DATA_ROOT="$HOME/model-opt-yolo"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/data" "$DATA_ROOT/artifacts"

docker run --gpus all --rm -it \
  -w /workspace/model-opt-yolo \
  -v "$DATA_ROOT/models:/workspace/model-opt-yolo/models" \
  -v "$DATA_ROOT/data:/workspace/model-opt-yolo/data" \
  -v "$DATA_ROOT/artifacts:/workspace/model-opt-yolo/artifacts" \
  modelopt-yolo-ptq
```

| Host | Container | Contents |
|------|-----------|----------|
| `$DATA_ROOT/models` | `/workspace/model-opt-yolo/models` | Input ONNX (you provide) |
| `$DATA_ROOT/data` | `/workspace/model-opt-yolo/data` | e.g. COCO after `download-coco` |
| `$DATA_ROOT/artifacts` | `/workspace/model-opt-yolo/artifacts` | Calibration, quantized ONNX, `trt_engine/` (`.engine`), logs ([Artifacts & logging](artifacts-and-logging.md)) |

Use any host path for `DATA_ROOT` (second disk, project-specific folder, etc.).

If **`MODELOPT_ARTIFACTS_ROOT`** is unset, the CLI uses `<cwd>/artifacts`. With `-w /workspace/model-opt-yolo`, that is `/workspace/model-opt-yolo/artifacts`, i.e. the mounted `$DATA_ROOT/artifacts`.

**Optional — full repo mount for development:** from a clone, `-v "$(pwd)":/workspace/model-opt-yolo -w /workspace/model-opt-yolo` overlays the project tree (editable source on the host). For production-style runs, the three mounts above are enough.
