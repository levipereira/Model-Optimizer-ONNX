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
| `MODELOPT_VERSION` | `0.43.0rc4` | Pin for `nvidia-modelopt[onnx]==…` from NVIDIA PyPI (`NVIDIA_PYPI_EXTRA_INDEX`) |
| `NVIDIA_PYPI_EXTRA_INDEX` | `https://pypi.nvidia.com` | `--extra-index-url` for `nvidia-modelopt` |
| `ORT_CUDA13_INDEX` | Azure DevOps **ort-cuda-13-nightly** PyPI simple URL | Source for `onnxruntime-gpu` rebuild |
| `TENSORRT_TREX_GIT_REF` | `release/10.15` | Git branch (or tag) for the shallow [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) clone used by TREx (TensorRT Engine Explorer) |

---

## TREx for model profiling

The image includes an optional **TensorRT Engine Explorer (TREx)** checkout for analyzing **`.engine`** plans and **`trtexec`** JSON (layers, timing, notebooks). It is **not** required for **`modelopt-onnx-ptq`** PTQ.

### `install.sh`: `--full` vs `--core` vs `--venv`

These flags come from [NVIDIA’s `install.sh`](https://github.com/NVIDIA/TensorRT/blob/release/10.15/tools/experimental/trt-engine-explorer/install.sh):

| Option | Meaning |
|--------|---------|
| **`--full`** (default) | `pip install -e .[notebook]` — TREx **core** deps plus **notebook** extras (Jupyter, plotting, etc.). |
| **`--core`** | `pip install -e .` — **core** dependencies only (`requirements.txt`; lighter, no notebook stack). |
| **`--venv`** | Before the pip step, create a subdirectory **`env_trex`** with **`virtualenv`** and **activate** it so TREx installs **only inside that venv**. |

**Important:** **`trtexec`** is the TensorRT **binary** on **`PATH`** from the NGC image. It is **not** installed by TREx or by a venv. **`--venv`** isolates **Python packages** (TREx, pandas, Jupyter, …), not **`trtexec`**.

This Dockerfile runs **`source ./install.sh --venv --full`**: TREx and its pins (**`pandas==2.2.1`**, etc.) live in **`env_trex`** at **`TREX_VENV`**, separate from the main **`pip`** where **`modelopt-onnx-ptq`**, Model Optimizer, and ORT are installed — avoiding clashes with **numpy** / **CuPy** / other stacks.

The build **`touch`**es **`bin/__init__.py`**, **`sed`**-patches **`bin/trex.py`** (see [Troubleshooting — TREx](troubleshooting.md#trex-cli-import-errors-bin--utils)). **`modelopt-onnx-ptq`** is **not** installed into **`env_trex`** (that would pull **torch** / **pycuda** and duplicate the whole CLI stack). Instead, **`trex-analyze`** prepends **`env_trex`**’s **`site-packages`** to **`sys.path`** so **`import trex`** works while the process keeps the **image** Python where **`modelopt-onnx-ptq`** is installed; it **re-executes** with **`TREX_VENV`**’s interpreter only if **trex** is still missing (set **`MODELOPT_TREX_NO_REEXEC=1`** to skip re-exec for debugging).

| Path | Contents |
|------|----------|
| **`/workspace/TREx`** | Shallow clone of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) at **`TENSORRT_TREX_GIT_REF`** (default **`release/10.15`**). |
| **`/workspace/TREx/tools/experimental/trt-engine-explorer/`** | TREx package root: **`install.sh`**, **`env_trex/`** (venv), `notebooks/`, `utils/`, upstream `README.md`. |

**Usage (inside the container):**

```bash
source /workspace/TREx/tools/experimental/trt-engine-explorer/env_trex/bin/activate
trex --help
# modelopt-onnx-ptq trex-analyze ...  # re-execs into env_trex automatically if needed
```

**Version note:** the **`release/10.15`** Git tree is **source** for TREx scripts/notebooks; the **TensorRT** runtime, **`trtexec`**, and **`import tensorrt`** still come from the [NGC base image](#base-image). Minor skew between branch docs and the container TRT version is possible.

**Build-time dependencies:** **`graphviz`** and **`sudo`** (apt). **`sudo`** is required because **`install.sh`** runs **`sudo apt install graphviz`**.

---

## Build sequence (summary)

1. Install **git** (for the TREx clone).
2. Install **`nvidia-modelopt[onnx]==MODELOPT_VERSION`** from NVIDIA PyPI (`NVIDIA_PYPI_EXTRA_INDEX`).
3. **`pip install /workspace/modelopt-onnx-ptq`** — installs **`modelopt-onnx-ptq`** from copied `pyproject.toml` and `modelopt_onnx_ptq/`.
4. Reinstall **`onnxruntime-gpu`** from the CUDA 13 nightly index with **`--no-deps`** so wheels match the container toolkit.
5. Clone **TensorRT**, patch TREx, run **`source ./install.sh --venv --full`** (no second **`pip install`** of **`modelopt-onnx-ptq`** into **`env_trex`**).

---

## Files copied into the image

- `pyproject.toml`, `README.md`, `LICENSE`, `NOTICE` → `/workspace/modelopt-onnx-ptq/`
- `modelopt_onnx_ptq/` package → `/workspace/modelopt-onnx-ptq/modelopt_onnx_ptq/`

`WORKDIR` is **`/workspace/modelopt-onnx-ptq`** so the shell and relative paths (`models/`, `data/`, `artifacts/`) align with that tree.

Rebuild the image after changing package code or dependencies.

---

## Data persistence (volume mapping)

`modelopt-onnx-ptq` is installed **in the image** (`pip install` during `docker build`). You do **not** have to mount the Git repository to run the CLI.

**Recommended run** — bind only three host directories so weights, datasets, and generated files survive after `exit`:

```bash
export DATA_ROOT="$HOME/modelopt-onnx-ptq"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/data" "$DATA_ROOT/artifacts"

docker run --gpus all --rm -it \
  -w /workspace/modelopt-onnx-ptq \
  -v "$DATA_ROOT/models:/workspace/modelopt-onnx-ptq/models" \
  -v "$DATA_ROOT/data:/workspace/modelopt-onnx-ptq/data" \
  -v "$DATA_ROOT/artifacts:/workspace/modelopt-onnx-ptq/artifacts" \
  modelopt-onnx-ptq
```

| Host | Container | Contents |
|------|-----------|----------|
| `$DATA_ROOT/models` | `/workspace/modelopt-onnx-ptq/models` | Input ONNX (you provide) |
| `$DATA_ROOT/data` | `/workspace/modelopt-onnx-ptq/data` | e.g. COCO after `download-coco` |
| `$DATA_ROOT/artifacts` | `/workspace/modelopt-onnx-ptq/artifacts` | Calibration, quantized ONNX, `trt_engine/` (`.engine`), logs ([Artifacts & logging](artifacts-and-logging.md)) |

Use any host path for `DATA_ROOT` (second disk, project-specific folder, etc.).

If **`MODELOPT_ARTIFACTS_ROOT`** is unset, the CLI uses `<cwd>/artifacts`. With `-w /workspace/modelopt-onnx-ptq`, that is `/workspace/modelopt-onnx-ptq/artifacts`, i.e. the mounted `$DATA_ROOT/artifacts`.

**Optional — full repo mount for development:** from a clone, `-v "$(pwd)":/workspace/modelopt-onnx-ptq -w /workspace/modelopt-onnx-ptq` overlays the project tree (editable source on the host). For production-style runs, the three mounts above are enough.
