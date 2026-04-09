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
| `$DATA_ROOT/artifacts` | `/workspace/model-opt-yolo/artifacts` | Calibration, quantized ONNX, engines, logs ([Artifacts & logging](artifacts-and-logging.md)) |

Use any host path for `DATA_ROOT` (second disk, project-specific folder, etc.).

If **`MODELOPT_ARTIFACTS_ROOT`** is unset, the CLI uses `<cwd>/artifacts`. With `-w /workspace/model-opt-yolo`, that is `/workspace/model-opt-yolo/artifacts`, i.e. the mounted `$DATA_ROOT/artifacts`.

**Optional — full repo mount for development:** from a clone, `-v "$(pwd)":/workspace/model-opt-yolo -w /workspace/model-opt-yolo` overlays the project tree (editable source on the host). For production-style runs, the three mounts above are enough.
