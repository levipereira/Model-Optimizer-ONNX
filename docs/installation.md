# Installation

## Requirements

**Default workflow (Docker)** — on the host you need:

| Requirement | Notes |
|-------------|--------|
| **NVIDIA GPU** | CUDA-capable GPU (consumer or datacenter). |
| **NVIDIA driver** | Installed so `nvidia-smi` works on the host. |
| **Docker** | [Docker Engine](https://docs.docker.com/engine/install/) running. |
| **NVIDIA Container Toolkit** | Required for `docker run --gpus all`. [Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). |

**Local / editable install** (without Docker): **Python ≥ 3.10** and a matching CUDA / TensorRT / ONNX Runtime stack on the host; the Docker image is the supported baseline.

---

## Install the CLI (editable)

From the repository root:

```bash
pip install -e .
```

This registers the **`model-opt-yolo`** command (see [CLI reference](cli-reference.md)).

**TREx:** do **not** install `trt-engine-explorer` (**trex**) into the same Python env as **`cupy`**, **PyTorch**, and **`model-opt-yolo`** unless you enjoy dependency fights (e.g. **trex** wants **`pandas==2.2.1`**; **cupy** may require **numpy ≥ 2**). Use a **dedicated venv** for TREx. The Docker image puts TREx in **`$TREX_VENV`**; **`trex-analyze`** switches to that interpreter when **trex** is not importable (see [Docker reference](docker-reference.md#trex-for-model-profiling)).

---

## COCO val2017 (optional)

For calibration and COCO mAP evaluation, download images and annotations:

```bash
model-opt-yolo download-coco --output-dir data/coco
```

This creates `data/coco/val2017/` and `data/coco/annotations/instances_val2017.json` (~1.3 GB total). Skips files that already exist.

---

## Docker (recommended)

Clone the repository and `cd` into it:

```bash
git clone https://github.com/levipereira/Model-Optimizer-YOLO.git
cd Model-Optimizer-YOLO
```

Build:

```bash
docker build -f docker/Dockerfile -t modelopt-yolo-ptq .
```

Run — mount only **`models/`**, **`data/`**, and **`artifacts/`** on the host (the package is already in the image):

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

The image includes:

- **NGC** [`nvcr.io/nvidia/tensorrt:26.02-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)
- **`nvidia-modelopt[onnx]`** pinned from **NVIDIA PyPI** (default `0.43.0rc4`, overridable with `MODELOPT_VERSION` at build time; index `NVIDIA_PYPI_EXTRA_INDEX`, default `https://pypi.nvidia.com`)
- **`model-opt-yolo`** via `pip install /workspace/model-opt-yolo`
- **`onnxruntime-gpu`** reinstalled from the **CUDA 13** nightly index (PyPI ORT targets CUDA 12; the image ships CUDA 13)

See [Docker reference](docker-reference.md) for build arguments.

### Edit mode with Docker (developers)

Use this when you want to **change the source code** and run commands in the **same** TensorRT image, with your **Git repository** on the host visible inside the container.

1. Build the image (see above).
2. From the **root of your clone**, start a shell with the repo mounted over `/workspace/model-opt-yolo`:

```bash
docker run --gpus all --rm -it \
  -w /workspace/model-opt-yolo \
  -v "$(pwd)":/workspace/model-opt-yolo \
  modelopt-yolo-ptq
```

Edits under `model_opt_yolo/` on the host apply inside the container immediately. After changing **`pyproject.toml`** or dependencies, reinstall with:

```bash
pip install -e /workspace/model-opt-yolo
```

You can add the same **`models/`**, **`data/`**, and **`artifacts/`** bind mounts as in the default run if you keep those folders **outside** the clone — see [Docker reference — Data persistence](docker-reference.md).

---

## ONNX Runtime on CUDA 13 (without Docker)

After `pip install -e .`, align **onnxruntime-gpu** with CUDA 13:

```bash
pip install --upgrade --force-reinstall --pre \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ \
  onnxruntime-gpu --no-deps
```

---

## Python dependencies

Runtime libraries are declared in [`pyproject.toml`](../pyproject.toml) (`datasets`, `numpy`, `opencv-python-headless`, `onnx`, `pycocotools`, `pycuda`, `torch`, etc.). The Docker build installs Model Optimizer and ORT separately as described above.
