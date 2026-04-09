# Model-Optimizer-YOLO

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorRT](https://img.shields.io/badge/NGC%20TensorRT-26.02--py3-76B900?logo=nvidia&logoColor=white)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)
[![Model Optimizer](https://img.shields.io/badge/NVIDIA%20Model%20Optimizer-GitHub%20%7C%20onnx-76B900?logo=nvidia&logoColor=white)](https://github.com/NVIDIA/Model-Optimizer)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime%20GPU-CUDA%2013%20nightly-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![model-opt-yolo](https://img.shields.io/badge/model--opt--yolo-v0.1.0-3775A9?logo=pypi&logoColor=white)](pyproject.toml)
[![Context7](https://img.shields.io/badge/Context7-Docs-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0id2hpdGUiLz48dGV4dCB4PSI3IiB5PSIxNyIgZm9udC1zaXplPSIxNCIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtd2VpZ2h0PSJib2xkIiBmaWxsPSIjMjU2M0VCIj5DPC90ZXh0Pjwvc3ZnPg==)](https://context7.com/levipereira/model-optimizer-yolo)

**ONNX post-training quantization (PTQ)** and **TensorRT** deployment helpers for **YOLO-style** detectors — built on [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), with COCO calibration and optional Q/DQ **autotune**.

| | |
|--|--|
| **CLI** | `model-opt-yolo` |
| **Docs** | [`docs/index.md`](docs/index.md) |

---

## Table of Contents

- [Pipeline](#pipeline)
- [Quick Steps](#quick-steps)
- [Supported Output Formats](#supported-output-formats)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Run with Docker (default)](#run-with-docker-default)
  - [Local Installation (optional)](#local-installation-optional)
- [License](#license)

---

## Pipeline

**Autotune** (Q/DQ placement search) is **optional**. **COCO images + annotations** and a **`calib.npy`** from **`calib`** are **required** for PTQ in the usual workflow (unless you supply an equivalent image set and build calibration yourself).

```mermaid
flowchart TD
  A[PyTorch weights — .pt] --> B[Export to ONNX<br/>--dynamic · --simplify · opset ≥ 18]
  B --> C[models/*.onnx]
  C --> H[download-coco]
  H --> I[model-opt-yolo calib]
  I --> J[artifacts/calibration/*.npy]
  J --> K{Autotune Q/DQ<br/>optional}
  K -->|yes| E[model-opt-yolo autotune]
  E --> F[artifacts/autotune/.../<br/>optimized_final.onnx]
  F --> Q[model-opt-yolo quantize]
  K -->|no| Q
  Q --> L[artifacts/quantized/*.quant.onnx]
  L --> M[model-opt-yolo build-trt]
  M --> N[artifacts/trt_engine/*.engine]
  N --> O[model-opt-yolo eval-trt<br/>COCO mAP]
```

*More detail: [docs/workflow.md](docs/workflow.md)*

---

## Quick Steps

Run these **inside the container** (or locally after `pip install -e .`):

1. Put ONNX under `models/` (export from PyTorch with the flags you use in production).
2. `model-opt-yolo download-coco --output-dir data/coco` — COCO val + annotations for **calib** and **eval**.
3. `model-opt-yolo calib --images_dir data/coco/val2017 --calibration_data_size 500 --img_size 640`
4. *(Optional)* `model-opt-yolo autotune …` if you want Q/DQ placement search before PTQ.
5. `model-opt-yolo quantize --calibration_data artifacts/calibration/…npy --onnx_path models/your.onnx` (use `artifacts/autotune/…/optimized_final.onnx` if you autotuned).
6. `model-opt-yolo build-trt --onnx artifacts/quantized/your…quant.onnx --img-size 640` (default engine: `artifacts/trt_engine/<same-stem>.engine`)
7. `model-opt-yolo eval-trt --output-format onnx_trt --engine …engine --images data/coco/val2017 --annotations data/coco/annotations/instances_val2017.json` (use `ultralytics` or `deepstream_yolo` if that matches your engine; see table below)

CLI details: [docs/cli-reference.md](docs/cli-reference.md) · optional docs site: `pip install -e ".[docs]" && mkdocs serve` ([`mkdocs.yml`](mkdocs.yml))

---

## Supported Output Formats

The **PyTorch → ONNX** step defines tensor names, ranks, and post-processing semantics. **`--output-format`** in `eval-trt` must match that export (and the TensorRT build derived from it); the `.engine` layout alone is not enough if the underlying ONNX was produced differently. Flows discussed here assume ONNX exported with **`--dynamic`**, **`--simplify`**, and **`--opset` 18 or newer** (or equivalent flags in your exporter) so shapes and graphs stay consistent through PTQ and `trtexec`.

`model-opt-yolo eval-trt` scores a **TensorRT `.engine`** on COCO by decoding **how detections leave the network** for your stack. Pass **`--output-format`** accordingly. Full flags and shapes: [`docs/cli-reference.md`](docs/cli-reference.md).

| `--output-format` | Typical source | Role |
|-------------------|----------------|------|
| **`onnx_trt`** | **[levipereira/ultralytics](https://github.com/levipereira/ultralytics)** — `format=onnx_trt` / `onnx_trt.py` (four fixed ONNX outputs; see that repo's detection table). This path is **not** the same as naming the graph "EfficientNMS": some heads are end-to-end in-network, others use EfficientNMS_TRT in the exporter — TensorRT still exposes `num_dets`, `det_boxes`, `det_scores`, `det_classes`. | Read the four tensors, take the first `num_dets` rows, filter by confidence, **undo letterbox**, COCO category mapping, **pycocotools** mAP. **`efficient_nms`** is accepted as an alias (legacy name). |
| **`ultralytics`** | **[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)** TensorRT export with integrated NMS: a **single** output tensor (e.g. `output0`) shaped `[B, N, 6]` (e.g. `N = 300`). | Each row is **`x1, y1, x2, y2, score, class`** in **letterboxed input space** (NMS already applied in the graph). Filter by `--conf-thres`, letterbox inverse, COCO mapping, mAP. |
| **`deepstream_yolo`** | **[marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)** — engines aligned with the **DeepStream custom bbox parser** (`nvdsparsebbox_Yolo`): one output (often named `output`) `[B, num_anchors, 6]` (e.g. **8400** proposals at 640×640). | Same six fields as the parser (**xyxy + score + class**). In DeepStream, clustering/NMS runs in the pipeline; in **`eval-trt`** we apply **per-class NMS** in Python (`--iou-thres`), then letterbox inverse and mAP. |

**Input tensor:** engines may use `images`, `input`, or another name; `eval-trt` binds the **first** input — ensure your build profile matches **NCHW** and the same letterbox normalization as calibration (**÷255**, RGB).

**Batch:** **`B`** may be dynamic in the engine; evaluation uses **`B = 1`** per image.

---

## Technology Stack

| Layer | Choice |
|------|--------|
| **Quantization** | `nvidia-modelopt[onnx]` (GitHub `main` in the image) |
| **Calibration** | ONNX Runtime **GPU** (CUDA **13** nightly, aligned with the image) |
| **Engine** | **TensorRT** **26.02** (NGC `tensorrt:26.02-py3`) |
| **License** | **Apache 2.0** — [LICENSE](LICENSE), [NOTICE](NOTICE) |

---

## Getting Started

### Prerequisites

You need a **machine with an NVIDIA GPU** and software on the host so containers can use CUDA / TensorRT:

| Requirement | Notes |
|-------------|--------|
| **NVIDIA GPU** | A CUDA-capable graphics card (e.g. GeForce / RTX / datacenter GPU). |
| **NVIDIA driver** | Installed on the host; `nvidia-smi` should work **before** you use Docker. |
| **Docker** | [Docker Engine](https://docs.docker.com/engine/install/) installed and running. |
| **NVIDIA Container Toolkit** | Lets `docker run --gpus all` pass the GPU into the container. [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). |

Verify the driver with `nvidia-smi` on the host. After installing the toolkit, follow NVIDIA's guide to confirm GPU access from Docker (e.g. run `nvidia-smi` inside a test container).

### Run with Docker (default)

The **`model-opt-yolo`** package is **installed inside the image** at build time. You do **not** need to mount the Git repository to run — only bind-mount three folders on the host so ONNX, datasets, and outputs persist when the container stops.

#### 1. Build the image (needs the Dockerfile)

Clone once (or copy the `docker/` context elsewhere) and build:

```bash
git clone https://github.com/levipereira/Model-Optimizer-YOLO.git
cd Model-Optimizer-YOLO
docker build -f docker/Dockerfile -t modelopt-yolo-ptq .
```

#### 2. Run with `models/`, `data/`, and `artifacts/` on the host

Pick a root directory on the host (any path you like) and create the three subfolders:

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

Inside the container, the working directory is **`/workspace/model-opt-yolo`**. Use the same **relative** paths as in the docs: `models/...`, `data/coco/...`, `artifacts/...` — they map to `$DATA_ROOT` on the host.

#### Host ↔ container mapping

| Host | Container |
|------|-----------|
| `$DATA_ROOT/models` | `/workspace/model-opt-yolo/models` |
| `$DATA_ROOT/data` | `/workspace/model-opt-yolo/data` |
| `$DATA_ROOT/artifacts` | `/workspace/model-opt-yolo/artifacts` |

Change `DATA_ROOT` to another disk or folder if you want.

See [docs/docker-reference.md](docs/docker-reference.md) for build args and persistence details.

#### Development (edit mode in Docker)

To **develop** using the image: build it, then **bind-mount your Git clone** into `/workspace/model-opt-yolo` so you edit the repo on the host and run inside the container. Step-by-step: **[Edit mode with Docker (developers)](docs/installation.md#edit-mode-with-docker-developers)** in [Installation](docs/installation.md).

### Local Installation (optional)

If you want to change this project and run **outside** Docker, clone the repo, then install in editable mode from the repository root:

```bash
git clone https://github.com/levipereira/Model-Optimizer-YOLO.git
cd Model-Optimizer-YOLO
pip install -e .
model-opt-yolo --help
```

You still need a matching CUDA / TensorRT / ONNX Runtime stack on the host; the Docker image is the supported baseline.

---

## License

Copyright © 2026 [Levi Pereira](mailto:levi.pereira@gmail.com). Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for terms and third-party notices.
