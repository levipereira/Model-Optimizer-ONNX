# Model-Optimizer-ONNX — Documentation

**modelopt-onnx-ptq** is a command-line toolkit for **ONNX post-training quantization (PTQ)** of object-detection models, built around [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), **TensorRT**, and **ONNX Runtime** — with COCO-based calibration and optional Q/DQ autotune.

---

## Contents

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | `pip install`, Docker image, ORT CUDA 13 alignment |
| [Workflow](workflow.md) | Pipeline steps, **autotune** (`quantize --autotune`), **`pipeline-e2e`** / `--quant-matrix` |
| [PTQ performance workflow](quantization-performance-workflow.md) | Iterative loop + **`pipeline-e2e` / `--quant-matrix`** to compare modes; **[`skills/ptq-trt-performance/SKILL.md`](../skills/ptq-trt-performance/SKILL.md)** for the measurement checklist |
| [YOLO26n end-to-end PTQ workflow](yolo26n-end-to-end-ptq-workflow.md) | Prerequisites (Ultralytics or DeepStream-Yolo export, `download-coco`, ONNX in `models/`); then baseline **`pipeline-e2e`** → **`trex-analyze`** → YAML profile → **`pipeline-e2e` + `--quantize-profile`**; charts + session report link |
| [CLI reference](cli-reference.md) | `modelopt-onnx-ptq` subcommands (**`eval-trt`** **`--output-format`** / **`auto`** + **`--onnx`**; **`trex-analyze`** — at most one of **`--graph`** \| **`--report`** \| **`--compare`**) |
| [Artifacts & logging](artifacts-and-logging.md) | `artifacts/` layout, session naming, log files |
| [Docker reference](docker-reference.md) | Base image, build args, environment variables, optional **TREx** (`/workspace/TREx`, engine profiling) |
| [Troubleshooting](troubleshooting.md) | Common errors (CUDA/ORT, TRT, dynamic shapes) |
| [License & attribution](license-and-attribution.md) | Apache 2.0, third-party components |

---

## Technology stack (summary)

| Component | Role in this project |
|-----------|----------------------|
| **NVIDIA Model Optimizer** (`nvidia-modelopt[onnx]`) | PTQ API and optional ONNX Q/DQ autotune |
| **TensorRT** (NGC container `26.02-py3`) | Engine build (`build-trt`), plan benchmark (`trt-bench`), autotune latency measurement |
| **ONNX Runtime GPU** (CUDA 13 nightly when in TRT 26 image) | Calibration execution providers |
| **PyCUDA** / **TensorRT Python** | `eval-trt` engine inference |
| **pycocotools** | COCO mAP evaluation |
| **OpenCV / NumPy / PyTorch** | Image preprocessing and tensors |

Pinned versions for the **recommended environment** match [`docker/Dockerfile`](../docker/Dockerfile): TensorRT **26.02** Python 3 image, Model Optimizer **`nvidia-modelopt[onnx]`** from **NVIDIA PyPI** (`MODELOPT_VERSION`, default `0.43.0`), ORT from **ort-cuda-13-nightly**.

---

## Quick links

- [README](../README.md) (repository overview and badges)
- [Model Optimizer — ONNX PTQ example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)
- [ONNX quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)
