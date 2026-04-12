# Model-Optimizer-YOLO — Documentation

**model-opt-yolo** is a command-line toolkit for **ONNX post-training quantization (PTQ)** of YOLO-style object detectors, built around [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), **TensorRT**, and **ONNX Runtime** — with COCO-based calibration and optional Q/DQ autotune.

---

## Contents

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | `pip install`, Docker image, ORT CUDA 13 alignment |
| [Workflow](workflow.md) | Pipeline steps, **autotune** (`quantize --autotune`), **`pipeline-e2e`** / `--quant-matrix` |
| [CLI reference](cli-reference.md) | `model-opt-yolo` subcommands (**`eval-trt`** **`--output-format`**; **`trex-analyze`** for TREx profile/graph/compare) |
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

Pinned versions for the **recommended environment** match [`docker/Dockerfile`](../docker/Dockerfile): TensorRT **26.02** Python 3 image, Model Optimizer from **GitHub** (`MODELOPT_GIT_REF`, default `main`), ORT from **ort-cuda-13-nightly**.

---

## Quick links

- [README](../README.md) (repository overview and badges)
- [Model Optimizer — ONNX PTQ example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/onnx_ptq)
- [ONNX quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html)
