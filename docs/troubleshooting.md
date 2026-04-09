# Troubleshooting

## CUDA / ONNX Runtime

**Symptom:** `libcublas.so.12: cannot open shared object file` or ORT CUDA EP fails.

**Cause:** PyPI **onnxruntime-gpu** is often built for **CUDA 12** while the TensorRT 26 image uses **CUDA 13**.

**Fix:** Use the provided **Dockerfile** (installs ORT from **ort-cuda-13-nightly**), or install that nightly on the host. Alternatively use CPU-only calibration: pass `--calibration_eps cpu` to Model Optimizer (via `quantize` passthrough).

---

## TensorRT / autotune

**Symptom:** Concat / shape errors when building engines during **autotune** (e.g. dynamic spatial dimensions collapsed to 1).

**Fix:** Pass **`--imagesize`** to the autotune wrapper so TRT gets explicit `min/opt/max` shapes. Ensure ONNX input dimensions match real inference sizes.

---

## Model Optimizer autotune CLI missing

**Symptom:** `python -m modelopt.onnx.quantization.autotune` not found from PyPI-only installs.

**Fix:** Install Model Optimizer **from GitHub** (as in the Docker image). PyPI wheels may omit the full autotune entry point.

---

## YOLO graphs and `high_precision_dtype`

**Symptom:** `infer_shapes` / ONNX shape inference errors after PTQ.

**Cause:** Model Optimizer default **`fp16`** post-pass can break dynamic detection heads.

**Fix:** This project defaults **`--high_precision_dtype fp32`** in `model-opt-yolo quantize`. Use `fp16` only if the graph supports it.

---

## Further detail

See the in-repo skill **modelopt-troubleshooting** (`.cursor/skills/modelopt-troubleshooting/SKILL.md`) for extended diagnostics and environment checks.
