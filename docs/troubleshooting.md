# Troubleshooting

## CUDA / ONNX Runtime

**Symptom:** `libcublas.so.12: cannot open shared object file` or ORT CUDA EP fails.

**Cause:** PyPI **onnxruntime-gpu** is often built for **CUDA 12** while the TensorRT 26 image uses **CUDA 13**.

**Fix:** Use the provided **Dockerfile** (installs ORT from **ort-cuda-13-nightly**), or install that nightly on the host. Alternatively use CPU-only calibration: pass `--calibration_eps cpu` to Model Optimizer (via `quantize` passthrough).

---

## TensorRT / autotune

**Symptom:** Concat / shape errors when building engines during autotune (e.g. dynamic spatial dimensions collapsed to 1).

**Fix:** Ensure ONNX input dimensions match real inference sizes. When using `quantize --autotune`, Model Optimizer handles shape profiles internally via the `--calibration_shapes` / `--override_shapes` pass-through flags if needed.

---

## Model Optimizer autotune not working

**Symptom:** `--autotune` flag on `quantize` has no effect, or autotune features missing.

**Fix:** Install Model Optimizer **from GitHub** (as in the Docker image). PyPI wheels may omit full autotune support.

---

## YOLO graphs and `high_precision_dtype`

**Symptom:** `infer_shapes` / ONNX shape inference errors after PTQ.

**Cause:** Model Optimizer default **`fp16`** post-pass can break dynamic detection heads.

**Fix:** This project defaults **`--high_precision_dtype fp32`** in `model-opt-yolo quantize`. Use `fp16` only if the graph supports it.

---

## Further detail

See the in-repo skill **modelopt-troubleshooting** (`.cursor/skills/modelopt-troubleshooting/SKILL.md`) for extended diagnostics and environment checks.
