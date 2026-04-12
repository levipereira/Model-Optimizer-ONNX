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

## TensorRT `trtexec`: `QuantizeLinear` / `convertAxis` / `nbDims (0)`

**Symptom (example):** Parsing fails on a **`QuantizeLinear`** node with:

`Assertion failed: (axis >= 0 && axis <= nbDims (0)). Provided axis is: 1`

**Meaning:** The ONNX parser treated the **input tensor to that `QuantizeLinear`** as **rank 0** (scalar), but the node still has **`axis` = 1** (typical of per-channel Q). That combination is invalid for TensorRT’s importer. This usually comes from **Q/DQ placement** in the quantized ONNX (Model Optimizer), not from `build-trt` flags.

**Things to try:**

1. **Re-run PTQ with graph simplification** (can change Q/DQ layout):
   ```bash
   model-opt-yolo quantize ... -- --simplify
   ```
2. **Try a different TensorRT / container build** (new GPUs such as Blackwell sometimes hit parser edge cases fixed in later TRT drops).
3. **Inspect the failing node** in [Netron](https://netron.app/) on `*.quant.onnx`: find the `QuantizeLinear` named in the log (e.g. `val_52_1_QuantizeLinear`) and check whether its input is effectively a scalar or mis-inferred.
4. **Experiment** (accuracy may differ from strict PTQ):  
   `model-opt-yolo build-trt --onnx ... --mode best`  
   If parsing succeeds, the issue is likely **strictly-typed** import of that Q/DQ pattern; treat as a workaround only.
5. If it **used to work** with the **same** MO/TRT stack, compare **`nvidia-modelopt`** and **TensorRT** versions and the **FP ONNX** + **calibration** inputs; file a minimal repro with **NVIDIA Model Optimizer** or **TensorRT** if the graph is valid in ONNX but TRT still rejects it.

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
