---
name: modelopt-troubleshooting
description: >-
  Diagnose and fix common NVIDIA Model Optimizer ONNX issues. Covers CUDA/ORT
  version mismatches, execution provider failures, libcublas/cuDNN errors,
  calibration problems, and Docker environment setup. Use when the user
  encounters errors running quantization, EP failures, library loading issues,
  or modelopt import errors.
---

# Model Optimizer Troubleshooting

## Diagnostic Commands

Run these first to understand the environment:

```bash
# Model Optimizer version
python -c "import modelopt; print(modelopt.__version__)"

# ORT version and available EPs
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"

# Check what CUDA libs ORT was built against
ldd $(python -c "import onnxruntime as ort, os; print(os.path.join(os.path.dirname(ort.__file__), 'capi/libonnxruntime_providers_cuda.so'))") | grep -E 'cublas|cudnn'

# System CUDA toolkit
ls /usr/local/cuda*/lib64/libcublas.so* 2>/dev/null
nvcc --version 2>/dev/null

# LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
```

## Issue: `libcublas.so.12: cannot open shared object file`

**Cause:** ORT was built for CUDA 12 but system only has CUDA 13 libraries.

**Solutions (pick one):**

1. **Use Docker image** (recommended): `nvcr.io/nvidia/tensorrt:26.02-py3` ships aligned CUDA/ORT/TRT.
2. **Install CUDA 12 compat**: `apt install cuda-cublas-12-*` and add to `LD_LIBRARY_PATH`.
3. **Use ORT nightly for CUDA 13**:
   ```bash
   pip uninstall onnxruntime onnxruntime-gpu -y
   pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
   ```
4. **Skip GPU EPs**: Pass `--calibration_eps cpu` (slow but works).

## Issue: TensorrtExecutionProvider EP Error

**Typical error:**
```
EP Error ... Please install TensorRT libraries ... make sure they're in the PATH or LD_LIBRARY_PATH
```

**Solutions:**

1. Verify TensorRT is installed: `python -c "import tensorrt; print(tensorrt.__version__)"`.
2. Ensure `LD_LIBRARY_PATH` includes TRT and cuBLAS directories.
3. If TRT is not needed, exclude it: `--calibration_eps cpu cuda:0` (omit `trt`).

## Issue: cuDNN not found

**Error:** `libcudnn_adv*.so* is not accessible in LD_LIBRARY_PATH`

**Fix:**
```bash
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
```

## Issue: `ImportError: modelopt.onnx`

**Cause:** Missing `[onnx]` extras.

**Fix:**
```bash
pip install -U "nvidia-modelopt[onnx]==0.43.0" --extra-index-url https://pypi.nvidia.com
```

## Issue: Model > 2GB ValueError

**Error:** `Onnx model size larger than 2GB`

**Fix:** Add `--use_external_data_format` flag. Output will have a `.onnx_data` sidecar file.

## Issue: Calibration Data Shape Mismatch

**Symptom:** Assertion error or wrong results during calibration.

**Checklist:**
1. Check model input shape: `onnx.load("model.onnx").graph.input`
2. Verify calib data shape: `np.load("calib.npy").shape`
3. Ensure preprocessing matches export (letterbox, RGB, /255, img_size).

## Issue: Fallback to CPUExecutionProvider

**Symptom:** Quantization is extremely slow; logs show EP fallback.

**Cause:** Both CUDA and TRT EPs failed to initialize.

**Action:** Fix the underlying CUDA/TRT library issue (see above), or accept CPU-only calibration for small models.

## Issue: Autotune -- TRT Concat / shape mismatch

**Symptom:** Logs like `IConcatenationLayer Concat: axis 2 dimensions must be equal`, `Failed to build TensorRT engine`, `latency=inf`, `Exported INT8 model with 0 Q/DQ pairs`.

**Context:** Autotune is triggered via `modelopt-onnx-ptq quantize --autotune <preset>` (or `--autotune` on `pipeline-e2e`). It only applies to **int8** and **fp8** -- int4 ignores the flag.

**Common causes:**

1. **Dynamic input height/width** (`-1` or symbolic dims on the input tensor). The autotuner's TensorRT Python benchmark sets **any `-1` input dim to `1`** when building the optimization profile unless shapes are customized. That shrinks the whole network so feature maps no longer align at FPN/neck **Concat** nodes, and TRT rejects the graph.

   **Check:**
   `python -c "import onnx; m=onnx.load('YOUR.onnx'); print([(i.name,[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"`

   **Fix:** Re-export ONNX with **fixed** input shape (e.g. `1x3x640x640`). Or pass shape hints via `--calibration_shapes` / `--override_shapes` through the quantize passthrough.

2. **Per-region ONNX** during autotune may still fail TRT for some YOLO exports even when the full model builds. Confirm the **full** model builds:
   `trtexec --onnx=YOUR.onnx --stronglyTyped --saveEngine=/tmp/t.plan`
   If the full model fails too, fix the export; if only region exports fail, treat as autotune limitation and quantize without `--autotune`.

3. **Workaround via passthrough:** Pass TRT shape hints through the quantize passthrough args:
   `modelopt-onnx-ptq quantize ... --autotune default -- --autotune_use_trtexec --autotune_trtexec_args '--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640'`
   (adjust tensor name `input` and size to match your graph.)

## Issue: trtexec parse error on `QuantizeLinear` — `convertAxis` / `nbDims (0)` / `axis` 1

**Typical log:**
```
While parsing node ... [QuantizeLinear -> "..."]:
ERROR: importerUtils.cpp:... In function convertAxis:
Assertion failed: (axis >= 0 && axis <= nbDims (0)). Provided axis is: 1
Failed to parse onnx file
```

**Meaning:** TensorRT thinks the **input** to that `QuantizeLinear` is **0-D** (scalar), but the op still carries **`axis` = 1**. Per-channel style Q on a scalar is invalid → parser assert. Cause is usually **quantized ONNX** (Q/DQ layout from Model Optimizer), sometimes exposed only on **new GPU / TRT** builds.

**Try (in order):**

1. Re-quantize with **`--simplify`** passthrough: `modelopt-onnx-ptq quantize ... -- --simplify`
2. Confirm **fixed** input shape on the FP ONNX (no batch `-1` on spatial dims if your stack requires static shapes for TRT).
3. **`build-trt --mode best`** as an experiment (not equivalent to strongly-typed PTQ; use only to see if import path changes).
4. Compare **TensorRT** / **modelopt** versions with a known-good run; upgrade TRT if on a very new architecture (e.g. Blackwell).
5. Inspect the named node in **Netron**; if the graph is valid ONNX, consider reporting upstream (TensorRT or Model Optimizer) with a minimal model.

## Issue: OOM During Calibration

**Fix:** Use `--calibrate_per_node` (int8/fp8 only). This runs inference per node instead of the full graph, reducing peak VRAM.

## Environment Verification Checklist

```
- [ ] CUDA toolkit version matches ORT's build (check with ldd)
- [ ] cuDNN libs in LD_LIBRARY_PATH
- [ ] tensorrt Python package installed and >= 10.0
- [ ] nvidia-modelopt[onnx] matches the supported release (Docker default **0.43.0**); verify with `python -c "import modelopt; print(modelopt.__version__)"`
- [ ] Calibration .npy shape matches model input
- [ ] Docker container if possible (cleanest setup)
```
