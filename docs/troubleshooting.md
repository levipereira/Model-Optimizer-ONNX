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

## Quantized engine slower than FP16 (latency / reformats)

**Symptom:** TensorRT **int8/fp8** engine from PTQ has **higher latency** or **more reformat** / cast-like layers than the **FP16** baseline, even though raw int8 compute is cheaper.

**Why:** Q/DQ placement changes fusion; some ops become **slower** when quantized, and extra **layout/dtype transitions** can dominate.

**What to do:** Follow the iterative loop in [PTQ performance workflow](quantization-performance-workflow.md): **`trt-bench`** and **`trex-analyze`** to find hot layers, then tighten scope with **`quantize --profile`** (YAML rules on the **quantized** graph’s node names / op types) and/or **`quantize --autotune`** for int8. Rebuild the engine after each PTQ change.

---

## TensorRT `trtexec`: `QuantizeLinear` / `convertAxis` / `nbDims (0)`

**Symptom (example):** Parsing fails on a **`QuantizeLinear`** node with:

`Assertion failed: (axis >= 0 && axis <= nbDims (0)). Provided axis is: 1`

**Meaning:** The ONNX parser treated the **input tensor to that `QuantizeLinear`** as **rank 0** (scalar), but the node still has **`axis` = 1** (typical of per-channel Q). That combination is invalid for TensorRT’s importer. This usually comes from **Q/DQ placement** in the quantized ONNX (Model Optimizer), not from `build-trt` flags.

**Things to try:**

1. **Re-run PTQ with graph simplification** (can change Q/DQ layout):
   ```bash
   modelopt-onnx-ptq quantize ... -- --simplify
   ```
2. **Try a different TensorRT / container build** (new GPUs such as Blackwell sometimes hit parser edge cases fixed in later TRT drops).
3. **Inspect the failing node** in [Netron](https://netron.app/) on `*.quant.onnx`: find the `QuantizeLinear` named in the log (e.g. `val_52_1_QuantizeLinear`) and check whether its input is effectively a scalar or mis-inferred.
4. **Experiment** (accuracy may differ from strict PTQ):  
   `modelopt-onnx-ptq build-trt --onnx ... --mode best`  
   If parsing succeeds, the issue is likely **strictly-typed** import of that Q/DQ pattern; treat as a workaround only.
5. If it **used to work** with the **same** MO/TRT stack, compare **`nvidia-modelopt`** and **TensorRT** versions and the **FP ONNX** + **calibration** inputs; file a minimal repro with **NVIDIA Model Optimizer** or **TensorRT** if the graph is valid in ONNX but TRT still rejects it.

---

## Model Optimizer autotune not working

**Symptom:** `--autotune` flag on `quantize` has no effect, or autotune features missing.

**Fix:** Install Model Optimizer **from GitHub** (as in the Docker image). PyPI wheels may omit full autotune support.

---

## YOLO graphs and `high_precision_dtype`

**Symptom:** `infer_shapes` / ONNX shape inference errors after PTQ.

**Cause:** Model Optimizer’s FP16 post-pass for non-quantized ops can break some dynamic detection heads.

**Fix:** `modelopt-onnx-ptq quantize` defaults **`--high_precision_dtype fp16`**. If quantization fails, retry with **`--high_precision_dtype fp32`**.

---

## TREx CLI import errors (`bin` / `utils`)

### `No module named 'bin'`

**Symptom:** **`trex --help`** fails with **`from bin.trex import main`** → **`No module named 'bin'`**.

**Cause:** **`setup.py`** uses **`trex=bin.trex:main`**, but **`bin/`** has no **`__init__.py`**, so **`setuptools`** does not install **`bin`** as a package.

**Fix (Docker):** Rebuild using [docker/Dockerfile](../docker/Dockerfile) — it runs **`touch …/bin/__init__.py`** before **`install.sh`**.

### `No module named 'utils'`

**Symptom:** After fixing **`bin`**, **`trex --help`** fails inside **`bin/trex.py`** on **`import utils.draw_engine`**.

**Cause:** Upstream **`bin/trex.py`** does **`sys.path.append(…/utils)`** but then imports **`utils.draw_engine`**. The import needs the **trt-engine-explorer** root on **`sys.path`** (so **`utils`** is a package under it), not the **`utils`** directory itself.

**Fix (Docker):** The Dockerfile **`sed`**-patches **`bin/trex.py`** to:

```python
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
```

**Fix (running container / manual install):**

```bash
TREX_ROOT=/workspace/TREx/tools/experimental/trt-engine-explorer
touch "${TREX_ROOT}/bin/__init__.py"
sed -i 's|sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "utils"))|sys.path.insert(0, os.path.dirname(SCRIPT_DIR))|' \
  "${TREX_ROOT}/bin/trex.py"
cd "${TREX_ROOT}"
pip install --no-cache-dir -e ".[notebook]"
trex --help
```

---

## `trex-analyze`: EnginePlan / `fillna` on string columns

**Symptom:** `TypeError: Invalid value '0' for dtype 'str'` (or similar) inside `trex.df_preprocessing` when building `EnginePlan` from `trtexec` layer JSON — often with **newer TensorRT** exports or **INT4 / Q/DQ** graphs.

**Cause:** Upstream TREx ends `__fix_columns_types` with `df.fillna(0)` on the **entire** frame. Pandas **string** / Arrow columns cannot be filled with the integer `0`.

**Fix:** **`trex-analyze`** patches `trex.df_preprocessing.__fix_columns_types` at import time so NaNs are filled per dtype (numeric → `0`, string-like → `""`, bool → `False`). Set **`MODELOPT_TREX_NO_DF_PATCH=1`** only to disable the patch and use unmodified TREx (for debugging).

---

## Further detail

See the in-repo Agent Skill **modelopt-troubleshooting** ([`skills/modelopt-troubleshooting/SKILL.md`](../skills/modelopt-troubleshooting/SKILL.md)) for extended diagnostics and environment checks.
