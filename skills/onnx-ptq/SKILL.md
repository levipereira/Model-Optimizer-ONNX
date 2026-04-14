---
name: onnx-ptq
description: >-
  Run ONNX post-training quantization (PTQ) using NVIDIA Model Optimizer.
  Covers calibration data prep, int8/fp8/int4 quantization, execution provider
  setup, and TensorRT deployment. Use when the user mentions ONNX quantization,
  PTQ, calibration, Model Optimizer, modelopt, int8, fp8, int4, TensorRT,
  model-opt-yolo quantize, or model-opt-yolo.
---

# ONNX Post-Training Quantization (PTQ)

## Workflow Overview

```
Task Progress:
- [ ] Step 1: Environment setup (Docker or local)
- [ ] Step 2: Prepare ONNX model
- [ ] Step 3: Build calibration data
- [ ] Step 4: Run quantization (with optional --autotune for int8/fp8)
- [ ] Step 5: Validate output
- [ ] Step 6: Deploy with TensorRT
```

## Step 1: Environment Setup

**Preferred: Docker**

```bash
docker build -f docker/Dockerfile -t modelopt-yolo-ptq .
export DATA_ROOT="$HOME/model-opt-yolo"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/data" "$DATA_ROOT/artifacts"
docker run --gpus all --rm -it -w /workspace/model-opt-yolo \
  -v "$DATA_ROOT/models:/workspace/model-opt-yolo/models" \
  -v "$DATA_ROOT/data:/workspace/model-opt-yolo/data" \
  -v "$DATA_ROOT/artifacts:/workspace/model-opt-yolo/artifacts" \
  modelopt-yolo-ptq
```

**Local: ensure matching CUDA/ORT versions**

```bash
conda create -n modelopt python=3.12 pip && conda activate modelopt
pip install -U "nvidia-modelopt[onnx]"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
```

Verify:

```bash
python -c "import modelopt.onnx.quantization; print('OK')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Step 2: Prepare ONNX Model

Place `.onnx` files under `models/`. Verify input spec:

```python
import onnx
model = onnx.load("models/yolov8n.onnx")
for inp in model.graph.input:
    print(inp.name, [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim])
```

## Step 3: Build Calibration Data

```bash
pip install -e .   # once: installs the `model-opt-yolo` command
model-opt-yolo calib \
  --images_dir=data/coco/val2017 \
  --calibration_data_size=500 \
  --img_size=640
```

Preprocessing must match the model's export conventions (letterbox, RGB, /255).

## Step 4: Run Quantization

**Via project wrapper:**

```bash
# Without autotune
model-opt-yolo quantize \
  --calibration_data=artifacts/calibration/calib_coco.npy \
  --onnx_glob="models/*.onnx" \
  --quantize_mode=int8 \
  --calibration_method=entropy

# With autotune (int8/fp8 only — ignored for int4)
model-opt-yolo quantize \
  --calibration_data=artifacts/calibration/calib_coco.npy \
  --onnx_path=models/yolov8n.onnx \
  --quantize_mode=int8 \
  --calibration_method=entropy \
  --autotune default
```

**Via modelopt CLI directly:**

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=models/yolov8n.onnx \
  --quantize_mode=int8 \
  --calibration_data_path=artifacts/calibration/calib_coco.npy \
  --calibration_method=entropy \
  --output_path=artifacts/quantized/yolov8n.int8.entropy.quant.onnx \
  --autotune=default
```

**Via Python API:**

```python
import numpy as np
from modelopt.onnx.quantization import quantize

calib = np.load("artifacts/calibration/calib_coco.npy")
quantize(
    onnx_path="models/yolov8n.onnx",
    quantize_mode="int8",
    calibration_data=calib,
    calibration_method="entropy",
    calibration_eps=["cpu", "cuda:0"],
    output_path="artifacts/quantized/yolov8n.int8.entropy.quant.onnx",
    autotune=True,  # only effective for int8/fp8
)
```

**End-to-end pipeline (all 6 combos + autotune):**

```bash
model-opt-yolo pipeline-e2e \
  --onnx models/yolo.onnx \
  --quant-matrix all \
  --autotune default \
  --continue-on-error
```

This runs int8.entropy, int8.max, fp8.entropy, fp8.max, int4.awq_clip, int4.rtn_dq. Autotune is applied to the 4 int8/fp8 combos; the 2 int4 combos proceed without it (modelopt ignores the flag for int4).

### Mode and Method Reference

| Mode | Methods | Autotune | Notes |
|------|---------|----------|-------|
| `int8` | `entropy` (default), `max` | **Yes** | Standard CNN/YOLO quantization |
| `fp8` | `entropy` (default), `max` | **Yes** | Requires SM >= 8.9 |
| `int4` | `awq_clip` (default), `rtn_dq` | **No** | Weight-only; fewer calib samples (~64) |

### Autotune Presets

| Preset | Behavior | Use case |
|--------|----------|----------|
| `quick` | Fewer schemes/region, fewer benchmark runs | Fast exploration |
| `default` | Balanced | Recommended for most models |
| `extensive` | More schemes, more runs | Production tuning |

### Useful Extra Flags

| Flag | Effect |
|------|--------|
| `--autotune <preset>` | Q/DQ placement optimization (int8/fp8 only) |
| `--calibrate_per_node` | Reduces VRAM during calibration |
| `--use_external_data_format` | Required for models > 2GB |
| `--simplify` | Runs onnxsim before quantization |
| `--calibration_eps cpu` | Force CPU-only (skip GPU EPs) |
| `--high_precision_dtype` | Default **`fp16`** in `model-opt-yolo quantize` for non-quantized ops; use **`fp32`** if shape inference fails |

## Step 5: Validate Output

```python
import onnx
model = onnx.load("artifacts/quantized/yolov8n.int8.entropy.quant.onnx")
onnx.checker.check_model(model)
print(f"Nodes: {len(model.graph.node)}")
qdq = [n for n in model.graph.node if "Quantize" in n.op_type or "Dequantize" in n.op_type]
print(f"Q/DQ nodes: {len(qdq)}")
```

## Step 6: Deploy with TensorRT

Prefer **`model-opt-yolo build-trt`** so shapes and logs stay consistent:

```bash
model-opt-yolo build-trt --onnx artifacts/quantized/yolov8n.int8.entropy.quant.onnx --img-size 640
# default --mode is strongly-typed for PTQ ONNX; for non-quantized FP ONNX use --mode best or fp16-int8 (see CLI docs)
```

**Modes (see project `docs/cli-reference.md`):** **`strongly-typed`** (`--stronglyTyped`, default) matches quantized PTQ Q/DQ types. **`best`** lets TensorRT search tactics/precisions—often useful for **non-quantized** FP ONNX. **`fp16`** targets **non-quantized** FP ONNX; **`fp16-int8`** is a common TensorRT combo for FP checkpoints.

The wrapper runs `trtexec` with `--stronglyTyped` by default for quantized exports; override with `--mode best` if you need to experiment.

## Additional Resources

- For API parameter details, see [reference.md](reference.md)
- Upstream example: NVIDIA Model-Optimizer `examples/onnx_ptq` on GitHub
- Model Optimizer docs: https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html
