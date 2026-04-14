# ONNX PTQ API Reference

## `modelopt.onnx.quantization.quantize()` — Full Signature

```python
def quantize(
    onnx_path: str,
    quantize_mode: str = "int8",
    calibration_data: np.ndarray | dict[str, np.ndarray] | None = None,
    calibration_method: str | None = None,
    calibration_cache_path: str | None = None,
    calibration_data_reader: CalibrationDataReader | None = None,
    calibration_shapes: str | None = None,
    calibration_eps: list[str] = ["cpu", "cuda:0", "trt"],
    override_shapes: str | None = None,
    op_types_to_quantize: list[str] | None = None,
    op_types_to_exclude: list[str] | None = None,
    op_types_to_exclude_fp16: list[str] | None = None,
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
    use_external_data_format: bool = False,
    keep_intermediate_files: bool = False,
    output_path: str | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    trt_plugins: list[str] | None = None,
    trt_plugins_precision: list[str] | None = None,
    high_precision_dtype: str = "fp16",
    mha_accumulation_dtype: str = "fp16",
    disable_mha_qdq: bool = False,
    dq_only: bool = False,
    block_size: int | None = None,
    use_zero_point: bool = False,
    passes: list[str] = ["concat_elimination"],
    simplify: bool = False,
    calibrate_per_node: bool = False,
    input_shapes_profile: Sequence[dict[str, str]] | None = None,
    direct_io_types: bool = False,
    opset: int | None = None,
    autotune: bool = False,
    **kwargs,
) -> None
```

## CLI Flags (`python -m modelopt.onnx.quantization`)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--onnx_path` | str | required | Input ONNX path |
| `--quantize_mode` | str | `int8` | `fp8`, `int8`, `int4` |
| `--calibration_method` | str | auto | `max`, `entropy`, `awq_clip`, `rtn_dq` |
| `--calibration_data_path` | str | None | `.npy` or `.npz` calibration data |
| `--calibration_cache_path` | str | None | TRT-style cache file |
| `--calibration_eps` | list | `cpu cuda:0 trt` | Execution provider priority |
| `--calibration_shapes` | str | None | `input0:1x3x256x256,input1:1x3x128x128` |
| `--override_shapes` | str | None | Force static input shapes |
| `--op_types_to_quantize` | list | None | Node types to quantize |
| `--op_types_to_exclude` | list | None | Node types to skip |
| `--nodes_to_quantize` | list | None | Node names (regex supported) |
| `--nodes_to_exclude` | list | None | Node names to skip (regex) |
| `--use_external_data_format` | flag | False | Models > 2GB |
| `--keep_intermediate_files` | flag | False | Keep temp files |
| `--output_path` | str | auto | Output ONNX path |
| `--trt_plugins` | list | None | `.so` plugin paths |
| `--high_precision_dtype` | str | `fp16` | `fp32`, `fp16`, `bf16` |
| `--calibrate_per_node` | flag | False | Reduce memory (int8/fp8 only) |
| `--simplify` | flag | False | Run onnxsim first |
| `--direct_io_types` | flag | False | Lower I/O precision |
| `--trust_calibration_data` | flag | False | Allow pickle in `.npy` |
| `--log_level` | str | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Calibration Data Classes (internal)

| Class | Source | Use Case |
|-------|--------|----------|
| `CalibrationDataProvider` | `calib_utils.py` | User-provided numpy data |
| `RandomDataProvider` | `calib_utils.py` | Auto-generated random data (no calib file) |

## Opset Requirements

| Precision | Minimum Opset |
|-----------|---------------|
| int8, fp8 | 19 |
| int4, uint4 | 21 |
| nvfp4 (float4_e2m1fn) | 23 |

## Execution Provider Resolution (`ort_utils.py`)

`_prepare_ep_list(["cpu", "cuda:0", "trt"])` produces:
```python
["CPUExecutionProvider", ("CUDAExecutionProvider", {"device_id": 0}), "TensorrtExecutionProvider"]
```

Checks performed: TensorRT >= 10.0 import, cuDNN in `LD_LIBRARY_PATH`, libcublas matching ORT's CUDA version.

## AutoCast API (Mixed Precision)

```python
from modelopt.onnx.autocast import convert_to_mixed_precision, convert_to_f16

model = convert_to_mixed_precision(
    onnx_path="model.onnx",
    low_precision_type="fp16",
    providers=["cpu"],
)

model = convert_to_f16(
    model=onnx.load("model.onnx"),
    keep_io_types=True,
)
```
