# Artifacts & logging

## Root directory

By default, generated files live under **`<current working directory>/artifacts/`**. That directory is **created automatically** on first use (equivalent to `mkdir -p`).

Override the root with an absolute or relative path:

```bash
export MODELOPT_ARTIFACTS_ROOT=/path/to/my_artifacts
```

If the path does not exist, it is created when a command first needs it. Relative values are resolved against the process current working directory at the time `artifacts_root()` runs.

---

## Layout

| Path | Contents |
|------|----------|
| `artifacts/calibration/` | Calibration `.npy` tensors |
| `artifacts/calibration/logs/` | `calib_prep_*.log` session logs |
| `artifacts/quantized/` | Quantized ONNX from `model-opt-yolo quantize` |
| `artifacts/quantized/logs/` | Per-run `quantize_*.log` files |
| `artifacts/trt_engine/` | TensorRT `.engine` files and `.engine.timing.cache` from `model-opt-yolo build-trt` (default output) |
| `artifacts/trt_engine/logs/` | Per-run `build_trt_*.log` and `trt_bench_*.log` |
| `artifacts/predictions/` | COCO prediction JSON from `eval-trt` |
| `artifacts/predictions/logs/` | `eval_*.log` |
| `artifacts/autotune/` | Autotune artifacts (when using `quantize --autotune`) |
| `artifacts/pipeline_e2e/sessions/<session_id>/` | `pipeline-e2e`: `session.json`, `pipeline.log`, session-scoped logs (`trt_engine/logs/`, `predictions/logs/`, `quantized/logs/`) and `e2e_report.md` — `report-runs` reads only these dirs so older global logs are not merged |

---

## Session naming

The Python module `model_opt_yolo.session_paths` generates **unique** directory and file names using **timestamps** and key hyperparameters so repeated runs do not overwrite previous outputs.

Examples:

- Calibration output: `calib_<dir>_sz<640>_n<500>_<timestamp>.npy`
- Quantize log: `quantize_<stem>_qt<int8>_<method>_<timestamp>.log`
- TensorRT bench log: `trt_bench_<engine-stem>_<timestamp>.log` (`trt-bench`)

---

## Logging flags

All main tools support:

- `--log-file PATH` — write a UTF-8 session log
- `-v` / `--verbose` — DEBUG on stderr

If `--log-file` is omitted, a default path under `artifacts/**/logs/` is chosen when applicable.
