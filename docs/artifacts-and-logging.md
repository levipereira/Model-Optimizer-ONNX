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
| `artifacts/quantized/` | Quantized ONNX and often `.engine` / timing cache from `model-opt-yolo build-trt` |
| `artifacts/quantized/logs/` | Per-run `quantize_*.log` files |
| `artifacts/predictions/` | COCO prediction JSON from `eval-trt` |
| `artifacts/predictions/logs/` | `eval_*.log` |
| `artifacts/autotune/` | Autotune runs: `optimized_final.onnx`, `autotuner_state.yaml`, `region_models/`, `logs/` |

---

## Session naming

The Python module `model_opt_yolo.session_paths` generates **unique** directory and file names using **timestamps** and key hyperparameters so repeated runs do not overwrite previous outputs.

Examples:

- Calibration output: `calib_<dir>_sz<640>_n<500>_<timestamp>.npy`
- Autotune run folder: `autotune_<model>_qt<int8>_spr<N>_img<640>_<timestamp>/`
- Quantize log: `quantize_<stem>_qt<int8>_<method>_<timestamp>.log`

---

## Logging flags

All main tools support:

- `--log-file PATH` — write a UTF-8 session log
- `-v` / `--verbose` — DEBUG on stderr

If `--log-file` is omitted, a default path under `artifacts/**/logs/` is chosen when applicable.
