# Artifacts & logging

## Root directory

By default, generated files live under **`<current working directory>/artifacts/`**. That directory is **created automatically** on first use (equivalent to `mkdir -p`).

Override the root with an absolute or relative path:

```bash
export MODELOPT_ARTIFACTS_ROOT=/path/to/my_artifacts
```

If the path does not exist, it is created when a command first needs it. Relative values are resolved against the process current working directory at the time `artifacts_root()` runs.

**Session id (optional):** If you set **`export SESSION_ID=my-run-1`**, commands that support **`--session-id`** (`pipeline-e2e`, `build-trt`, `eval-trt`, `trt-bench`, `report-runs`, **`trex-analyze`**) use that value automatically when the flag is omitted. A **`--session-id`** on the command line overrides **`SESSION_ID`**.

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
| `artifacts/trex/runs/<name>/` | **`trex-analyze`**: `mode__<mode>/` or `primary/` + `compare/` when **`--compare --compare-onnx`**; `trex_analyze.log`, TREx JSON, optional plan graph if **`--graph`**, optional `compare_layers__*.csv` |
| `artifacts/pipeline_e2e/sessions/<id>/trex/` | Session-scoped **`trex-analyze`** runs (same layout under each session) |
| `artifacts/predictions/` | COCO prediction JSON from `eval-trt` |
| `artifacts/predictions/logs/` | `eval_*.log` |
| `artifacts/autotune/` | Autotune artifacts (when using `quantize --autotune`) |
| `artifacts/pipeline_e2e/sessions/<session_id>/` | `pipeline-e2e`: `session.json`, `pipeline.log`, session-scoped logs (`trt_engine/logs/`, `predictions/logs/`, `quantized/logs/`), FP16 baseline + PTQ run logs, and `report_<session_id>.md` (plus `chart_ips_latency_<session_id>.png`, `chart_eval_<session_id>.png`). **`report-runs`** must use **`--session-id <id>`** (or explicit paths to those session `…/logs` folders) to aggregate them; the default `report-runs` without `--session-id` scans only the **global** `artifacts/trt_engine/logs` and `artifacts/predictions/logs`, which usually **do not** contain the full `pipeline-e2e` run. `pipeline-e2e` calls `report-runs` with `--session-id` and **`--merge-global-logs`** so metrics from session + global are merged when needed. |

**Manual runs with the same session:** `model-opt-yolo build-trt`, `eval-trt`, and `trt-bench` accept **`--session-id <session_id>`** (when **`--log-file`** is omitted). Default logs are written under the same `artifacts/pipeline_e2e/sessions/<session_id>/…` tree as `pipeline-e2e`. Then **`model-opt-yolo report-runs --session-id <session_id>`** picks them up without passing `--trt-logs-dir` / `--eval-logs-dir`.

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
