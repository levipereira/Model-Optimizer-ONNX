"""Scan trt_bench / eval-trt log files and emit a Markdown comparison report with PNG charts."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from model_opt_yolo.session_paths import (
    artifacts_root,
    effective_session_id,
    pipeline_e2e_session_eval_logs,
    pipeline_e2e_session_root,
    pipeline_e2e_session_trt_logs,
)


# trt_bench_yolo26n_marcos_luciano.fp8.entropy.quant_20260410-162441.log
RE_TRT_BENCH = re.compile(r"^trt_bench_(.+)_(\d{8}-\d{6})\.log$")
# eval_yolo26n_marcos_luciano.fp8.entropy.quant_20260410-162355.log
RE_EVAL = re.compile(r"^eval_(.+)_(\d{8}-\d{6})\.log$")

RE_THROUGHPUT = re.compile(r"Throughput:\s*([\d.]+)\s*qps", re.IGNORECASE)
# trtexec "[I] Latency: min = ..." summary line (avoid matching "H2D Latency:" / "D2H Latency:")
RE_LATENCY_SUMMARY = re.compile(
    r"\[I\]\s+Latency:\s*min\s*=\s*[\d.]+\s*ms,\s*max\s*=\s*[\d.]+\s*ms,\s*mean\s*=\s*([\d.]+)\s*ms",
    re.IGNORECASE,
)
# Prefer the Performance summary block (last match often enough)
RE_MAP5095 = re.compile(r"mAP@0\.5:0\.95\s*=\s*([\d.]+)")
RE_MAP50 = re.compile(r"mAP@0\.5\s*=\s*([\d.]+)")
RE_SPEED = re.compile(
    r"Speed per image:\s*preprocess\s*([\d.]+)ms,\s*inference\s*([\d.]+)ms,\s*postprocess\s*([\d.]+)ms"
)
RE_DEVICE = re.compile(r"Selected Device:\s*(.+)$", re.MULTILINE)
RE_TRT_VER = re.compile(r"TensorRT version:\s*([\d.]+)", re.MULTILINE)
RE_CUDA_BANNER = re.compile(r"CUDA Version:\s*([\d.]+)", re.IGNORECASE)

KNOWN_PREC = ("fp8", "int8", "int4", "fp16", "int16")
KNOWN_CALIB = ("entropy", "max", "awq_clip", "rtn_dq", "minmax", "percentile")


@dataclass
class TrtBenchRecord:
    path: Path
    config_key: str
    run_ts: str
    throughput_qps: float | None = None
    latency_mean_ms: float | None = None
    raw_device: str | None = None
    raw_trt_version: str | None = None


@dataclass
class EvalRecord:
    path: Path
    config_key: str
    run_ts: str
    map_5095: float | None = None
    map_50: float | None = None
    infer_ms: float | None = None
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None


@dataclass
class ConfigMeta:
    precision: str | None = None
    calibrator: str | None = None
    extra_tokens: list[str] = field(default_factory=list)


def _parse_config_meta(config_key: str) -> ConfigMeta:
    parts = config_key.split(".")
    meta = ConfigMeta()
    for p in parts:
        pl = p.lower()
        if pl in KNOWN_PREC:
            meta.precision = pl
        elif pl in KNOWN_CALIB:
            meta.calibrator = pl
        elif pl not in ("quant",) and not meta.extra_tokens:
            meta.extra_tokens.append(p)
    return meta


def _ts_key(ts: str) -> tuple[int, int]:
    """Sortable (date, time) from YYYYMMDD-HHMMSS."""
    m = re.match(r"^(\d{8})-(\d{6})$", ts)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _parse_trt_bench(path: Path) -> tuple[float | None, float | None, str | None, str | None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tp = None
    for m in RE_THROUGHPUT.finditer(text):
        tp = float(m.group(1))
    lat = None
    for m in RE_LATENCY_SUMMARY.finditer(text):
        lat = float(m.group(1))
    dev_m = RE_DEVICE.search(text)
    trt_m = RE_TRT_VER.search(text)
    return tp, lat, (dev_m.group(1).strip() if dev_m else None), (
        trt_m.group(1) if trt_m else None
    )


def _parse_eval(path: Path) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m5095 = None
    for m in RE_MAP5095.finditer(text):
        m5095 = float(m.group(1))
    m50 = None
    for m in RE_MAP50.finditer(text):
        m50 = float(m.group(1))
    pre = inf = post = None
    sm = RE_SPEED.search(text)
    if sm:
        pre = float(sm.group(1))
        inf = float(sm.group(2))
        post = float(sm.group(3))
    return m5095, m50, pre, inf, post


def _collect_latest_dirs(
    directories: list[Path], pattern_glob: str, regex: re.Pattern[str]
) -> dict[str, tuple[Path, str, str]]:
    """config_key -> (path, run_ts, fname). Keeps newest run_ts per key across dirs."""
    latest: dict[str, tuple[Path, str, str]] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(pattern_glob)):
            m = regex.match(path.name)
            if not m:
                continue
            key, ts = m.group(1), m.group(2)
            if key not in latest:
                latest[key] = (path, ts, path.name)
            else:
                if _ts_key(ts) > _ts_key(latest[key][1]):
                    latest[key] = (path, ts, path.name)
    return latest


def _short_label(key: str, max_len: int = 28) -> str:
    s = key.replace("yolo26n_marcos_luciano.", "").replace("_", " ")
    if len(s) > max_len:
        return s[: max_len - 2] + "…"
    return s


def _write_bar_chart_png(
    out_png: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
) -> None:
    """Render a bar chart to PNG (non-interactive backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    x = range(len(labels))
    ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ymax = max(values) if values else 1.0
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
    fig.tight_layout()
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)


def _combined_score(map5095: float, qps: float) -> float:
    """Higher is better: balance accuracy and throughput (geometric-style)."""
    if map5095 <= 0 or qps <= 0:
        return 0.0
    return math.sqrt(map5095 * (qps / 1000.0))


def _dist_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _tensorrt_python_version() -> str | None:
    try:
        import tensorrt as trt  # type: ignore[import-untyped]

        return str(getattr(trt, "__version__", None) or "")
    except Exception:
        return None


def _torch_cuda_versions() -> tuple[str | None, str | None]:
    try:
        import torch

        tv = str(torch.__version__)
        cv = getattr(torch.version, "cuda", None)
        return tv, str(cv) if cv else None
    except Exception:
        return None, None


def _nvidia_smi_bin() -> str | None:
    return shutil.which("nvidia-smi")


def _nvidia_smi_cuda_banner() -> str | None:
    """CUDA version as reported by the driver (header of ``nvidia-smi``)."""
    nsmi = _nvidia_smi_bin()
    if not nsmi:
        return None
    try:
        p = subprocess.run(
            [nsmi],
            capture_output=True,
            text=True,
            timeout=25,
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = RE_CUDA_BANNER.search(p.stdout or "")
    return m.group(1) if m else None


def _nvidia_smi_gpu_rows() -> list[dict[str, str]]:
    """Per-GPU fields from ``nvidia-smi --query-gpu`` (best effort)."""
    nsmi = _nvidia_smi_bin()
    if not nsmi:
        return []
    field_sets = (
        "name,memory.total,memory.free,driver_version,compute_cap,sm_count",
        "name,memory.total,memory.free,driver_version,compute_cap",
        "name,memory.total,driver_version",
    )
    for fields in field_sets:
        try:
            p = subprocess.run(
                [
                    nsmi,
                    f"--query-gpu={fields}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=25,
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if p.returncode != 0 or not (p.stdout or "").strip():
            continue
        header = [h.strip() for h in fields.split(",")]
        rows: list[dict[str, str]] = []
        for line in (p.stdout or "").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                reader = csv.reader(io.StringIO(line))
                parts = next(reader)
            except StopIteration:
                continue
            if len(parts) < len(header):
                continue
            row: dict[str, str] = {}
            for i, key in enumerate(header):
                row[key] = parts[i].strip() if i < len(parts) else ""
            rows.append(row)
        if rows:
            return rows
    return []


def _escape_md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _environment_table_lines(
    *,
    tensorrt_from_log: str | None,
    device_from_log: str | None,
) -> list[str]:
    """Markdown lines for ## Environment & versions."""
    lines: list[str] = []
    lines.append("## Environment & versions")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")

    moy = _dist_version("model-opt-yolo")
    lines.append(
        f"| model-opt-yolo | `{_escape_md_cell(moy or 'unknown')}` |"
    )

    mo = _dist_version("nvidia-modelopt", "modelopt")
    lines.append(
        f"| NVIDIA Model Optimizer (pip) | `{_escape_md_cell(mo or '—')}` |"
    )

    trt_py = _tensorrt_python_version()
    lines.append(
        f"| TensorRT (Python bindings) | `{_escape_md_cell(trt_py or '—')}` |"
    )
    if tensorrt_from_log:
        lines.append(
            f"| TensorRT (trtexec log) | `{_escape_md_cell(tensorrt_from_log)}` |"
        )
    elif trt_py:
        lines.append("| TensorRT (trtexec log) | *not found in scanned logs* |")
    else:
        lines.append("| TensorRT (trtexec log) | — |")

    tv, tcuda = _torch_cuda_versions()
    lines.append(
        f"| PyTorch | `{_escape_md_cell(tv or '—')}` |"
    )
    lines.append(
        f"| PyTorch CUDA build | `{_escape_md_cell(tcuda or '—')}` |"
    )

    cuda_drv = _nvidia_smi_cuda_banner()
    lines.append(
        f"| CUDA (driver / `nvidia-smi`) | `{_escape_md_cell(cuda_drv or '—')}` |"
    )

    gpu_rows = _nvidia_smi_gpu_rows()
    if not gpu_rows:
        lines.append("| GPU (nvidia-smi) | *not available* |")
        lines.append("| Driver | — |")
        lines.append("| GPU memory | — |")
        lines.append("| Compute capability | — |")
        lines.append("| SM count | — |")
    else:
        for i, g in enumerate(gpu_rows):
            idx = f" [{i}]" if len(gpu_rows) > 1 else ""
            name = g.get("name", "—")
            mem = g.get("memory.total", "—")
            if mem and mem != "—":
                mem = f"{mem} MiB"
            drv = g.get("driver_version", "—")
            cc = g.get("compute_cap", "—")
            sm = g.get("sm_count", "—")
            mfree = g.get("memory.free", "")
            mem_note = mem
            if mfree and mfree != "—":
                mem_note = f"{mem} (free {_escape_md_cell(mfree)} MiB)" if mem != "—" else f"free {mfree} MiB"
            lines.append(
                f"| GPU{idx} | `{_escape_md_cell(name)}` |"
            )
            lines.append(f"| Driver{idx} | `{_escape_md_cell(drv)}` |")
            lines.append(f"| GPU memory{idx} | `{_escape_md_cell(mem_note)}` |")
            lines.append(
                f"| Compute capability{idx} | `{_escape_md_cell(cc)}` |"
            )
            lines.append(f"| SM count{idx} | `{_escape_md_cell(sm)}` |")

    if device_from_log:
        lines.append(
            f"| GPU (from TRT bench log) | `{_escape_md_cell(device_from_log)}` |"
        )

    lines.append("")
    lines.append(
        "*CUDA (driver): maximum CUDA version supported by the installed NVIDIA driver "
        "(from the `nvidia-smi` banner). "
        "SM count is only filled when your driver's `nvidia-smi --query-gpu` supports it; "
        "otherwise rely on compute capability.*"
    )
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Read trt_bench_*.log and eval_*.log, merge by ONNX/engine stem, "
            "and write a Markdown report with tables and PNG bar charts."
        )
    )
    _default_trt = artifacts_root() / "trt_engine" / "logs"
    _default_eval = artifacts_root() / "predictions" / "logs"
    p.add_argument(
        "--session-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Use logs under artifacts/pipeline_e2e/sessions/<id>/ (same layout as pipeline-e2e). "
            "Sets --trt-logs-dir and --eval-logs-dir unless you pass those explicitly; "
            "sets -o to <session>/e2e_report.md unless -o is set. "
            "If omitted, SESSION_ID environment variable is used when set (CLI wins). "
            "Without --session-id and without explicit log dirs, defaults scan **global** "
            "artifacts/trt_engine/logs (not the session folder) — pipeline outputs are often only under sessions/<id>/."
        ),
    )
    p.add_argument(
        "--trt-logs-dir",
        type=Path,
        default=None,
        help=f"Directory containing trt_bench_*.log files (default: {_default_trt})",
    )
    p.add_argument(
        "--eval-logs-dir",
        type=Path,
        default=None,
        help=f"Directory containing eval_*.log files (default: {_default_eval})",
    )
    p.add_argument(
        "--merge-global-logs",
        action="store_true",
        help=(
            "Also scan global artifacts/trt_engine/logs and artifacts/predictions/logs "
            "(in addition to --trt-logs-dir / --eval-logs-dir) and merge by config key "
            "(newest timestamp wins). Use after pipeline-e2e if logs were split between "
            "session folders and the default flat logs dir."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output .md path (default: artifacts/reports/trt_eval_report_<timestamp>.md, or "
        "<session>/e2e_report.md with --session-id)",
    )
    args = p.parse_args(argv)

    session_id = effective_session_id(args.session_id) or ""
    if session_id:
        trt_logs_dir = args.trt_logs_dir or pipeline_e2e_session_trt_logs(session_id)
        eval_logs_dir = args.eval_logs_dir or pipeline_e2e_session_eval_logs(session_id)
    else:
        trt_logs_dir = args.trt_logs_dir or _default_trt
        eval_logs_dir = args.eval_logs_dir or _default_eval
    args.trt_logs_dir = trt_logs_dir
    args.eval_logs_dir = eval_logs_dir

    out_arg = args.output
    if session_id and args.output is None:
        out_arg = pipeline_e2e_session_root(session_id) / "e2e_report.md"
    args.output = out_arg

    trt_dirs: list[Path] = [Path(args.trt_logs_dir).resolve()]
    eval_dirs: list[Path] = [Path(args.eval_logs_dir).resolve()]
    if args.merge_global_logs:
        g_trt = (artifacts_root() / "trt_engine" / "logs").resolve()
        g_ev = (artifacts_root() / "predictions" / "logs").resolve()
        if g_trt not in trt_dirs:
            trt_dirs.append(g_trt)
        if g_ev not in eval_dirs:
            eval_dirs.append(g_ev)

    trt_latest = _collect_latest_dirs(trt_dirs, "trt_bench_*.log", RE_TRT_BENCH)
    eval_latest = _collect_latest_dirs(eval_dirs, "eval_*.log", RE_EVAL)

    all_keys = sorted(set(trt_latest) | set(eval_latest))

    rows: list[dict[str, object]] = []
    trt_by_key: dict[str, TrtBenchRecord] = {}
    eval_by_key: dict[str, EvalRecord] = {}

    for key in all_keys:
        if key in trt_latest:
            path, ts, _ = trt_latest[key]
            tp, lat, dev, trtv = _parse_trt_bench(path)
            trt_by_key[key] = TrtBenchRecord(
                path=path,
                config_key=key,
                run_ts=ts,
                throughput_qps=tp,
                latency_mean_ms=lat,
                raw_device=dev,
                raw_trt_version=trtv,
            )
        if key in eval_latest:
            path, ts, _ = eval_latest[key]
            m95, m50, pre, inf, post = _parse_eval(path)
            eval_by_key[key] = EvalRecord(
                path=path,
                config_key=key,
                run_ts=ts,
                map_5095=m95,
                map_50=m50,
                preprocess_ms=pre,
                infer_ms=inf,
                postprocess_ms=post,
            )

    device = None
    trtver = None
    for rec in trt_by_key.values():
        if rec.raw_device:
            device = rec.raw_device
        if rec.raw_trt_version:
            trtver = rec.raw_trt_version

    for key in all_keys:
        meta = _parse_config_meta(key)
        tr = trt_by_key.get(key)
        ev = eval_by_key.get(key)
        m95 = ev.map_5095 if ev else None
        m50 = ev.map_50 if ev else None
        qps = tr.throughput_qps if tr else None
        lat = tr.latency_mean_ms if tr else None
        score = (
            _combined_score(m95, qps)
            if (m95 is not None and qps is not None)
            else None
        )
        rows.append(
            {
                "config_key": key,
                "precision": meta.precision or "—",
                "calibrator": meta.calibrator or "—",
                "map5095": m95,
                "map50": m50,
                "qps": qps,
                "latency_ms": lat,
                "infer_ms": ev.infer_ms if ev else None,
                "combined": score,
                "trt_log": str(tr.path) if tr else "—",
                "eval_log": str(ev.path) if ev else "—",
            }
        )

    def best_by(metric: str, maximize: bool) -> tuple[str | None, float | None]:
        candidates = [
            (r["config_key"], r[metric])
            for r in rows
            if isinstance(r.get(metric), (int, float))
        ]
        if not candidates:
            return None, None
        if maximize:
            k, v = max(candidates, key=lambda x: x[1])  # type: ignore[operator]
        else:
            k, v = min(candidates, key=lambda x: x[1])  # type: ignore[operator]
        return str(k), float(v)  # type: ignore[arg-type]

    best_map, bv_map = best_by("map5095", True)
    best_qps, bv_qps = best_by("qps", True)
    best_lat, bv_lat = best_by("latency_ms", False)
    best_combo, bv_combo = best_by("combined", True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.output
    if out is None:
        rep = artifacts_root() / "reports"
        rep.mkdir(parents=True, exist_ok=True)
        out = rep / f"trt_eval_report_{stamp}.md"

    chart_rows = [r for r in rows if r.get("qps") is not None]
    chart_rows.sort(key=lambda x: float(x["qps"] or 0), reverse=True)
    labels = [_short_label(str(r["config_key"])) for r in chart_rows]
    qps_vals = [float(r["qps"] or 0) for r in chart_rows]

    map_rows = [r for r in rows if r.get("map5095") is not None]
    map_rows.sort(key=lambda x: float(x["map5095"] or 0), reverse=True)
    map_labels = [_short_label(str(r["config_key"])) for r in map_rows]
    map_vals = [float(r["map5095"] or 0) for r in map_rows]

    out_dir = out.resolve().parent
    stem = out.stem
    png_qps = out_dir / f"{stem}_throughput_qps.png"
    png_map = out_dir / f"{stem}_map5095.png"
    rel_qps = png_qps.name
    rel_map = png_map.name

    if chart_rows:
        _write_bar_chart_png(png_qps, "Throughput (QPS) — trt-bench", labels, qps_vals, "QPS")
    if map_rows:
        _write_bar_chart_png(
            png_map, "mAP@0.5:0.95 — eval-trt (pycocotools)", map_labels, map_vals, "mAP"
        )

    lines: list[str] = []
    lines.append("# TensorRT bench + COCO eval — summary")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    trt_src = ", ".join(f"`{d}`" for d in trt_dirs)
    ev_src = ", ".join(f"`{d}`" for d in eval_dirs)
    lines.append(
        f"- TRT logs (`trt_bench_*.log`): scanned {trt_src} → **{len(trt_latest)}** unique config(s)"
    )
    lines.append(
        f"- Eval logs (`eval_*.log`): scanned {ev_src} → **{len(eval_latest)}** unique config(s)"
    )
    if args.merge_global_logs:
        lines.append(
            "- **Note:** `--merge-global-logs` was set; session + global `artifacts/.../logs` were merged (newest timestamp per stem)."
        )
    lines.append("")
    lines.extend(
        _environment_table_lines(
            tensorrt_from_log=trtver,
            device_from_log=device,
        )
    )
    lines.append("## Best configuration (by metric)")
    lines.append("")
    lines.append("| Metric | Config (stem) | Value |")
    lines.append("|---|---|---:|")
    if best_map:
        lines.append(
            f"| **mAP@0.5:0.95** (higher is better) | `{best_map}` | {bv_map:.4f} |"
        )
    if best_qps:
        lines.append(f"| **Throughput** (higher QPS) | `{best_qps}` | {bv_qps:.2f} qps |")
    if best_lat:
        lines.append(
            f"| **Mean latency** (lower ms, trtexec) | `{best_lat}` | {bv_lat:.4f} ms |"
        )
    if best_combo:
        lines.append(
            f"| **Combined** √(mAP × QPS/1000) | `{best_combo}` | {bv_combo:.4f} |"
        )
    lines.append("")
    lines.append(
        "The combined score balances accuracy and throughput; adjust `_combined_score` in "
        "`model_opt_yolo/report_runs.py` if you want different weighting."
    )
    lines.append("")
    lines.append("## Charts (PNG)")
    lines.append("")
    if chart_rows:
        lines.append("### Throughput (QPS) — trt-bench")
        lines.append("")
        lines.append(f"![Throughput QPS]({rel_qps})")
        lines.append("")
    else:
        lines.append("*No throughput values found in TRT logs.*")
        lines.append("")
    if map_rows:
        lines.append("### mAP@0.5:0.95 — eval-trt (pycocotools)")
        lines.append("")
        lines.append(f"![mAP@0.5:0.95]({rel_map})")
        lines.append("")
    else:
        lines.append("*No mAP values found in eval logs.*")
        lines.append("")
    lines.append("## Full table")
    lines.append("")
    lines.append(
        "| Config (stem) | Precision | Calibrator | mAP@0.5:0.95 | mAP@0.5 | QPS | Latency ms (mean) | Infer ms (eval) | Combined score |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: str(x["config_key"])):
        fmt4 = lambda v: f"{v:.4f}" if isinstance(v, float) else "—"
        fmt2 = lambda v: f"{v:.2f}" if isinstance(v, float) else "—"
        lines.append(
            f"| `{r['config_key']}` | {r['precision']} | {r['calibrator']} | "
            f"{fmt4(r.get('map5095'))} | {fmt4(r.get('map50'))} | {fmt2(r.get('qps'))} | "
            f"{fmt4(r.get('latency_ms'))} | {fmt4(r.get('infer_ms'))} | {fmt4(r.get('combined'))} |"
        )
    lines.append("")
    lines.append("## Log files used (latest per config)")
    lines.append("")
    lines.append("| Config | trt_bench log | eval log |")
    lines.append("|---|---|---|")
    for key in sorted(all_keys):
        t = f"`{trt_latest[key][0].name}`" if key in trt_latest else "—"
        e = f"`{eval_latest[key][0].name}`" if key in eval_latest else "—"
        lines.append(f"| `{key}` | {t} | {e} |")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    if chart_rows:
        print(f"Wrote {png_qps}")
    if map_rows:
        print(f"Wrote {png_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
