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
# Full GPU latency line from "=== Performance summary ===" (not H2D/D2H)
RE_LATENCY_FULL = re.compile(
    r"\[I\]\s+Latency:\s*min\s*=\s*([\d.]+)\s*ms,\s*max\s*=\s*([\d.]+)\s*ms,\s*mean\s*=\s*([\d.]+)\s*ms,\s*"
    r"median\s*=\s*([\d.]+)\s*ms,\s*percentile\(90%\)\s*=\s*([\d.]+)\s*ms,\s*"
    r"percentile\(95%\)\s*=\s*([\d.]+)\s*ms,\s*percentile\(99%\)\s*=\s*([\d.]+)\s*ms",
    re.IGNORECASE,
)
# Fallback: mean only (older trtexec / truncated logs)
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
RE_TOTAL_DETECTIONS = re.compile(r"Total detections:\s*(\d+)", re.IGNORECASE)
RE_DEVICE = re.compile(r"Selected Device:\s*(.+)$", re.MULTILINE)
RE_TRT_VER = re.compile(r"TensorRT version:\s*([\d.]+)", re.MULTILINE)
# trtexec: "Input binding for input with dimensions 1x3x640x640 is created."
RE_INPUT_BATCH = re.compile(
    r"Input binding for .+ with dimensions (\d+)x\d+x\d+x\d+",
    re.IGNORECASE,
)
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
    latency_median_ms: float | None = None
    latency_p90_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    batch_size: int | None = None
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
    total_detections: int | None = None


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


def _parse_trt_bench(
    path: Path,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    int | None,
    str | None,
    str | None,
]:
    """Parse trt-bench log: throughput, latencies, batch (input dim N), device, TRT version."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tp = None
    for m in RE_THROUGHPUT.finditer(text):
        tp = float(m.group(1))
    mean = med = p90 = p95 = p99 = None
    m_full = RE_LATENCY_FULL.search(text)
    if m_full:
        mean = float(m_full.group(3))
        med = float(m_full.group(4))
        p90 = float(m_full.group(5))
        p95 = float(m_full.group(6))
        p99 = float(m_full.group(7))
    else:
        m_mean = RE_LATENCY_SUMMARY.search(text)
        if m_mean:
            mean = float(m_mean.group(1))
    batch = None
    m_b = RE_INPUT_BATCH.search(text)
    if m_b:
        batch = int(m_b.group(1))
    dev_m = RE_DEVICE.search(text)
    trt_m = RE_TRT_VER.search(text)
    return (
        tp,
        mean,
        med,
        p90,
        p95,
        p99,
        batch,
        (dev_m.group(1).strip() if dev_m else None),
        (trt_m.group(1) if trt_m else None),
    )


def _parse_eval(path: Path) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    int | None,
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
    n_det = None
    m_td = RE_TOTAL_DETECTIONS.search(text)
    if m_td:
        n_det = int(m_td.group(1))
    return m5095, m50, pre, inf, post, n_det


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


def _chart_axis_label(r: dict) -> str:
    """Short label: FP16 baseline vs ``precision · calibrator`` for quantized rows."""
    prec = str(r.get("precision") or "—")
    cal = str(r.get("calibrator") or "—")
    if prec == "fp16" and cal == "—":
        return "FP16 baseline"
    if cal != "—":
        return f"{prec} · {cal}"
    return prec


def _is_fp16_baseline_row(r: dict) -> bool:
    pk = str(r.get("config_key", ""))
    if str(r.get("precision") or "") != "fp16":
        return False
    return ".quant" not in pk


def _sort_rows_report_order(rows: list[dict]) -> list[dict]:
    """fp16 first, then int8 / fp8 / int4, then calibrator, then key."""

    def sk(r: dict) -> tuple:
        meta = _parse_config_meta(str(r["config_key"]))
        prec_order = {"fp16": 0, "int8": 1, "fp8": 2, "int4": 3, "int16": 4}
        po = prec_order.get((meta.precision or "").lower(), 99)
        return (po, meta.calibrator or "", str(r["config_key"]))

    return sorted(rows, key=sk)


def _write_eval_map_detections_dual_axis_png(
    out_png: Path,
    title: str,
    labels: list[str],
    map_vals: list[float],
    det_vals: list[float],
) -> None:
    """Left: mAP@0.5:0.95; right: total bbox detections (eval-trt log)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11.5, 5.2), dpi=120)
    c_map, c_det = "#2ca02c", "#ff7f0e"
    ax1.plot(x, map_vals, "o-", color=c_map, linewidth=2.0, markersize=8, label="mAP@0.5:0.95")
    ax1.set_ylabel("mAP@0.5:0.95", color=c_map, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=c_map)
    for xv, yv in zip(x, map_vals):
        if not np.isnan(yv):
            ax1.annotate(
                f"{yv:.4f}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color=c_map,
            )
    ax2 = ax1.twinx()
    ax2.plot(x, det_vals, "s-", color=c_det, linewidth=2.0, markersize=7, label="Total detections")
    ax2.set_ylabel("Total detections (count)", color=c_det, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=c_det)
    for xv, yv in zip(x, det_vals):
        if not np.isnan(yv):
            ax2.annotate(
                f"{int(yv):,}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=6,
                color=c_det,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax1.set_title(title, fontsize=11)
    ax1.grid(axis="y", alpha=0.26)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)


def _write_line_chart_annotated_png(
    out_png: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
) -> None:
    """Line chart with numeric labels at each point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    x = np.arange(len(labels))
    ax.plot(x, values, "o-", color="#2ca02c", linewidth=2.0, markersize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ymin = min(values) if values else 0.0
    ymax = max(values) if values else 1.0
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.04
    ax.set_ylim(max(0, ymin - pad), ymax + pad)
    for i, (xv, yv) in enumerate(zip(x, values)):
        ax.annotate(
            f"{yv:.4f}",
            (xv, yv),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color="#1a5c1a",
        )
    fig.tight_layout()
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)


def _write_ips_latency_twin_png(
    out_png: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    ips_vals: list[float],
    mean_ms: list[float],
    p99_ms: list[float],
) -> None:
    """Left axis: IPS = batch × QPS; right axis: mean & p99 GPU latency (trtexec)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(11.8, 5.5), dpi=120)
    c_ips, c_m, c_p = "#1f77b4", "#d62728", "#9467bd"
    ax1.plot(x, ips_vals, "o-", color=c_ips, linewidth=2.0, markersize=8, label="IPS (batch × QPS)")
    ax1.set_ylabel("IPS (inferences/s)", color=c_ips, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=c_ips)
    for xv, yv in zip(x, ips_vals):
        if not np.isnan(yv):
            ax1.annotate(
                f"{yv:.1f}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color=c_ips,
            )
    ax2 = ax1.twinx()
    ax2.plot(x, mean_ms, "^-", color=c_m, linewidth=1.9, markersize=6, label="Mean latency (ms)")
    ax2.plot(x, p99_ms, "s-", color=c_p, linewidth=1.9, markersize=6, label="p99 latency (ms)")
    ax2.set_ylabel("Latency (ms, GPU)", fontsize=10)
    ax2.tick_params(axis="y")
    for xv, yv in zip(x, mean_ms):
        if not np.isnan(yv):
            ax2.annotate(
                f"{yv:.4f}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=6,
                color=c_m,
            )
    for xv, yv in zip(x, p99_ms):
        if not np.isnan(yv):
            ax2.annotate(
                f"{yv:.4f}",
                (xv, yv),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=6,
                color=c_p,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax1.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=10)
    ax1.grid(axis="y", alpha=0.26)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_png, format="png", bbox_inches="tight")
    plt.close(fig)


def _fmt_opt4(v: object) -> str:
    return f"{float(v):.4f}" if isinstance(v, (int, float)) else "—"


def _fmt_opt2(v: object) -> str:
    return f"{float(v):.2f}" if isinstance(v, (int, float)) else "—"


def _pct_vs_ref(value: float | None, ref: float | None) -> str:
    if value is None or ref is None or ref == 0:
        return "—"
    return f"{100.0 * (float(value) - float(ref)) / float(ref):+.2f}%"


def _pick_best_row(rows: list[dict]) -> dict | None:
    """Prefer highest combined score (mAP + QPS); else highest mAP."""
    with_combo = [r for r in rows if isinstance(r.get("combined"), (int, float))]
    if with_combo:
        return max(with_combo, key=lambda r: float(r["combined"]))
    with_map = [r for r in rows if isinstance(r.get("map5095"), (int, float))]
    if with_map:
        return max(with_map, key=lambda r: float(r["map5095"]))
    return None


def _comparison_table_md(
    ref: dict,
    others: list[dict],
    heading: str,
    ref_description: str,
) -> list[str]:
    """Markdown table: each row shows absolute metrics and % vs reference."""
    lines: list[str] = [
        f"## {heading}",
        "",
        f"Reference: **{ref_description}** — mAP `{ref.get('config_key')}`, "
        f"mAP@0.5:0.95 = {_fmt_opt4(ref.get('map5095'))}, QPS = {_fmt_opt2(ref.get('qps'))}, "
        f"mean ms = {_fmt_opt4(ref.get('latency_ms'))}, p99 ms = {_fmt_opt4(ref.get('latency_p99_ms'))}.",
        "",
        "% vs reference = `100 × (value − ref) / ref` (for latency ms, **+** means slower).",
        "",
        "| Config | mAP@0.5:0.95 | vs ref | QPS | vs ref | Mean ms | vs ref | p99 ms | vs ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rk = ref
    for r in others:
        lines.append(
            "| `{cfg}` | {m} | {mp} | {q} | {qp} | {lm} | {lmp} | {p9} | {p9p} |".format(
                cfg=r["config_key"],
                m=_fmt_opt4(r.get("map5095")),
                mp=_pct_vs_ref(
                    float(r["map5095"]) if r.get("map5095") is not None else None,
                    float(rk["map5095"]) if rk.get("map5095") is not None else None,
                ),
                q=_fmt_opt2(r.get("qps")),
                qp=_pct_vs_ref(
                    float(r["qps"]) if r.get("qps") is not None else None,
                    float(rk["qps"]) if rk.get("qps") is not None else None,
                ),
                lm=_fmt_opt4(r.get("latency_ms")),
                lmp=_pct_vs_ref(
                    float(r["latency_ms"]) if r.get("latency_ms") is not None else None,
                    float(rk["latency_ms"]) if rk.get("latency_ms") is not None else None,
                ),
                p9=_fmt_opt4(r.get("latency_p99_ms")),
                p9p=_pct_vs_ref(
                    float(r["latency_p99_ms"]) if r.get("latency_p99_ms") is not None else None,
                    float(rk["latency_p99_ms"]) if rk.get("latency_p99_ms") is not None else None,
                ),
            )
        )
    lines.append("")
    return lines


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
            "and write a Markdown report with tables and PNG line charts."
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
            tp, mean, med, p90, p95, p99, batch, dev, trtv = _parse_trt_bench(path)
            trt_by_key[key] = TrtBenchRecord(
                path=path,
                config_key=key,
                run_ts=ts,
                throughput_qps=tp,
                latency_mean_ms=mean,
                latency_median_ms=med,
                latency_p90_ms=p90,
                latency_p95_ms=p95,
                latency_p99_ms=p99,
                batch_size=batch,
                raw_device=dev,
                raw_trt_version=trtv,
            )
        if key in eval_latest:
            path, ts, _ = eval_latest[key]
            m95, m50, pre, inf, post, n_det = _parse_eval(path)
            eval_by_key[key] = EvalRecord(
                path=path,
                config_key=key,
                run_ts=ts,
                map_5095=m95,
                map_50=m50,
                preprocess_ms=pre,
                infer_ms=inf,
                postprocess_ms=post,
                total_detections=n_det,
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
        lat_mean = tr.latency_mean_ms if tr else None
        lat_med = tr.latency_median_ms if tr else None
        p90 = tr.latency_p90_ms if tr else None
        p95 = tr.latency_p95_ms if tr else None
        p99 = tr.latency_p99_ms if tr else None
        batch_sz = tr.batch_size if tr else None
        bs_eff = int(batch_sz) if isinstance(batch_sz, int) else 1
        ips: float | None = None
        if qps is not None:
            ips = bs_eff * float(qps)
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
                "preprocess_ms": ev.preprocess_ms if ev else None,
                "infer_ms": ev.infer_ms if ev else None,
                "postprocess_ms": ev.postprocess_ms if ev else None,
                "total_detections": ev.total_detections if ev else None,
                "batch_size": batch_sz,
                "ips": ips,
                "qps": qps,
                "latency_ms": lat_mean,
                "latency_median_ms": lat_med,
                "latency_p90_ms": p90,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "combined": score,
                "trt_log": str(tr.path) if tr else "—",
                "eval_log": str(ev.path) if ev else "—",
            }
        )

    rows_sorted = _sort_rows_report_order(rows)

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

    def best_latency_row() -> tuple[str | None, float | None]:
        """Lowest mean GPU latency (trtexec summary)."""
        best_k, best_v = None, None
        for r in rows:
            lv = r.get("latency_ms")
            if not isinstance(lv, (int, float)):
                continue
            lv = float(lv)
            if best_v is None or lv < best_v:
                best_k, best_v = str(r["config_key"]), lv
        return best_k, best_v

    best_map, bv_map = best_by("map5095", True)
    best_qps, bv_qps = best_by("qps", True)
    best_ips, bv_ips = best_by("ips", True)
    best_lat_k, bv_lat = best_latency_row()
    best_combo, bv_combo = best_by("combined", True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.output
    if out is None:
        rep = artifacts_root() / "reports"
        rep.mkdir(parents=True, exist_ok=True)
        out = rep / f"trt_eval_report_{stamp}.md"

    thr_rows = [r for r in rows_sorted if r.get("qps") is not None]
    thr_labels = [_chart_axis_label(r) for r in thr_rows]
    ips_vals: list[float] = []
    for r in thr_rows:
        ip = r.get("ips")
        ips_vals.append(float(ip) if isinstance(ip, (int, float)) else float("nan"))
    lat_mean_plot: list[float] = []
    lat_p99_plot: list[float] = []
    for r in thr_rows:
        m = r.get("latency_ms")
        p9 = r.get("latency_p99_ms")
        lat_mean_plot.append(float(m) if isinstance(m, (int, float)) else float("nan"))
        lat_p99_plot.append(float(p9) if isinstance(p9, (int, float)) else float("nan"))
    has_perf_chart = bool(thr_rows) and (
        any(isinstance(r.get("latency_ms"), (int, float)) for r in thr_rows)
        or any(isinstance(r.get("latency_p99_ms"), (int, float)) for r in thr_rows)
    )
    bs_set = {r.get("batch_size") for r in thr_rows if isinstance(r.get("batch_size"), int)}
    if len(bs_set) == 1 and bs_set:
        b0 = next(iter(bs_set))
        perf_sub = f"IPS = batch × QPS. Batch size = {b0} (trtexec input binding)."
    elif len(bs_set) > 1:
        perf_sub = "IPS = batch × QPS; batch from trtexec input binding (varies per engine)."
    else:
        perf_sub = "IPS = batch × QPS; batch not found in log — assumed 1 for IPS."

    map_rows_ord = [r for r in rows_sorted if r.get("map5095") is not None]
    map_labels = [_chart_axis_label(r) for r in map_rows_ord]
    map_vals = [float(r["map5095"] or 0) for r in map_rows_ord]
    det_plot: list[float] = []
    for r in map_rows_ord:
        td = r.get("total_detections")
        det_plot.append(float(td) if isinstance(td, int) else float("nan"))
    has_det_counts = any(
        isinstance(r.get("total_detections"), int) for r in map_rows_ord
    )

    out_dir = out.resolve().parent
    stem = out.stem
    png_perf = out_dir / f"{stem}_ips_latency.png"
    png_map = out_dir / f"{stem}_map5095.png"
    rel_perf = png_perf.name
    rel_map = png_map.name

    wrote_perf_png = (
        thr_rows
        and has_perf_chart
        and len(ips_vals) == len(thr_rows)
        and len(lat_mean_plot) == len(thr_rows)
        and len(lat_p99_plot) == len(thr_rows)
    )
    if wrote_perf_png:
        _write_ips_latency_twin_png(
            png_perf,
            "IPS (inferences/s) vs GPU latency — trt-bench / trtexec",
            perf_sub,
            thr_labels,
            ips_vals,
            lat_mean_plot,
            lat_p99_plot,
        )
    if map_rows_ord and has_det_counts and len(det_plot) == len(map_vals):
        _write_eval_map_detections_dual_axis_png(
            png_map,
            "mAP@0.5:0.95 & total detections (COCO eval — `Total detections` in eval log)",
            map_labels,
            map_vals,
            det_plot,
        )
    elif map_rows_ord:
        _write_line_chart_annotated_png(
            png_map,
            "mAP@0.5:0.95 (COCO eval — no `Total detections` line in logs)",
            map_labels,
            map_vals,
            "mAP@0.5:0.95",
        )

    has_fp16_baseline = any(_is_fp16_baseline_row(r) for r in rows)

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
    if not has_fp16_baseline:
        lines.append(
            "- **Note:** No FP16 baseline row appears in these logs (e.g. pipeline ran with `--no-fp16-baseline`, "
            "or only PTQ engines were built). The tables and charts list **only** the configurations found."
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
    if best_ips:
        lines.append(
            f"| **IPS** (batch × QPS, inferences/s) | `{best_ips}` | {bv_ips:.2f} |"
        )
    if best_lat_k and bv_lat is not None:
        lines.append(
            f"| **Mean GPU latency** (lower ms, trtexec) | `{best_lat_k}` | {bv_lat:.4f} ms |"
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
    if wrote_perf_png:
        lines.append("### IPS, QPS (implicit) & GPU latency — trt-bench / trtexec")
        lines.append("")
        lines.append(
            "Primary series: **IPS** = batch × QPS (inferences/s). "
            "Right axis: **mean** and **p99** latency (ms). QPS is listed in the throughput table below."
        )
        lines.append("")
        lines.append(f"![IPS and latency]({rel_perf})")
        lines.append("")
    elif thr_rows:
        lines.append(
            "*Could not build IPS/latency chart (missing trtexec latency summary or mismatched series).*"
        )
        lines.append("")
    else:
        lines.append("*No trt-bench throughput found in TRT logs.*")
        lines.append("")
    if map_rows_ord:
        if has_det_counts:
            lines.append("### mAP & total detections (COCO eval)")
        else:
            lines.append("### mAP@0.5:0.95 — eval (no `Total detections` line in logs)")
        lines.append("")
        lines.append(f"![Eval mAP / detections]({rel_map})")
        lines.append("")
    else:
        lines.append("*No mAP values found in eval logs.*")
        lines.append("")
    lines.append("## Table: Eval (COCO / pycocotools)")
    lines.append("")
    lines.append(
        "| Config (stem) | Precision | Calibrator | mAP@0.5:0.95 | mAP@0.5 | Total det. |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    fmt4 = lambda v: f"{v:.4f}" if isinstance(v, float) else "—"

    def fmt_int_det(v: object) -> str:
        if isinstance(v, int):
            return f"{v:,}"
        return "—"

    for r in rows_sorted:
        lines.append(
            f"| `{r['config_key']}` | {r['precision']} | {r['calibrator']} | "
            f"{fmt4(r.get('map5095'))} | {fmt4(r.get('map50'))} | {fmt_int_det(r.get('total_detections'))} |"
        )
    lines.append("")
    lines.append("## Table: Throughput & latency (trtexec summary)")
    lines.append("")
    lines.append(
        "| Config (stem) | Precision | Calibrator | Batch | IPS | QPS | Mean ms | Median ms | p90 ms | p95 ms | p99 ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    fmt2 = lambda v: f"{v:.2f}" if isinstance(v, float) else "—"

    def fmt_batch(v: object) -> str:
        if isinstance(v, int):
            return str(v)
        return "—"

    for r in rows_sorted:
        lines.append(
            f"| `{r['config_key']}` | {r['precision']} | {r['calibrator']} | "
            f"{fmt_batch(r.get('batch_size'))} | {fmt2(r.get('ips'))} | {fmt2(r.get('qps'))} | "
            f"{fmt4(r.get('latency_ms'))} | {fmt4(r.get('latency_median_ms'))} | "
            f"{fmt4(r.get('latency_p90_ms'))} | {fmt4(r.get('latency_p95_ms'))} | {fmt4(r.get('latency_p99_ms'))} |"
        )
    lines.append("")

    fp16_row = next((r for r in rows_sorted if _is_fp16_baseline_row(r)), None)
    if fp16_row:
        others_f = [r for r in rows_sorted if r["config_key"] != fp16_row["config_key"]]
        if others_f:
            lines.extend(
                _comparison_table_md(
                    fp16_row,
                    others_f,
                    "Comparison vs TensorRT FP16 baseline",
                    "FP16 TensorRT (original ONNX, `build-trt --mode fp16`)",
                )
            )

    best_row = _pick_best_row(rows)
    if (
        best_row is not None
        and len(rows) > 1
        and not (
            fp16_row is not None
            and best_row["config_key"] == fp16_row["config_key"]
        )
    ):
        others_b = [r for r in rows_sorted if r["config_key"] != best_row["config_key"]]
        if others_b:
            lines.extend(
                _comparison_table_md(
                    best_row,
                    others_b,
                    "Comparison vs best overall run",
                    f"Best by combined score √(mAP×QPS/1000): `{best_row['config_key']}`",
                )
            )

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
    if wrote_perf_png:
        print(f"Wrote {png_perf}")
    if map_rows_ord:
        print(f"Wrote {png_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
