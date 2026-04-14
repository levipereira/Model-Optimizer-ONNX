#!/usr/bin/env python3
"""TensorRT engine profiling + TREx artifacts: ``trtexec`` build/profile, layer JSON, optional graph + compare.

Requires the **trex** package (TensorRT ``trt-engine-explorer``). In Docker, TREx lives in
``$TREX_VENV`` (separate venv from **model-opt-yolo**); this command prepends that venv's
``site-packages`` to ``sys.path``, then re-executes with that interpreter only if **trex** is still
not importable."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import site
import subprocess
import sys
import math
import statistics
from pathlib import Path
from typing import Any

from model_opt_yolo.build_trt import build_trtexec_argv
from model_opt_yolo.io_checks import validate_readable_file
from model_opt_yolo.logutil import add_logging_arguments, setup_logging
from model_opt_yolo.session_paths import (
    default_trex_analyze_run_dir,
    effective_session_id,
    run_timestamp,
    safe_component,
)

# Match docker/Dockerfile ENV; user may override via TREX_VENV / TREX_HOME.
_DEFAULT_TREX_HOME = "/workspace/TREx"
_DEFAULT_TREX_VENV = (
    f"{_DEFAULT_TREX_HOME}/tools/experimental/trt-engine-explorer/env_trex"
)


def _effective_trex_venv() -> Path:
    """Resolve TREX_VENV. Treat unset or blank like unset (``os.environ.get("X", default)`` misses empty string)."""
    raw = (os.environ.get("TREX_VENV") or "").strip()
    return Path(raw or _DEFAULT_TREX_VENV)


def _effective_trex_home() -> Path:
    raw = (os.environ.get("TREX_HOME") or "").strip()
    return Path(raw or _DEFAULT_TREX_HOME)


def _shape(batch: int, img_size: int) -> str:
    return f"{batch}x3x{img_size}x{img_size}"


def _trex_venv_site_packages(venv: Path) -> list[Path]:
    """Return ``…/lib(64)/python*/site-packages`` dirs under a virtualenv."""
    out: list[Path] = []
    for name in ("lib", "lib64"):
        base = venv / name
        if base.is_dir():
            out.extend(sorted(base.glob("python*/site-packages")))
    return out


def _inject_trex_paths() -> None:
    """Register TREx venv ``site-packages`` via ``site.addsitedir`` (processes ``.pth`` from ``pip -e``)."""
    venv = _effective_trex_venv()
    for sp in _trex_venv_site_packages(venv):
        if sp.is_dir():
            site.addsitedir(str(sp.resolve()))
    trex_home = _effective_trex_home()
    explorer = trex_home / "tools" / "experimental" / "trt-engine-explorer"
    if explorer.is_dir():
        root = str(explorer.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


def _maybe_reexec_with_trex_venv(argv: list[str]) -> None:
    """Try ``import trex``; inject venv paths; re-exec venv python only if still needed."""
    if os.environ.get("MODELOPT_TREX_NO_REEXEC") == "1":
        return
    try:
        import trex  # noqa: F401
    except ImportError:
        pass
    else:
        return
    _inject_trex_paths()
    sys.modules.pop("trex", None)
    try:
        import trex  # noqa: F401
    except ImportError:
        pass
    else:
        return
    venv = _effective_trex_venv()
    py = venv / "bin" / "python"
    if not py.is_file():
        return
    if Path(sys.executable).resolve() == py.resolve():
        return
    os.environ["MODELOPT_TREX_NO_REEXEC"] = "1"
    os.execv(str(py), [str(py), "-m", "model_opt_yolo.trex_analyze", *argv])


_last_trex_import_error: str | None = None


def _verify_trex_import() -> bool:
    """After :func:`_maybe_reexec_with_trex_venv`, confirm ``trex`` loads (paths + optional re-exec)."""
    global _last_trex_import_error
    _last_trex_import_error = None
    try:
        import trex  # noqa: F401
    except ImportError as e:
        _last_trex_import_error = str(e) or repr(e)
    else:
        return True
    _inject_trex_paths()
    sys.modules.pop("trex", None)
    try:
        import trex  # noqa: F401
    except ImportError as e:
        _last_trex_import_error = str(e) or repr(e)
        return False
    return True


def _apply_trex_df_fillna_patch(logger: logging.Logger | None = None) -> None:
    """Work around TREx ``df_preprocessing.__fix_columns_types`` calling ``df.fillna(0)`` on the full frame.

    Newer TensorRT ``exportLayerInfo`` JSON includes string columns (pandas Arrow ``string`` dtype).
    ``fillna(0)`` then raises ``TypeError`` (invalid fill for ``str``). We mirror TREx's int-column
    handling, then fill NaNs per column: numeric → ``0``, boolean → ``False``, else → ``\"\"``.

    Set ``MODELOPT_TREX_NO_DF_PATCH=1`` to skip and use upstream TREx behavior.
    """
    if os.environ.get("MODELOPT_TREX_NO_DF_PATCH") == "1":
        return
    import pandas as pd
    import trex.df_preprocessing as dfp

    # Keep in sync with ``trex.df_preprocessing.__fix_columns_types`` int list.
    int_cols = (
        "Groups",
        "OutMaps",
        "HasBias",
        "HasReLU",
        "AllowSparse",
        "NbInputArgs",
        "NbOutputVars",
        "NbParams",
        "NbLiterals",
    )

    def _fix_columns_types_safe(df: Any) -> None:
        for col in int_cols:
            try:
                df[col] = df[col].fillna(value=0)
                df[col] = df[col].astype("int32")
            except KeyError:
                pass
        for col in list(df.columns):
            if col in int_cols:
                continue
            s = df[col]
            try:
                if pd.api.types.is_numeric_dtype(s):
                    df[col] = s.fillna(0)
                elif pd.api.types.is_bool_dtype(s):
                    df[col] = s.fillna(False)
                else:
                    df[col] = s.fillna("")
            except (TypeError, ValueError):
                pass

    dfp.__fix_columns_types = _fix_columns_types_safe
    if logger is not None:
        logger.debug("Patched trex.df_preprocessing.__fix_columns_types (string-safe fillna).")


def _trex_env_diagnostic_lines() -> list[str]:
    """Human-readable lines for logs when ``trex`` cannot be imported."""
    venv = _effective_trex_venv()
    sps = _trex_venv_site_packages(venv)
    return [
        f"TREX_VENV -> {venv} (directory exists: {venv.is_dir()})",
        f"TREX_HOME -> {_effective_trex_home()}",
        f"site-packages paths found: {sps!r}",
        f"sys.executable: {sys.executable}",
    ]


def _run_trtexec_logged(
    argv: list[str],
    *,
    log_path: Path,
    logger: logging.Logger,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    trtexec_bin = shutil.which("trtexec")
    if not trtexec_bin:
        logger.error("trtexec not found in PATH.")
        return 127
    argv = [trtexec_bin, *argv[1:]]
    logger.info("Running: %s", " ".join(argv))
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# cmd: {' '.join(argv)}\n\n")
        fh.flush()
        p = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        fh.write(p.stdout or "")
    if p.returncode != 0:
        logger.error("trtexec failed (exit %s). See %s", p.returncode, log_path)
    else:
        logger.info("Wrote log: %s", log_path.resolve())
    return p.returncode


def _engine_paths(
    *,
    out_dir: Path,
    onnx_stem: str,
    label: str,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return engine path and TREx JSON paths (graph/profile/timing share engine prefix)."""
    eng = out_dir / f"{onnx_stem}.{label}.engine"
    base = str(eng)
    graph = Path(base + ".graph.json")
    profile = Path(base + ".profile.json")
    timing = Path(base + ".timing.json")
    return eng, graph, profile, timing, Path(base + ".profile.log")


def _build_argv_with_trex(
    *,
    onnx_path: Path,
    engine_path: Path,
    timing_cache: Path,
    input_name: str,
    img_size: int,
    batch: int,
    mode: str,
    graph_json: Path,
    extra: list[str],
) -> list[str]:
    argv = build_trtexec_argv(
        onnx_path=onnx_path,
        engine_path=engine_path,
        timing_cache=timing_cache,
        input_name=input_name,
        img_size=img_size,
        batch=batch,
        mode=mode,
        extra=[
            "--profilingVerbosity=detailed",
            f"--exportLayerInfo={graph_json.resolve()}",
            "--dumpLayerInfo",
            *extra,
        ],
    )
    return argv


def _profile_argv(
    *,
    engine_path: Path,
    timing_cache: Path,
    graph_json: Path,
    profile_json: Path,
    timing_json: Path,
    input_name: str,
    shape_str: str,
    extra: list[str],
) -> list[str]:
    return [
        "trtexec",
        "--verbose",
        "--noDataTransfers",
        "--useCudaGraph",
        "--separateProfileRun",
        "--useSpinWait",
        f"--loadEngine={engine_path.resolve()}",
        f"--exportTimes={timing_json.resolve()}",
        f"--exportProfile={profile_json.resolve()}",
        f"--exportLayerInfo={graph_json.resolve()}",
        f"--timingCacheFile={timing_cache.resolve()}",
        "--profilingVerbosity=detailed",
        f"--minShapes={input_name}:{shape_str}",
        f"--optShapes={input_name}:{shape_str}",
        f"--maxShapes={input_name}:{shape_str}",
        *extra,
    ]


def _optional_metadata_paths(
    profile_json: Path, engine_path: Path
) -> tuple[str | None, str | None]:
    """TREx can enrich EnginePlan when trtexec (or tooling) emitted metadata JSON next to the engine."""
    prof_meta: str | None = None
    cand_p = Path(str(profile_json).replace(".profile.json", ".profile.metadata.json"))
    if cand_p.is_file():
        prof_meta = str(cand_p.resolve())
    build_meta: str | None = None
    cand_b = Path(str(engine_path) + ".build.metadata.json")
    if cand_b.is_file():
        build_meta = str(cand_b.resolve())
    return prof_meta, build_meta


def _load_engine_plan(
    graph_json: Path, profile_json: Path, name: str, *, engine_path: Path
) -> Any:
    from trex.engine_plan import EnginePlan

    prof_meta, build_meta = _optional_metadata_paths(profile_json, engine_path)
    return EnginePlan(
        str(graph_json.resolve()),
        str(profile_json.resolve()),
        profiling_metadata_file=prof_meta,
        build_metadata_file=build_meta,
        name=name,
    )


def _write_plan_graph(
    plan: Any,
    out_base: Path,
    *,
    fmt: str,
    logger: logging.Logger,
) -> Path | None:
    try:
        from trex.graphing import layer_type_formatter, render_dot, to_dot
    except ImportError as e:
        logger.warning("TREx graphing not available: %s", e)
        return None

    dot = to_dot(
        plan,
        layer_node_formatter=layer_type_formatter,
        display_regions=False,
        display_latency=True,
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    name = str(out_base.resolve())
    path = render_dot(dot, name, fmt)
    logger.info("Plan graph: %s", path)
    return Path(path)


def _compare_plans_to_csv(
    plan_a: Any,
    plan_b: Any,
    csv_path: Path,
    logger: logging.Logger,
) -> None:
    try:
        from trex.compare_engines import aligned_merge_plans, match_layers
    except ImportError as e:
        logger.warning("TREx compare_engines not available: %s", e)
        return

    if plan_a.name == plan_b.name:
        plan_a.name = f"{plan_a.name}.a"
        plan_b.name = f"{plan_b.name}.b"
    pairs = match_layers(plan_a, plan_b, exact_matching=False)
    df = aligned_merge_plans(plan_a, plan_b, pairs)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Wrote comparison table: %s", csv_path.resolve())


def _timing_stats_ms(latencies: list[float]) -> dict[str, float | int]:
    if not latencies:
        return {}
    xs = sorted(latencies)
    n = len(xs)

    def percentile(p: float) -> float:
        if n == 1:
            return float(xs[0])
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(xs[int(k)])
        return float(xs[f] * (c - k) + xs[c] * (k - f))

    return {
        "samples": n,
        "min_ms": float(xs[0]),
        "max_ms": float(xs[-1]),
        "mean_ms": float(statistics.mean(xs)),
        "median_ms": float(statistics.median(xs)),
        "p99_ms": percentile(99.0),
    }


def _markdown_kv_section(title: str, d: dict[str, Any]) -> str:
    lines = [f"## {title}", ""]
    if not d:
        lines.append("*No data.*")
        lines.append("")
        return "\n".join(lines)
    for k, v in d.items():
        vs = str(v).strip()
        vs = vs.replace("\n", "<br/>")
        lines.append(f"- **{k}:** {vs}")
    lines.append("")
    return "\n".join(lines)


def _df_to_pipe_markdown(df: Any, *, max_rows: int | None = None) -> str:
    import pandas as pd

    if max_rows is not None:
        df = df.head(max_rows)
    if df is None or len(df) == 0:
        return "*Empty table.*\n"
    cols = [str(c) for c in df.columns]
    esc = []
    for c in cols:
        s = c.replace("|", "\\|")
        esc.append(s)
    header = "| " + " | ".join(esc) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    out_lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                s = ""
            else:
                s = str(v)
            s = s.replace("|", "\\|")
            if len(s) > 240:
                s = s[:237] + "..."
            cells.append(s)
        out_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(out_lines) + "\n"


def _write_engine_report_markdown(
    plan: Any,
    *,
    timing_json: Path | None,
    out_md: Path,
    logger: logging.Logger,
    max_layer_rows: int,
) -> None:
    """Markdown analogue of TREx ``engine_report_card.ipynb`` (text tables + summaries)."""
    from trex.df_preprocessing import clean_for_display
    from trex.engine_plan import summary_dict
    from trex.misc import group_count, group_sum_attr
    from trex import parser

    lines: list[str] = [
        "# TensorRT Engine Report Card",
        "",
        "Text summary aligned with the TREx notebook "
        "(`engine_report_card.ipynb`): plan summary, engine timings, and layer tables. "
        "Interactive plots (Plotly) from the notebook are not reproduced here.",
        "",
    ]

    lines.append(_markdown_kv_section("Model", summary_dict(plan)))

    lines.append(_markdown_kv_section("Device properties", getattr(plan, "device_properties", {}) or {}))
    lines.append(_markdown_kv_section("Builder configuration", getattr(plan, "builder_cfg", {}) or {}))
    lines.append(_markdown_kv_section("Performance summary (metadata)", getattr(plan, "performance_summary", {}) or {}))

    latencies: list[float] = []
    if timing_json is not None and timing_json.is_file():
        try:
            latencies = list(parser.read_timing_file(str(timing_json.resolve())))
        except Exception as e:
            logger.warning("Could not read timing JSON for report: %s", e)
    if latencies:
        st = _timing_stats_ms(latencies)
        lines.append("## Engine timing samples (`exportTimes`)")
        lines.append("")
        lines.append(
            "End-to-end latency samples from `trtexec --exportTimes` (not per-layer profile). "
            "Per-layer **Average time** below is the sum of layer averages when profiled separately."
        )
        lines.append("")
        for k in ("samples", "min_ms", "max_ms", "mean_ms", "median_ms", "p99_ms"):
            if k in st:
                lines.append(f"- **{k}:** {st[k]:.6f}" if k != "samples" else f"- **{k}:** {int(st[k])}")
        lines.append("")
    else:
        lines.append("## Engine timing samples")
        lines.append("")
        lines.append("*No timing JSON or empty file — skipped.*")
        lines.append("")

    df = plan.df
    lines.append("## Latency by layer type")
    lines.append("")
    by_type = (
        df.groupby("type")[["latency.pct_time", "latency.avg_time"]]
        .sum()
        .reset_index()
        .sort_values("latency.pct_time", ascending=False)
    )
    lines.append(_df_to_pipe_markdown(by_type))
    lines.append("")

    lines.append("## Top layers by % of total time")
    lines.append("")
    top = df.sort_values("latency.pct_time", ascending=False)[
        ["Name", "type", "precision", "latency.pct_time", "latency.avg_time"]
    ].head(min(25, len(df)))
    lines.append(_df_to_pipe_markdown(top))
    lines.append("")

    lines.append("## Latency rollup by precision")
    lines.append("")
    prec_sum = group_sum_attr(df, "precision", "latency.pct_time")
    prec_cnt = group_count(df, "precision")
    prec_merged = prec_sum.merge(prec_cnt, on="precision", how="outer").sort_values(
        "latency.pct_time", ascending=False
    )
    lines.append(_df_to_pipe_markdown(prec_merged))
    lines.append("")

    if "tactic" in df.columns:
        lines.append("## Tactics (layer counts)")
        lines.append("")
        tact = group_count(df, "tactic").sort_values("count", ascending=False)
        lines.append(_df_to_pipe_markdown(tact.head(40)))
        lines.append("")

    lines.append("## Memory footprint (per layer, top by total footprint)")
    lines.append("")
    mem_cols = [c for c in ("Name", "type", "weights_size", "total_io_size_bytes", "total_footprint_bytes") if c in df.columns]
    if mem_cols:
        mem = df.sort_values("total_footprint_bytes", ascending=False)[mem_cols].head(25)
        lines.append(_df_to_pipe_markdown(mem))
        lines.append("")

    lines.append("## Layer table (cleaned)")
    lines.append("")
    try:
        cleaned = clean_for_display(df)
    except Exception as e:
        logger.warning("clean_for_display failed, using raw df: %s", e)
        cleaned = df
    n = len(cleaned)
    lines.append(
        f"Showing up to **{max_layer_rows}** of **{n}** rows "
        f"(see `--engine-report-max-layer-rows`)."
    )
    lines.append("")
    lines.append(_df_to_pipe_markdown(cleaned, max_rows=max_layer_rows))
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Engine report card (Markdown): %s", out_md.resolve())


def _write_run_readme(
    run_dir: Path,
    *,
    primary: Path,
    onnx_a: str,
    mode_a: str,
    compare_dir: Path | None,
    onnx_b: str | None,
    mode_b: str | None,
    csv_name: str | None,
) -> None:
    lines = [
        "# TREx analyze run",
        "",
        f"- Primary ONNX ({mode_a}): {onnx_a}",
        f"- Outputs: `{primary.name}/`",
        "- Per folder: `.engine`, `.graph.json`, `.profile.json`, `.timing.json`, `build.log`, `profile.log`",
    ]
    if compare_dir and onnx_b and mode_b:
        lines.extend(
            [
                f"- Compare ONNX ({mode_b}): {onnx_b}",
                f"- Outputs: `{compare_dir.name}/`",
            ]
        )
        if csv_name:
            lines.append(f"- Layer alignment CSV: `{csv_name}`")
    (run_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_one_mode(
    *,
    onnx_path: Path,
    out_dir: Path,
    onnx_stem: str,
    label: str,
    plan_name: str,
    input_name: str,
    img_size: int,
    batch: int,
    mode: str,
    extra: list[str],
    logger: logging.Logger,
    graph_fmt: str | None,
    engine_report_md: Path | None = None,
    engine_report_max_layer_rows: int = 40,
) -> tuple[Any | None, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eng, graph_j, prof_j, timing_j, _ = _engine_paths(
        out_dir=out_dir, onnx_stem=onnx_stem, label=label
    )
    timing_cache = Path(str(eng) + ".timing.cache")

    argv_b = _build_argv_with_trex(
        onnx_path=onnx_path,
        engine_path=eng,
        timing_cache=timing_cache,
        input_name=input_name,
        img_size=img_size,
        batch=batch,
        mode=mode,
        graph_json=graph_j,
        extra=extra,
    )
    rc = _run_trtexec_logged(
        argv_b, log_path=out_dir / "build.log", logger=logger
    )
    if rc != 0:
        return None, out_dir

    shp = _shape(batch, img_size)
    argv_p = _profile_argv(
        engine_path=eng,
        timing_cache=timing_cache,
        graph_json=graph_j,
        profile_json=prof_j,
        timing_json=timing_j,
        input_name=input_name,
        shape_str=shp,
        extra=extra,
    )
    rc = _run_trtexec_logged(
        argv_p, log_path=out_dir / "profile.log", logger=logger
    )
    if rc != 0:
        return None, out_dir

    try:
        plan = _load_engine_plan(graph_j, prof_j, name=plan_name, engine_path=eng)
    except Exception as e:
        logger.exception("Failed to load EnginePlan: %s", e)
        return None, out_dir

    if engine_report_md is not None:
        try:
            _write_engine_report_markdown(
                plan,
                timing_json=timing_j,
                out_md=engine_report_md,
                logger=logger,
                max_layer_rows=engine_report_max_layer_rows,
            )
        except Exception as e:
            logger.warning("Could not write engine report Markdown: %s", e)

    try:
        import json

        from trex.engine_plan import summary_dict

        blob: dict[str, Any] = {
            "summary": summary_dict(plan),
            "device_properties": getattr(plan, "device_properties", {}),
            "performance_summary": getattr(plan, "performance_summary", {}),
            "builder_cfg": getattr(plan, "builder_cfg", {}),
        }
        (out_dir / "plan_summary.json").write_text(
            json.dumps(blob, indent=2, default=str), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("Could not write plan_summary.json: %s", e)

    if graph_fmt:
        graph_base = out_dir / f"{onnx_stem}.{safe_component(label, 24)}.plan_graph"
        _write_plan_graph(plan, graph_base, fmt=graph_fmt, logger=logger)

    return plan, out_dir


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv if argv is not None else sys.argv[1:])
    _maybe_reexec_with_trex_venv(argv_list)

    parser = argparse.ArgumentParser(
        description=(
            "Build a TensorRT engine from ONNX, profile with trtexec (layer JSON, timing). "
            "Pick at most one mode: --compare (two ONNX → CSV), --graph (one ONNX → plan graph + "
            "timing JSON), --report (one ONNX → Markdown report), or none (profile JSON only). "
            "Modes are mutually exclusive. Outputs go under artifacts/trex/runs/ (or pipeline "
            "session trex/). Requires the trex Python package (Docker image with /workspace/TREx)."
        )
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Primary ONNX path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("best", "strongly-typed", "fp16", "fp16-int8"),
        default="strongly-typed",
        help="trtexec builder mode for --onnx (same semantics as build-trt).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Compare mode (exclusive): two ONNX models via --compare-onnx; primary/, compare/, "
            "and compare_layers__*.csv only — no graph or report."
        ),
    )
    parser.add_argument(
        "--compare-onnx",
        type=str,
        default=None,
        metavar="PATH",
        help="Second ONNX (different graph); only used with --compare.",
    )
    parser.add_argument(
        "--compare-onnx-mode",
        type=str,
        default=None,
        choices=("best", "strongly-typed", "fp16", "fp16-int8"),
        metavar="MODE",
        help="trtexec mode for --compare-onnx (default: same as --mode).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        metavar="N",
        help="Square H=W for min/opt/max shapes.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch for shape profile.",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="images",
        metavar="NAME",
        help="ONNX input tensor name for shapes.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help=(
            "Graph mode (exclusive, single --onnx): TREx Dot plan graph plus timing/profile JSON "
            "from trtexec — no report or compare."
        ),
    )
    parser.add_argument(
        "--graph-format",
        type=str,
        default="svg",
        choices=("svg", "png", "pdf"),
        help="Format when --graph is set (default: svg).",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Report mode (exclusive, single --onnx): Engine Report Card Markdown only — "
            "no graph or compare."
        ),
    )
    parser.add_argument(
        "--engine-report-md",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "With --report only: optional path for engine_report_card.md "
            "(default: under mode__<mode>/)."
        ),
    )
    parser.add_argument(
        "--engine-report-max-layer-rows",
        type=int,
        default=40,
        metavar="N",
        help="Max rows for the cleaned layer table when --report is used (default: 40).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Explicit run directory (default: auto under artifacts/trex/runs/).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        metavar="ID",
        help="Optional pipeline session id (artifacts/pipeline_e2e/sessions/<id>/trex/…).",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args after -- forwarded to trtexec (both build and profile), e.g. "
            "-- --memPoolSize=workspace:8192MiB if you need an explicit workspace pool."
        ),
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv_list)

    extra: list[str] = []
    if args.passthrough:
        pt = args.passthrough
        if pt and pt[0] == "--":
            pt = pt[1:]
        extra.extend(pt)

    if args.engine_report_md is not None and not args.report:
        print("--engine-report-md requires --report.", file=sys.stderr)
        return 2

    exclusive_modes = int(bool(args.compare)) + int(bool(args.graph)) + int(bool(args.report))
    if exclusive_modes > 1:
        print(
            "Use at most one of --compare, --graph, or --report.",
            file=sys.stderr,
        )
        return 2

    err = validate_readable_file(args.onnx, label="Primary ONNX model")
    if err:
        print(err, file=sys.stderr)
        return 1
    if args.compare_onnx_mode and not args.compare_onnx:
        print(
            "--compare-onnx-mode requires --compare-onnx.",
            file=sys.stderr,
        )
        return 2
    if args.compare and not args.compare_onnx:
        print("--compare requires --compare-onnx PATH.", file=sys.stderr)
        return 2
    if args.compare_onnx and not args.compare:
        print(
            "--compare-onnx requires --compare (e.g. ... --compare --compare-onnx PATH).",
            file=sys.stderr,
        )
        return 2

    onnx_path = Path(args.onnx).expanduser().resolve()
    onnx_stem = onnx_path.stem
    compare_onnx_path: Path | None = None
    compare_stem: str | None = None
    if args.compare_onnx:
        err_c = validate_readable_file(args.compare_onnx, label="Compare ONNX model")
        if err_c:
            print(err_c, file=sys.stderr)
            return 1
        compare_onnx_path = Path(args.compare_onnx).expanduser().resolve()
        compare_stem = compare_onnx_path.stem
        if compare_onnx_path == onnx_path:
            print(
                "--compare-onnx must be a different file than --onnx (resolved paths were equal).",
                file=sys.stderr,
            )
            return 2

    sid = effective_session_id(args.session_id)
    ts = run_timestamp()

    mode_slug = safe_component(args.mode, 24)
    if compare_stem:
        mode_slug = f"{safe_component(onnx_stem, 20)}_vs_{safe_component(compare_stem, 20)}"

    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = default_trex_analyze_run_dir(
            onnx_stem=onnx_stem,
            mode_slug=mode_slug,
            ts=ts,
            session_id=sid,
        )

    log_file = args.log_file or str(run_dir / "trex_analyze.log")
    setup_logging("trex_analyze", log_file=log_file, verbose=args.verbose)
    log = logging.getLogger("trex_analyze")
    log.info("Run directory: %s", run_dir.resolve())
    log.info("Primary: %s mode=%s", onnx_path, args.mode)
    if compare_onnx_path:
        mode_b = args.compare_onnx_mode or args.mode
        log.info("Compare: %s mode=%s", compare_onnx_path, mode_b)

    if not _verify_trex_import():
        log.error(
            "The 'trex' package is not importable. Rebuild the image with the TREx layer "
            "(install.sh --venv --full) or set TREX_VENV to a venv that contains trex. "
            "ImportError: %s. See docs/docker-reference.md (TREx).",
            _last_trex_import_error or "(unknown)",
        )
        for line in _trex_env_diagnostic_lines():
            log.error("%s", line)
        return 127

    _apply_trex_df_fillna_patch(log)

    if args.graph and args.no_graph:
        print("Use only one of --graph or --no-graph.", file=sys.stderr)
        return 2
    graph_fmt: str | None
    if args.compare:
        graph_fmt = None
    elif args.graph:
        graph_fmt = args.graph_format
    else:
        graph_fmt = None

    want_report = bool(args.report)

    label_a = safe_component(args.mode, 32)
    plan_name_a = f"{onnx_stem}__{args.mode}"
    if compare_onnx_path:
        primary_dir = run_dir / "primary"
    else:
        primary_dir = run_dir / f"mode__{label_a}"

    engine_report_md_primary: Path | None = None
    if want_report:
        if args.engine_report_md is not None:
            if args.engine_report_md == "":
                engine_report_md_primary = primary_dir / "engine_report_card.md"
            else:
                engine_report_md_primary = Path(args.engine_report_md).expanduser().resolve()
        else:
            engine_report_md_primary = primary_dir / "engine_report_card.md"

    plan_a, _ = _process_one_mode(
        onnx_path=onnx_path,
        out_dir=primary_dir,
        onnx_stem=onnx_stem,
        label=label_a,
        plan_name=plan_name_a,
        input_name=args.input_name,
        img_size=args.img_size,
        batch=args.batch,
        mode=args.mode,
        extra=extra,
        logger=log,
        graph_fmt=graph_fmt,
        engine_report_md=engine_report_md_primary,
        engine_report_max_layer_rows=args.engine_report_max_layer_rows,
    )

    compare_dir: Path | None = None
    plan_b = None
    csv_file_name: str | None = None
    mode_b: str | None = None
    if compare_onnx_path and compare_stem:
        mode_b = args.compare_onnx_mode or args.mode
        label_b = safe_component(mode_b, 32)
        plan_name_b = f"{compare_stem}__{mode_b}"
        compare_dir = run_dir / "compare"
        plan_b, _ = _process_one_mode(
            onnx_path=compare_onnx_path,
            out_dir=compare_dir,
            onnx_stem=compare_stem,
            label=label_b,
            plan_name=plan_name_b,
            input_name=args.input_name,
            img_size=args.img_size,
            batch=args.batch,
            mode=mode_b,
            extra=extra,
            logger=log,
            graph_fmt=None,
            engine_report_md=None,
            engine_report_max_layer_rows=args.engine_report_max_layer_rows,
        )
        if plan_a is not None and plan_b is not None:
            csv_file_name = (
                f"compare_layers__{safe_component(onnx_stem, 32)}__vs__{safe_component(compare_stem, 32)}.csv"
            )
            _compare_plans_to_csv(
                plan_a,
                plan_b,
                run_dir / csv_file_name,
                log,
            )

    _write_run_readme(
        run_dir,
        primary=primary_dir,
        onnx_a=str(onnx_path),
        mode_a=args.mode,
        compare_dir=compare_dir,
        onnx_b=str(compare_onnx_path) if compare_onnx_path else None,
        mode_b=mode_b,
        csv_name=csv_file_name,
    )
    log.info("Done. Artifacts: %s", run_dir.resolve())
    return 0 if plan_a is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
