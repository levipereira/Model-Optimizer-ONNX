"""Load YAML quantization profiles and translate them to Model Optimizer CLI arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Optional dependency: PyYAML
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _pkg_profiles_dir() -> Path:
    return Path(__file__).resolve().parent / "profiles"


def resolve_profile_path(name_or_path: str) -> Path:
    """Resolve ``name_or_path`` to an existing YAML file.

    - If it is an existing path, use it.
    - Else try ``<stem>.yaml`` / ``<stem>.yml`` under the packaged ``profiles/`` dir, then CWD ``profiles/``, then CWD.
    """
    raw = name_or_path.strip()
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()

    stem = raw.replace(".yaml", "").replace(".yml", "")
    candidates = [
        _pkg_profiles_dir() / f"{stem}.yaml",
        _pkg_profiles_dir() / f"{stem}.yml",
        Path("profiles") / f"{stem}.yaml",
        Path("profiles") / f"{stem}.yml",
        Path(f"{stem}.yaml"),
        Path(f"{stem}.yml"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Quantization profile not found: {name_or_path!r}. "
        f"Tried packaged profiles under {_pkg_profiles_dir()} and relative paths."
    )


def load_quantize_profile(path: Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required for --profile. Install with: pip install pyyaml"
        )
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping at top level: {path}")
    return data


def _append_flag_values(out: list[str], flag: str, values: list[str]) -> None:
    if not values:
        return
    out.append(flag)
    out.extend(values)


def modelopt_args_from_profile(
    profile: dict[str, Any],
    *,
    profile_path: Path,
) -> list[str]:
    """Build argument tokens for ``python -m modelopt.onnx.quantization`` from the ``modelopt`` section.

    Also accepts friendly aliases at the top level of the profile (merged into ``modelopt``).
    """
    raw_mo = profile.get("modelopt")
    mo: dict[str, Any] = dict(raw_mo) if isinstance(raw_mo, dict) else {}

    # Top-level shortcuts (optional) — same keys as inside modelopt
    for key in (
        "include_op_types",
        "exclude_op_types",
        "op_types_to_exclude_fp16",
        "include_nodes",
        "exclude_nodes",
        "simplify",
        "calibrate_per_node",
        "direct_io_types",
        "use_external_data_format",
        "disable_mha_qdq",
        "autotune_node_filter_list",
        "extra_args",
    ):
        if key in profile and key not in mo:
            mo[key] = profile[key]

    out: list[str] = []
    base = profile_path.parent.resolve()

    inc = mo.get("include_op_types") or mo.get("op_types_to_quantize")
    if isinstance(inc, list) and inc:
        _append_flag_values(out, "--op_types_to_quantize", [str(x) for x in inc])

    exc = mo.get("exclude_op_types") or mo.get("op_types_to_exclude")
    if isinstance(exc, list) and exc:
        _append_flag_values(out, "--op_types_to_exclude", [str(x) for x in exc])

    excl_fp16 = mo.get("op_types_to_exclude_fp16")
    if isinstance(excl_fp16, list) and excl_fp16:
        _append_flag_values(out, "--op_types_to_exclude_fp16", [str(x) for x in excl_fp16])

    nodes_q = mo.get("include_nodes") or mo.get("nodes_to_quantize")
    if isinstance(nodes_q, list) and nodes_q:
        _append_flag_values(out, "--nodes_to_quantize", [str(x) for x in nodes_q])

    nodes_x = mo.get("exclude_nodes") or mo.get("nodes_to_exclude")
    if isinstance(nodes_x, list) and nodes_x:
        _append_flag_values(out, "--nodes_to_exclude", [str(x) for x in nodes_x])

    if mo.get("simplify") is True:
        out.append("--simplify")
    if mo.get("calibrate_per_node") is True:
        out.append("--calibrate_per_node")
    if mo.get("direct_io_types") is True:
        out.append("--direct_io_types")
    if mo.get("use_external_data_format") is True:
        out.append("--use_external_data_format")
    if mo.get("disable_mha_qdq") is True:
        out.append("--disable_mha_qdq")

    filt = mo.get("autotune_node_filter_list")
    if filt:
        fp = Path(str(filt)).expanduser()
        if not fp.is_absolute():
            fp = (base / fp).resolve()
        if not fp.is_file():
            raise FileNotFoundError(f"autotune_node_filter_list not found: {fp}")
        out.append(f"--autotune_node_filter_list={fp}")

    extra = mo.get("extra_args")
    if isinstance(extra, list):
        for item in extra:
            if not isinstance(item, str):
                raise TypeError(f"extra_args entries must be strings, got {type(item)}")
            out.append(item)
    elif extra is not None:
        raise TypeError("extra_args must be a list of strings")

    return out


def defaults_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return the optional ``defaults`` mapping (quantize CLI knobs)."""
    d = profile.get("defaults")
    return d if isinstance(d, dict) else {}


def merge_autotune_from_profile(
    *,
    cli_autotune: str | None,
    profile: dict[str, Any],
) -> str | None:
    """If CLI did not request autotune, use ``defaults.autotune`` from the profile when set."""
    if cli_autotune is not None:
        return cli_autotune
    d = defaults_from_profile(profile)
    at = d.get("autotune")
    if at is None or at is False:
        return None
    if isinstance(at, str) and at.lower() in ("none", "off", "false", "0"):
        return None
    if at is True:
        return "quick"
    if isinstance(at, str) and at in ("quick", "default", "extensive"):
        return at
    raise ValueError(f"Invalid defaults.autotune in profile: {at!r}")


def describe_profile(profile: dict[str, Any], profile_path: Path) -> str:
    name = profile.get("name") or profile_path.stem
    ver = profile.get("version", "?")
    return f"{name} (version {ver}, {profile_path})"
