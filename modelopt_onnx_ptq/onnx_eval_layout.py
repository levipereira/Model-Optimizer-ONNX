"""Infer eval-trt ``output_format`` from ONNX graphs or TensorRT output bindings.

DeepStream-Yolo export scripts (``utils/export_*.py``) overwhelmingly use:

- ``input_names=[\"input\"]``, ``output_names=[\"output\"]``
- A ``DeepStreamOutput`` head that packs detections as **one** float tensor
  ``[batch, N, 6]`` (xyxy + score + class), with per-image NMS typically done in
  the parser / eval (``deepstream_yolo`` in ``eval_trt.py``).

Exceptions noted in-tree: PP-YOLOE Paddle export uses input name ``image``; DETR-style
exports still use output name ``output`` and ``[B, Q, 6]`` after the wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import shape_inference

log = logging.getLogger(__name__)

# Four-tensor ``num_dets`` / ``det_*`` graphs are not supported by ``eval-trt`` (any mode).
_UNSUPPORTED_FOUR_TENSOR_OUTPUTS = frozenset({"num_dets", "det_boxes", "det_scores", "det_classes"})

# Typical Ultralytics native ONNX uses this name for the single detection tensor.
_ULTRALYTICS_SINGLE_NAMES = frozenset({"output0"})

# DeepStream-Yolo utils (almost all ``export_*.py``) use this output tensor name.
_DEEPSTREAM_SINGLE_NAME = "output"

# Above this middle dimension, treat ambiguous [B,N,6] as dense anchors (DeepStream-style NMS in eval).
_DENSE_CANDIDATE_MIN_N = 512


def normalize_eval_output_format(fmt: str) -> str:
    """Map legacy CLI names to canonical ``eval-trt`` format ids.

    ``ultralytics`` is an alias for ``ultralytics_e2e`` (single ``[B, N, 6]`` tensor with
    NMS already in the exported graph).
    """
    if fmt == "ultralytics":
        return "ultralytics_e2e"
    return fmt


def _tensor_type_dims(
    value_info: onnx.ValueInfoProto,
) -> list[int | str]:
    """Return dimension list (int or str param) or empty if unknown."""
    ttype = value_info.type.tensor_type
    if not ttype.HasField("shape"):
        return []
    out: list[int | str] = []
    for d in ttype.shape.dim:
        if d.dim_value:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(d.dim_param)
        else:
            out.append(-1)
    return out


def _last_dim_is_6(dims: list[int | str]) -> bool:
    if not dims:
        return False
    last = dims[-1]
    return last == 6 if isinstance(last, int) else False


def _is_ultralytics_raw_head(dims: list[int | str]) -> bool:
    """True if shape looks like ``[B, 4+nc, num_anchors]`` (Ultralytics ONNX without in-graph NMS)."""
    if len(dims) != 3:
        return False
    fea, anc = dims[1], dims[2]
    if isinstance(fea, int) and isinstance(anc, int):
        if fea < 8 or anc <= 0:
            return False
        # Packed [B, N, 6] uses last dim 6; raw head has many anchors and fewer channels.
        if _last_dim_is_6(dims) and fea >= anc:
            return False
        return fea < anc and anc != 6
    # e.g. ['batch', 84, 'anchors']
    if isinstance(fea, int) and 8 <= fea <= 1024 and isinstance(anc, str):
        return True
    return False


def _middle_dim_large(dims: list[int | str]) -> bool | None:
    """If rank-3 and middle dim is static int, return whether it looks like dense anchors."""
    if len(dims) != 3:
        return None
    mid = dims[1]
    if isinstance(mid, int) and mid > 0:
        return mid >= _DENSE_CANDIDATE_MIN_N
    return None


def infer_eval_output_format_from_onnx(onnx_path: str | Path) -> str:
    """Return ``ultralytics_e2e``, ``deepstream_yolo``, or ``ultralytics_raw`` for ``eval_trt`` decoding.

    Loads the model, runs ``onnx.shape_inference.infer_shapes`` when possible, then classifies
    graph outputs. Raises ``ValueError`` if the layout is unknown or unsupported for bbox eval.
    """
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {path}")

    model = onnx.load(str(path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001 — best-effort; graph may still have partial info
        log.debug("shape_inference.infer_shapes skipped or failed: %s", exc)

    out_infos = list(model.graph.output)
    names = [o.name for o in out_infos]

    if _UNSUPPORTED_FOUR_TENSOR_OUTPUTS <= set(names):
        raise ValueError(
            "Four-tensor detection outputs (num_dets, det_boxes, det_scores, det_classes) "
            "are not supported by eval-trt. Export or build engines with a single [B,N,6] tensor."
        )

    if len(out_infos) == 1:
        vi = out_infos[0]
        dims = _tensor_type_dims(vi)
        if not dims:
            # Shape inference missing — use DeepStream vs Ultralytics naming conventions.
            if vi.name == _DEEPSTREAM_SINGLE_NAME:
                return "deepstream_yolo"
            if vi.name in _ULTRALYTICS_SINGLE_NAMES:
                return "ultralytics_e2e"
            raise ValueError(
                f"Single output {vi.name!r} has no static shape in ONNX; cannot auto-detect. "
                "Re-export with shape info or set --output-format explicitly."
            )
        if len(dims) != 3:
            raise ValueError(
                f"Single output {vi.name!r} has shape {dims!r}; expected rank-3 detection output."
            )
        if _is_ultralytics_raw_head(dims):
            return "ultralytics_raw"
        if not _last_dim_is_6(dims):
            raise ValueError(
                f"Single output {vi.name!r} has shape {dims!r}; expected last dimension 6 "
                "(packed DeepStream-Yolo / Ultralytics exports) or raw ``[B, 4+nc, anchors]``."
            )
        return _classify_single_tensor_output(name=vi.name, dims=dims)

    if len(out_infos) == 2:
        raise ValueError(
            f"Two graph outputs {names!r} — likely segmentation or non-bbox head; "
            "eval-trt bbox COCO path does not support this layout."
        )

    raise ValueError(
        f"Unsupported ONNX outputs for auto layout: {names!r}. "
        "Expected a single tensor [B,N,6] (DeepStream-Yolo / Ultralytics single-output exports)."
    )


def _classify_single_tensor_output(*, name: str, dims: list[int | str]) -> str:
    """Pick ``ultralytics_e2e`` vs ``deepstream_yolo`` for packed ``[B, N, 6]`` tensors."""
    if name == _DEEPSTREAM_SINGLE_NAME:
        return "deepstream_yolo"
    if name in _ULTRALYTICS_SINGLE_NAMES:
        return "ultralytics_e2e"
    large = _middle_dim_large(dims)
    if large is True:
        return "deepstream_yolo"
    if large is False:
        return "ultralytics_e2e"
    # Dynamic N: prefer DeepStream naming convention
    if name in ("output", "out", "dets"):
        return "deepstream_yolo"
    return "ultralytics_e2e"


def infer_eval_output_format_from_trt_outputs(
    outputs: list[tuple[str, tuple[int, ...]]],
) -> str:
    """Classify using TensorRT engine output tensor names and shapes (runtime shapes)."""
    names = [n for n, _ in outputs]
    if _UNSUPPORTED_FOUR_TENSOR_OUTPUTS <= set(names):
        raise ValueError(
            "Four-tensor engine outputs (num_dets, det_*) are not supported by eval-trt. "
            "Use a single [B,N,6] detection output."
        )

    if len(outputs) == 1:
        name, shape = outputs[0]
        if len(shape) != 3:
            raise ValueError(
                f"Single output {name!r} has shape {shape!r}; expected rank 3."
            )
        ld = shape[-1]
        fd = shape[1]
        if ld in (6, -1):
            return _classify_single_tensor_output(name=name, dims=list(shape))
        if (
            isinstance(ld, int)
            and isinstance(fd, int)
            and fd < ld
            and fd >= 8
            and ld != 6
        ):
            return "ultralytics_raw"
        raise ValueError(
            f"Single output {name!r} has shape {shape!r}; expected packed [B, N, 6] "
            "(last dim 6 or -1) or raw Ultralytics head [B, 4+nc, anchors]."
        )

    if len(outputs) == 2:
        raise ValueError(
            f"Two outputs {names!r} — segmentation or unsupported; cannot auto-select eval layout."
        )

    raise ValueError(
        f"Unsupported TensorRT outputs for auto layout: {names!r}. "
        "Expected one [B,N,6] tensor."
    )


def infer_default_input_tensor_name_from_onnx(onnx_path: str | Path) -> str | None:
    """Return the sole graph input name, or ``None`` if ambiguous.

    DeepStream-Yolo PyTorch exports use ``input``; Ultralytics often uses ``images``.
    PP-YOLOE uses ``image`` (see DeepStream-Yolo ``export_ppyoloe.py``).
    """
    path = Path(onnx_path)
    if not path.is_file():
        return None
    model = onnx.load(str(path))
    inits = {x.name for x in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in inits]
    if len(inputs) == 1:
        return inputs[0].name
    return None


def infer_default_output_tensor_name_from_onnx(onnx_path: str | Path) -> str | None:
    """Return the sole graph output name, or ``None`` if ambiguous (0 or 2+ outputs)."""
    path = Path(onnx_path)
    if not path.is_file():
        return None
    model = onnx.load(str(path))
    outs = list(model.graph.output)
    if len(outs) == 1:
        return outs[0].name
    return None
