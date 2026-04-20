#!/usr/bin/env python3
"""Evaluate a TensorRT engine on COCO val2017 — mAP@0.5:0.95.

Batch **B** may be dynamic; this eval loop uses **B=1** per image.

**Output formats** (``--output-format``; ``auto`` with optional ``--onnx`` infers the rest):

1. ``ultralytics_e2e`` — End-to-end Ultralytics export: single tensor (e.g. ``output0``),
   shape ``[B, N, 6]``, one row per post-NMS box (NMS **in the graph**). Alias: ``ultralytics``.

2. ``ultralytics_raw`` — **Default ONNX export** from many Ultralytics versions: no NMS in the
   graph; shape ``[B, 4+nc, num_anchors]`` (e.g. ``[1, 84, 8400]`` at 640px). NMS runs in
   ``eval-trt``. Class logits are sigmoid-scored; boxes ``cx, cy, w, h`` in letterboxed space.

3. ``deepstream_yolo`` — Single tensor (e.g. ``output``), packed ``[B, num_anchors, 6]`` as in
   DeepStream-Yolo ``nvdsparsebbox_Yolo``; per-class NMS in ``eval-trt``.

Usage:
    modelopt-onnx-ptq eval-trt --output-format auto --onnx model.onnx \\
        --engine model.engine --images data/coco/val2017 \\
        --annotations data/coco/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from modelopt_onnx_ptq.io_checks import validate_existing_dir, validate_readable_file
from modelopt_onnx_ptq.logutil import add_logging_arguments, setup_logging
from modelopt_onnx_ptq.onnx_eval_layout import (
    infer_default_output_tensor_name_from_onnx,
    infer_eval_output_format_from_onnx,
    infer_eval_output_format_from_trt_outputs,
    normalize_eval_output_format,
)
from modelopt_onnx_ptq.session_paths import (
    artifacts_root,
    default_eval_session_log,
    effective_session_id,
    run_timestamp,
)


# ---------------------------------------------------------------------------
# TensorRT + PyCUDA inference
# ---------------------------------------------------------------------------
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# Register libnvinfer_plugin before deserializing engines that use TRT plugins.
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class HostDeviceMem:
    __slots__ = ("host", "device", "shape", "name")

    def __init__(self, host_mem, device_mem, shape, name):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.name = name


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        mem = HostDeviceMem(host_mem, device_mem, shape, name)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(mem)
        else:
            outputs.append(mem)
    return inputs, outputs, bindings, stream


def do_inference(context, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    stream.synchronize()
    return {out.name: out.host.reshape(out.shape) for out in outputs}


# ---------------------------------------------------------------------------
# Preprocessing (letterbox — same as modelopt_onnx_ptq.calib_prep)
# ---------------------------------------------------------------------------

def letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess_image(path: str, img_size: int) -> tuple[np.ndarray, dict]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lb, ratio, (dw, dh) = letterbox(rgb, (img_size, img_size))
    tensor = lb.astype(np.float32).transpose(2, 0, 1) / 255.0
    meta = {"orig_h": orig_h, "orig_w": orig_w, "ratio": ratio, "pad": (dw, dh)}
    return tensor, meta


# ---------------------------------------------------------------------------
# Postprocessing — single tensor [B, N, 6]
# ---------------------------------------------------------------------------

def nms_xyxy_one_class(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS; boxes [N,4] xyxy, scores [N]. Returns indices into boxes."""
    if boxes.size == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[rest] - inter
        ious = np.where(union > 0, inter / union, 0.0)
        order = rest[ious <= iou_thres]
    return np.array(keep, dtype=np.int64)


def nms_per_class_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thres: float,
) -> np.ndarray:
    """Per-class greedy NMS; boxes [N,4] xyxy, scores [N], classes [N] int."""
    if boxes.size == 0:
        return np.zeros(0, dtype=np.int64)
    keep_all: list[int] = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        b = boxes[idx]
        s = scores[idx]
        picked = nms_xyxy_one_class(b, s, iou_thres)
        keep_all.extend(idx[picked].tolist())
    return np.array(keep_all, dtype=np.int64)


def letterbox_to_original(
    boxes_xyxy: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Rescale xyxy from letterboxed input tensor coords to original image coords."""
    out = boxes_xyxy.astype(np.float32).copy()
    ratio = meta["ratio"]
    dw, dh = meta["pad"]
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, meta["orig_w"])
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, meta["orig_h"])
    return out


def postprocess_single_tensor_xyxy(
    pred: np.ndarray,
    meta: dict,
    conf_thres: float,
    *,
    apply_nms: bool,
    iou_thres: float,
) -> np.ndarray:
    """Decode ``[B, N, 6]`` -> Nx6 in original image coords (x1,y1,x2,y2,conf,cls)."""
    if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[2] != 6:
        raise ValueError(
            f"Expected pred shape [1, N, 6], got {pred.shape}"
        )
    p = pred[0]  # [N, 6]
    boxes = p[:, :4].astype(np.float32)
    scores = p[:, 4].astype(np.float32)
    classes = p[:, 5]

    mask = scores > conf_thres
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    cls_int = np.rint(classes).astype(np.int64)
    if apply_nms:
        keep = nms_per_class_xyxy(boxes, scores, cls_int, iou_thres)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep].astype(np.float32)

    boxes_orig = letterbox_to_original(boxes, meta)
    return np.column_stack(
        [boxes_orig, scores[:, None], classes[:, None].astype(np.float32)]
    ).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))).astype(np.float32)


def postprocess_ultralytics_raw(
    pred: np.ndarray,
    meta: dict,
    conf_thres: float,
    iou_thres: float,
) -> np.ndarray:
    """Decode ``[1, 4+nc, num_anchors]`` -> Nx6 in original image coords (x1,y1,x2,y2,conf,cls)."""
    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError(f"Expected pred shape [1, 4+nc, num_anchors], got {pred.shape}")
    _, c, _ = pred.shape
    nc = c - 4
    if nc < 1:
        raise ValueError(f"Need at least one class channel; got C={c}.")
    x = pred[0].astype(np.float32).transpose(1, 0)
    boxes_xywh = x[:, :4]
    cls_prob = _sigmoid(x[:, 4:])
    scores = cls_prob.max(axis=1).astype(np.float32)
    cls_ids = cls_prob.argmax(axis=1).astype(np.int64)
    mask = scores > conf_thres
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]
    cx, cy, w, h = boxes_xywh.T
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    keep = nms_per_class_xyxy(boxes, scores, cls_ids, iou_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    boxes_orig = letterbox_to_original(boxes, meta)
    return np.column_stack(
        [boxes_orig, scores[:, None], cls_ids.astype(np.float32)]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# COCO class mapping (80 training classes -> 91 COCO category IDs)
# ---------------------------------------------------------------------------

COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _input_tensor_names(engine) -> list[str]:
    """Names of INPUT tensors in engine binding order (same order as ``allocate_buffers`` inputs)."""
    out: list[str] = []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            out.append(n)
    return out


def _output_tensor_names(engine) -> set[str]:
    names: set[str] = set()
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            names.add(n)
    return names


def _trt_output_specs(engine) -> list[tuple[str, tuple[int, ...]]]:
    """TensorRT output tensor names and shapes (for layout auto-detection)."""
    out: list[tuple[str, tuple[int, ...]]] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            out.append((name, tuple(engine.get_tensor_shape(name))))
    return out


def run_eval(
    engine_path: str,
    images_dir: str,
    annotations_json: str,
    img_size: int = 640,
    conf_thres: float = 0.001,
    save_json: str | None = None,
    *,
    output_format: str,
    onnx_path: str | None = None,
    output_tensor: str | None = None,
    iou_thres: float = 0.45,
    log: logging.Logger,
) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    output_format = normalize_eval_output_format(output_format)

    # Load engine
    log.info("Loading TRT engine: %s", engine_path)
    runtime = trt.Runtime(TRT_LOGGER)
    try:
        with open(engine_path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        log.error("Cannot read engine file %s: %s", engine_path, exc)
        raise RuntimeError(f"Cannot read engine: {exc}") from exc
    engine = runtime.deserialize_cuda_engine(raw)
    if engine is None:
        log.error("TensorRT failed to deserialize engine (invalid or incompatible plan): %s", engine_path)
        raise RuntimeError("TensorRT engine deserialization failed")

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

    # I/O names come from the plan: num_io_tensors + get_tensor_name(i) + get_tensor_mode (TensorRT 8+ API).
    log.info("Engine I/O tensors (%d bindings):", engine.num_io_tensors)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        mode = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
        log.info("  %6s  %-15s  %s", mode, name, list(shape))

    if inputs:
        in_names = _input_tensor_names(engine)
        log.info(
            "Letterboxed image is fed to input %r (first input binding; %d input tensor(s) on engine).",
            inputs[0].name,
            len(in_names),
        )

    input_shape = inputs[0].shape
    engine_img_size = input_shape[2]
    if engine_img_size != img_size:
        log.info("Overriding img_size to %s (from engine)", engine_img_size)
        img_size = engine_img_size

    if output_format == "auto":
        try:
            if onnx_path:
                output_format = infer_eval_output_format_from_onnx(onnx_path)
                log.info("Auto output-format from ONNX %s → %s", onnx_path, output_format)
            else:
                output_format = infer_eval_output_format_from_trt_outputs(_trt_output_specs(engine))
                log.info("Auto output-format from engine bindings → %s", output_format)
        except ValueError as exc:
            raise ValueError(
                f"{exc} Use an explicit --output-format, or pass --onnx pointing to the "
                "same graph used to build the engine."
            ) from exc

    out_names = _output_tensor_names(engine)
    single_tensor_name: str | None = None
    if output_format in ("ultralytics_e2e", "deepstream_yolo", "ultralytics_raw"):
        if not out_names:
            raise ValueError("Engine has no output tensors")
        if output_tensor is not None:
            if output_tensor not in out_names:
                raise ValueError(
                    f"--output-tensor {output_tensor!r} is not an output. Available: {sorted(out_names)}"
                )
            single_tensor_name = output_tensor
        else:
            if onnx_path is not None:
                inferred_out = infer_default_output_tensor_name_from_onnx(onnx_path)
                if inferred_out is not None and inferred_out in out_names:
                    single_tensor_name = inferred_out
                    log.info("Auto output tensor from ONNX: %s", single_tensor_name)
                else:
                    single_tensor_name = None
            else:
                single_tensor_name = None
            if single_tensor_name is None:
                if len(out_names) == 1:
                    single_tensor_name = next(iter(out_names))
                else:
                    for cand in ("output0", "output", "output1"):
                        if cand in out_names:
                            single_tensor_name = cand
                            break
                    if single_tensor_name is None:
                        raise ValueError(
                            "Multiple output tensors; set --output-tensor to one of: "
                            f"{sorted(out_names)}"
                        )
        log.info("Detection output tensor: %s", single_tensor_name)

    # Load COCO annotations
    try:
        coco_gt = COCO(annotations_json)
    except Exception as exc:
        log.error("Cannot load COCO annotations %s: %s", annotations_json, exc)
        raise RuntimeError(f"Invalid or unreadable annotations file: {annotations_json}") from exc
    img_ids = sorted(coco_gt.getImgIds())
    log.info("COCO images: %d", len(img_ids))

    jdict: list[dict] = []
    t_pre, t_inf, t_post = 0.0, 0.0, 0.0

    for img_info in tqdm(coco_gt.loadImgs(img_ids), desc="Evaluating"):
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            continue

        # Preprocess
        t0 = time.perf_counter()
        tensor, meta = preprocess_image(img_path, img_size)
        t_pre += time.perf_counter() - t0

        # Inference
        t0 = time.perf_counter()
        inputs[0].host = np.ascontiguousarray(tensor)
        result = do_inference(context, inputs, outputs, stream)
        t_inf += time.perf_counter() - t0

        t0 = time.perf_counter()
        if output_format == "ultralytics_e2e":
            if single_tensor_name is None:
                raise RuntimeError("internal: single_tensor_name unset for ultralytics_e2e")
            dets = postprocess_single_tensor_xyxy(
                result[single_tensor_name],
                meta,
                conf_thres,
                apply_nms=False,
                iou_thres=iou_thres,
            )
        elif output_format == "deepstream_yolo":
            if single_tensor_name is None:
                raise RuntimeError("internal: single_tensor_name unset for deepstream_yolo")
            dets = postprocess_single_tensor_xyxy(
                result[single_tensor_name],
                meta,
                conf_thres,
                apply_nms=True,
                iou_thres=iou_thres,
            )
        elif output_format == "ultralytics_raw":
            if single_tensor_name is None:
                raise RuntimeError("internal: single_tensor_name unset for ultralytics_raw")
            dets = postprocess_ultralytics_raw(
                result[single_tensor_name],
                meta,
                conf_thres,
                iou_thres,
            )
        else:
            raise ValueError(f"Unknown output_format: {output_format!r}")
        t_post += time.perf_counter() - t0

        image_id = img_info["id"]
        for det in dets:
            x1, y1, x2, y2, score, cls_id = det
            coco_cls = COCO80_TO_COCO91[int(cls_id)] if int(cls_id) < len(COCO80_TO_COCO91) else int(cls_id)
            jdict.append({
                "image_id": image_id,
                "category_id": coco_cls,
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(score), 5),
            })

    n = max(len(img_ids), 1)
    log.info(
        "Speed per image: preprocess %.1fms, inference %.1fms, postprocess %.1fms",
        t_pre / n * 1e3,
        t_inf / n * 1e3,
        t_post / n * 1e3,
    )
    log.info("Total detections: %d", len(jdict))

    # Save predictions JSON
    if save_json is None:
        stem = Path(engine_path).stem
        save_json = str(artifacts_root() / "predictions" / f"{stem}_predictions.json")
    Path(save_json).parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(jdict, f)
    log.info("Predictions saved: %s", save_json)

    # COCO eval
    if len(jdict) == 0:
        log.warning("No detections — cannot compute mAP.")
        return

    log.info("--- COCO mAP Evaluation ---")
    coco_dt = coco_gt.loadRes(save_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    # pycocotools prints AP/AR tables to stdout — capture so session log file includes them.
    coco_buf = io.StringIO()
    with redirect_stdout(coco_buf):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    coco_text = coco_buf.getvalue().rstrip()
    if coco_text:
        sys.stdout.write(coco_text + ("\n" if not coco_text.endswith("\n") else ""))
        sys.stdout.flush()
        log.info("--- pycocotools COCOeval (stdout) ---\n%s", coco_text)

    map_val = coco_eval.stats[0]
    map50 = coco_eval.stats[1]
    log.info("mAP@0.5:0.95 = %.4f", map_val)
    log.info("mAP@0.5      = %.4f", map50)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TensorRT YOLO engines on COCO val2017 (mAP). "
            "Use --output-format auto (--onnx recommended) or ultralytics_e2e / ultralytics_raw / "
            "deepstream_yolo (legacy alias: ultralytics → ultralytics_e2e)."
        )
    )
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to TensorRT .engine file.")
    parser.add_argument(
        "--output-format",
        type=str,
        required=True,
        choices=(
            "auto",
            "ultralytics_e2e",
            "ultralytics",
            "ultralytics_raw",
            "deepstream_yolo",
        ),
        help=(
            "auto: infer from --onnx (recommended) or engine I/O. "
            "ultralytics_e2e: [B,N,6] with NMS in graph (alias: ultralytics). "
            "ultralytics_raw: [B,4+nc,anchors] typical default ONNX (NMS in eval). "
            "deepstream_yolo: [B,num_anchors,6] DeepStream-style (NMS in eval)."
        ),
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional ONNX path (same graph as the engine). With --output-format auto, used for "
            "layout detection; if the ONNX has a single output, that name is used when picking "
            "the detection tensor (must exist on the engine)."
        ),
    )
    parser.add_argument(
        "--output-tensor",
        type=str,
        default=None,
        help=(
            "Override detection output name. If omitted: with --onnx and one graph output, use "
            "that name when present on the engine; else single engine output; else try "
            "output0, output, output1."
        ),
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="IoU threshold for per-class NMS (deepstream_yolo and ultralytics_raw).",
    )
    parser.add_argument("--images", type=str, default="data/coco/val2017",
                        help="Path to COCO val2017 images directory.")
    parser.add_argument("--annotations", type=str,
                        default="data/coco/annotations/instances_val2017.json",
                        help="Path to COCO instances_val2017.json.")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size (overridden by engine if different).")
    parser.add_argument("--conf-thres", type=float, default=0.001,
                        help="Confidence threshold for filtering detections.")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Output predictions JSON path.")
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Optional pipeline session id: when set (and --log-file is omitted), "
            "logs go under artifacts/pipeline_e2e/sessions/<id>/predictions/logs/. "
            "If omitted, SESSION_ID is used when set."
        ),
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    onnx_err = None
    if args.onnx is not None:
        onnx_err = validate_readable_file(args.onnx, label="ONNX model")

    for err in (
        validate_readable_file(args.engine, label="TensorRT engine"),
        validate_existing_dir(args.images, label="Images directory"),
        validate_readable_file(args.annotations, label="COCO annotations"),
        onnx_err,
    ):
        if err:
            print(err, file=sys.stderr)
            return 1

    ts = run_timestamp()
    log_path = args.log_file
    sid = effective_session_id(args.session_id)
    if log_path is None:
        log_path = str(
            default_eval_session_log(engine_stem=Path(args.engine).stem, ts=ts, session_id=sid)
        )

    setup_logging("eval_trt", log_file=log_path, verbose=args.verbose)
    log = logging.getLogger("eval_trt")

    try:
        run_eval(
            engine_path=args.engine,
            images_dir=args.images,
            annotations_json=args.annotations,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            save_json=args.save_json,
            output_format=args.output_format,
            onnx_path=args.onnx,
            output_tensor=args.output_tensor,
            iou_thres=args.iou_thres,
            log=log,
        )
    except (RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
