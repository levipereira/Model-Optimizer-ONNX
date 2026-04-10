#!/usr/bin/env python3
"""Build a NumPy calibration tensor from COCO val images for ONNX PTQ (NVIDIA Model Optimizer)."""
# Preprocessing defaults match common Ultralytics YOLO ONNX exports: RGB, NCHW, /255, letterbox.

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from model_opt_yolo.io_checks import validate_existing_dir
from model_opt_yolo.logutil import add_logging_arguments, setup_logging
from model_opt_yolo.session_paths import default_calib_npy_path, default_calib_prep_log, run_timestamp

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path, limit: int | None) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    if limit is not None:
        if limit < len(paths):
            rng = random.Random(42)
            paths = rng.sample(paths, limit)
        paths.sort()
    return paths


def letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize image to new_shape with aspect ratio preserved (padding). BGR input."""
    shape = im.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)


def preprocess(
    path: Path,
    img_size: int,
    use_letterbox: bool,
    bgr: bool,
) -> np.ndarray:
    bgr_img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if use_letterbox:
        bgr_img, _, _ = letterbox(bgr_img, (img_size, img_size))
    else:
        bgr_img = cv2.resize(bgr_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if not bgr:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    else:
        tensor = bgr_img.astype(np.float32).transpose(2, 0, 1) / 255.0
    return tensor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build calib.npy from COCO (or any folder of) images for Model Optimizer ONNX PTQ."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory of images (e.g. data/coco/val2017).",
    )
    parser.add_argument(
        "--calibration_data_size",
        type=int,
        default=500,
        help="Number of images (TensorRT recommends >=500 for CNN/ViT-style models).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Square input size H=W (must match your ONNX preprocessing).",
    )
    parser.add_argument(
        "--no-letterbox",
        action="store_true",
        help="Disable letterbox; use plain resize to img_size (default: letterbox on).",
    )
    parser.add_argument(
        "--bgr",
        action="store_true",
        help="Keep BGR channel order (default: RGB, typical for PyTorch/YOLO exports).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Save calibration tensor as float16 (use if your ONNX expects FP16 activations).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output .npy path (default: <artifacts root>/calibration/calib_<dir>_sz<N>_n<M>_<timestamp>.npy).",
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    ts = run_timestamp()
    images_dir_name = Path(args.images_dir).name
    if args.output_path is None:
        args.output_path = str(
            default_calib_npy_path(
                images_dir_name=images_dir_name,
                img_size=args.img_size,
                n_images=args.calibration_data_size,
                ts=ts,
            )
        )
    log_path = args.log_file
    if log_path is None:
        log_path = str(default_calib_prep_log(images_dir_name=images_dir_name, ts=ts))

    setup_logging("coco_calib_prep", log_file=log_path, verbose=args.verbose)
    log = logging.getLogger("coco_calib_prep")

    use_letterbox = not args.no_letterbox
    err = validate_existing_dir(args.images_dir, label="Images directory")
    if err:
        log.error("%s", err)
        return 1
    root = Path(args.images_dir).expanduser().resolve()

    paths = list_images(root, args.calibration_data_size)
    if not paths:
        log.error("No images under %s", root)
        return 1

    out_file = Path(args.output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    batch: list[np.ndarray] = []
    for p in tqdm(paths, desc="Preprocessing"):
        batch.append(preprocess(p, args.img_size, use_letterbox, args.bgr))

    calib = np.stack(batch, axis=0)
    if args.fp16:
        calib = calib.astype(np.float16)

    np.save(str(out_file), calib)
    log.info(
        "Saved %s %s to %s (%d images, letterbox=%s, bgr=%s)",
        calib.shape,
        calib.dtype,
        out_file,
        len(paths),
        use_letterbox,
        args.bgr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
