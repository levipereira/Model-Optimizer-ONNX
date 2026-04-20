#!/usr/bin/env python3
"""Download COCO val2017 images and train/val annotations (calibration + mAP eval)."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm

from modelopt_onnx_ptq.logutil import add_logging_arguments, setup_logging

VAL_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _dir_nonempty(d: Path) -> bool:
    if not d.is_dir():
        return False
    try:
        next(d.iterdir())
    except StopIteration:
        return False
    return True


def _download_wget(url: str, dest: Path) -> None:
    wget = shutil.which("wget")
    assert wget is not None
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([wget, "-c", "-O", str(dest), url], check=True)


def _download_urllib(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "modelopt-onnx-ptq/1.0"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(dest, "wb") as f, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def download_file(url: str, dest: Path, log: logging.Logger) -> None:
    if shutil.which("wget"):
        try:
            log.info("Using wget (-c) for %s", dest.name)
            _download_wget(url, dest)
            return
        except subprocess.CalledProcessError as e:
            log.warning("wget failed (%s); falling back to urllib", e)
    log.info("Streaming download with urllib for %s", dest.name)
    _download_urllib(url, dest)


def extract_zip(zip_path: Path, out_dir: Path, log: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", zip_path, out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelopt-onnx-ptq download-coco",
        description="Download COCO val2017 images and 2017 annotations (instances_val2017.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/coco",
        help="Root directory (default: data/coco). Creates val2017/ and annotations/.",
    )
    add_logging_arguments(parser)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    setup_logging("download_coco", log_file=args.log_file, verbose=args.verbose)
    log = logging.getLogger("download_coco")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    val_dir = out / "val2017"
    val_zip = out / "val2017.zip"
    ann_zip = out / "annotations_trainval2017.zip"
    ann_json = out / "annotations" / "instances_val2017.json"

    # --- Images ---
    if _dir_nonempty(val_dir):
        log.info("Found existing %s — skipping image download.", val_dir)
    else:
        log.info("Downloading COCO val2017 images (~1 GB) to %s ...", val_zip)
        download_file(VAL_ZIP_URL, val_zip, log)
        extract_zip(val_zip, out, log)
        log.info("Done. Images: %s", val_dir)
        log.info("Optional: rm -f %s", val_zip)

    # --- Annotations ---
    if ann_json.is_file():
        log.info("Found existing %s — skipping annotations download.", ann_json)
    else:
        log.info("Downloading COCO annotations (~252 MB) to %s ...", ann_zip)
        download_file(ANN_ZIP_URL, ann_zip, log)
        extract_zip(ann_zip, out, log)
        log.info("Done. Annotations: %s", ann_json.parent)
        log.info("Optional: rm -f %s", ann_zip)

    log.info("COCO layout ready under %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
