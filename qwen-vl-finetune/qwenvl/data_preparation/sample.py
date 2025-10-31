#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare a balanced subset and QWEN-style annotations for watermark counting.

Dataset layout:
  <root>/
    images/{train,val}/*.{jpg,jpeg,png,bmp,webp,tiff}
    labels/{train,val}/*.txt

This script:
  1) Reads label .txt files and counts non-empty, non-comment lines (watermark detections).
  2) Samples up to N_ZERO images with zero watermarks and up to N_NONZERO with non-zero watermarks.
  3) Copies selected images into <output>/images (flat directory, resolves filename collisions).
  4) Writes <output>/annotations.json in QWEN training format. Only "watermarks" is filled; other fields are null.

Usage:
  python prepare_qwen_watermarks.py \
      --root /path/to/dataset \
      --output /path/to/output_dir \
      --n-zero 2500 \
      --n-nonzero 2500 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare QWEN annotations for watermark counting.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing images/ and labels/")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-zero", type=int, default=2500, help="Target count with zero watermarks")
    parser.add_argument("--n-nonzero", type=int, default=2500, help="Target count with non-zero watermarks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def find_images(root: Path) -> Dict[Tuple[str, str], Path]:
    """Return mapping (split, stem) -> image_path for images in images/{train,val}."""
    out: Dict[Tuple[str, str], Path] = {}
    images_dir = root / "images"
    for split in ("train", "val"):
        split_dir = images_dir / split
        if not split_dir.is_dir():
            logging.warning("Missing images split: %s", split_dir)
            continue
        for p in split_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out[(split, p.stem)] = p
    return out


def count_label_lines(label_path: Path) -> int:
    """Count non-empty, non-comment lines in a YOLO-style txt label."""
    def _count(fp: Path, enc: str) -> int:
        with fp.open("r", encoding=enc) as f:
            n = 0
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                n += 1
            return n

    try:
        return _count(label_path, "utf-8")
    except UnicodeDecodeError:
        return _count(label_path, "latin-1")


def collect_stats(root: Path) -> List[Dict]:
    """Scan labels/{train,val} and produce a list of records with watermark counts."""
    records: List[Dict] = []
    labels_dir = root / "labels"
    for split in ("train", "val"):
        split_dir = labels_dir / split
        if not split_dir.is_dir():
            logging.warning("Missing labels split: %s", split_dir)
            continue
        for p in split_dir.rglob("*.txt"):
            wm = count_label_lines(p)
            records.append({"split": split, "stem": p.stem, "label_path": p, "watermarks": wm})
    return records


def match_images(records: List[Dict], img_index: Dict[Tuple[str, str], Path]) -> List[Dict]:
    """Attach image_path to each record if available; drop records without matching image."""
    matched: List[Dict] = []
    missing = 0
    for r in records:
        key = (r["split"], r["stem"])
        img_path = img_index.get(key)
        if img_path is None:
            missing += 1
            continue
        r2 = dict(r)
        r2["image_path"] = img_path
        r2["image_name"] = img_path.name
        matched.append(r2)
    if missing:
        logging.warning("Labels without matching images: %d", missing)
    return matched


def sample_balanced(records: List[Dict], n_zero: int, n_nonzero: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Sample up to n_zero with wm==0 and up to n_nonzero with wm>0 across all splits."""
    zeros = [r for r in records if r["watermarks"] == 0]
    nonzeros = [r for r in records if r["watermarks"] > 0]

    rng0 = random.Random(seed)
    rng1 = random.Random(seed + 1)
    rng0.shuffle(zeros)
    rng1.shuffle(nonzeros)

    take_zero = min(n_zero, len(zeros))
    take_nonzero = min(n_nonzero, len(nonzeros))

    if take_zero < n_zero:
        logging.warning("Requested %d zero-watermark images, but only %d available. Taking %d.",
                        n_zero, len(zeros), take_zero)
    if take_nonzero < n_nonzero:
        logging.warning("Requested %d non-zero-watermark images, but only %d available. Taking %d.",
                        n_nonzero, len(nonzeros), take_nonzero)

    return zeros[:take_zero], nonzeros[:take_nonzero]


def ensure_unique_name(dst_dir: Path, base_name: str, split: str) -> str:
    """Ensure unique filename in dst_dir; resolve collisions deterministically."""
    candidate = base_name
    stem = Path(base_name).stem
    ext = Path(base_name).suffix
    if not (dst_dir / candidate).exists():
        return candidate
    candidate = f"{stem}_{split}{ext}"
    if not (dst_dir / candidate).exists():
        return candidate
    i = 1
    while (dst_dir / candidate).exists():
        candidate = f"{stem}_{split}_{i}{ext}"
        i += 1
    return candidate


def copy_and_build_annotations(selected: List[Dict], out_images_dir: Path, prompt_text: str) -> List[Dict]:
    """Copy images and build QWEN annotations list."""
    out_images_dir.mkdir(parents=True, exist_ok=True)
    ann: List[Dict] = []
    for rec in tqdm(selected, desc="Copying & annotating"):
        src = rec["image_path"]
        split = rec["split"]
        final_name = ensure_unique_name(out_images_dir, src.name, split)
        dst = out_images_dir / final_name
        shutil.copy2(src, dst)
        ann.append(
            {
                "image": f"images/{final_name}",
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            "<image>\n"
                            "Analyze the image and return STRICT JSON with EXACTLY these keys and types:\n"
                            "{\n"
                            '  \"watermarks\": <integer>,\n'
                            '  \"text\": <array of strings>,\n'
                            '  \"main object\": <string>,\n'
                            '  \"style\": <string>\n'
                            "}\n"
                            "Do not add any extra text or explanation."
                        ),
                    },
                    {
                        "from": "gpt",
                        "value": json.dumps(
                            {
                                "watermarks": int(rec["watermarks"]),
                                "text": None,
                                "main object": None,
                                "style": None,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            }
        )
    return ann


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    random.seed(args.seed)

    root = args.root
    output = args.output
    out_images = output / "images"
    out_annotations = output / "annotations.json"

    img_index = find_images(root)
    logging.info("Found images: %d", len(img_index))

    records = collect_stats(root)
    logging.info("Found label files: %d", len(records))

    matched = match_images(records, img_index)
    logging.info("Matched pairs: %d", len(matched))

    zeros, nonzeros = sample_balanced(matched, args.n_zero, args.n_nonzero, args.seed)
    selected = zeros + nonzeros
    random.Random(args.seed).shuffle(selected)
    logging.info("Selected zeros: %d; non-zeros: %d; total: %d", len(zeros), len(nonzeros), len(selected))

    annotations = copy_and_build_annotations(selected, out_images, prompt_text="")
    output.mkdir(parents=True, exist_ok=True)
    with out_annotations.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    logging.info("Wrote annotations to: %s", out_annotations)


if __name__ == "__main__":
    main()
