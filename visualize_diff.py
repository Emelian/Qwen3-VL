#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize side-by-side JSON predictions (old vs new) under each image.

- Shows image on top.
- Under the image: left = old model (red), right = new model (green).
- Controls: Space → next image; Esc or window close → exit.

Assumptions:
- Both JSON files are lists of items like:
  {
    "image": "ILSVRC2012_val_00005391v2.jpg",
    "conversations": [
      {"from": "human", "value": "<image>...prompt..."},
      {"from": "gpt",   "value": "{\"watermarks\": 8, \"text\": [\"deer\"], ... }"}
    ]
  }
- The actual prediction to render is the JSON string in the "gpt" message's "value".

Usage:
    python viz_compare_preds.py --old old.json --new new.json --images-dir /path/to/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Visualize old vs new JSON predictions under each image.")
    p.add_argument("--old", required=True, type=Path, help="Path to JSON with predictions from the old/original model.")
    p.add_argument("--new", required=True, type=Path, help="Path to JSON with predictions from the new/LoRA model.")
    p.add_argument("--images-dir", required=True, type=Path, help="Root folder with images referenced in JSON.")
    p.add_argument("--height", type=int, default=1000,
                   help="Target display height (image will be scaled to fit; text panel scales accordingly). Default: 1000 px.")
    p.add_argument("--panel-ratio", type=float, default=0.35,
                   help="Fraction of total display height reserved for text panel (0-0.7). Default: 0.35.")
    p.add_argument("--font-size", type=int, default=18, help="Base font size for text panel. Auto-scaled per width.")
    p.add_argument("--wrap", type=int, default=60, help="Soft wrap width (chars) for pretty JSON lines.")
    p.add_argument("--union", action="store_true",
                   help="Iterate over union of image sets (by default, iterate in the order of --old and only images present in both).")
    return p.parse_args()


def load_pred_map(json_path: Path) -> Tuple[List[str], Dict[str, dict]]:
    """Load list JSON and convert to ordered names + map image->prediction dict.

    Args:
        json_path: Path to JSON file (list format as described).

    Returns:
        Tuple[List[str], Dict[str, dict]]: (ordered image names, mapping image->parsed prediction dict).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    order: List[str] = []
    mapping: Dict[str, dict] = {}
    for item in data:
        img_name = item.get("image")
        if not img_name:
            continue
        # Find the 'gpt' message
        gpt_msg = None
        for msg in item.get("conversations", []):
            if msg.get("from") == "gpt":
                gpt_msg = msg.get("value")
        if gpt_msg is None:
            continue
        # Some models might return extra whitespace or code fences — attempt robust parsing
        pred_dict = _safe_json_parse(gpt_msg)
        if pred_dict is None:
            # Fallback: store raw string for visibility
            pred_dict = {"_raw": gpt_msg}
        order.append(img_name)
        mapping[img_name] = pred_dict
    return order, mapping


def _safe_json_parse(s: str) -> Optional[dict]:
    """Try to parse a JSON string robustly.

    Args:
        s: JSON text (possibly with leading/trailing whitespace or fences).

    Returns:
        Parsed dict or None if parsing failed.
    """
    s = s.strip()
    # Strip common fences like ```json ... ```
    if s.startswith("```"):
        # take content between first and last fence
        parts = s.split("```")
        if len(parts) >= 3:
            s = "".join(parts[1:-1]).strip()
    # Some models might prefix with things like 'JSON:' or similar
    for prefix in ("JSON:", "json:", "Json:", "Output:", "Answer:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def pretty_json(d: dict, wrap: int = 60) -> List[str]:
    """Pretty-print dict into multiple wrapped lines.

    Args:
        d: Dict to print; if contains key "_raw", prints raw string.
        wrap: Approximate max characters per line.

    Returns:
        List[str]: Lines to render.
    """
    if "_raw" in d and isinstance(d["_raw"], str):
        text = d["_raw"]
        return _wrap_text(text, wrap)
    text = json.dumps(d, ensure_ascii=False, indent=2)
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        out.extend(_wrap_text(ln, wrap))
    return out


def _wrap_text(text: str, width: int) -> List[str]:
    """Soft wrap text by characters.

    Args:
        text: Input text (single line).
        width: Max characters per line.

    Returns:
        List[str]: Wrapped lines.
    """
    if len(text) <= width:
        return [text]
    res = []
    start = 0
    while start < len(text):
        res.append(text[start:start + width])
        start += width
    return res


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_show_and_wait(win_name: str, frame: np.ndarray) -> int:
    """Show frame and return pressed key code (or -1)."""
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(0) & 0xFF
    # Handle window close (Linux/Windows)
    try:
        visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        if visible < 1:
            return 27  # treat as Esc
    except Exception:
        pass
    return key


def _measure_text(draw, text: str, font) -> tuple[int, int]:
    """Return (w, h) for text using modern Pillow APIs with fallbacks."""
    try:
        # Pillow >= 8.0: preferred
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:
        try:
            # Font-level bbox (Pillow >= 8.0)
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        except Exception:
            # Heuristic fallback
            size = getattr(font, "size", 16)
            return int(len(text) * size * 0.6), int(size * 1.2)


def build_canvas(
    pil_img: Image.Image,
    left_lines: List[str],
    right_lines: List[str],
    panel_height: int,
    base_font_size: int,
) -> Image.Image:
    """Compose a canvas with image on top and two-column text panel below.

    Args:
        pil_img: PIL image (already resized to target display width).
        left_lines: Lines for left (old) text, red color.
        right_lines: Lines for right (new) text, green color.
        panel_height: Height in pixels for the bottom panel.
        base_font_size: Base font size (will be slightly adjusted by width).

    Returns:
        PIL Image with composition.
    """
    w, h = pil_img.size
    canvas = Image.new("RGB", (w, h + panel_height), (20, 20, 20))
    canvas.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Choose font (default PIL font if truetype not available)
    try:
        # A widely available monospace on many systems; if not present, fallback
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=base_font_size)
        bold = ImageFont.truetype("DejaVuSansMono-Bold.ttf", size=base_font_size)
    except Exception:
        font = ImageFont.load_default()
        bold = font

    # Titles
    margin = 8
    col_gap = 20
    col_w = (w - 3 * margin) // 2
    left_x = margin
    right_x = left_x + col_w + col_gap
    top_y = h + margin

    # Headers
    draw.text((left_x, top_y), "OLD (original)", fill=(220, 80, 80), font=bold)
    draw.text((right_x, top_y), "NEW (LoRA)", fill=(80, 220, 120), font=bold)

    # Compute starting y after header
    _, header_h = _measure_text(draw, "NEW (LoRA)", bold)
    y_left = top_y + header_h + 6
    y_right = y_left

    # Render lines with clipping if exceed panel
    line_spacing = int(base_font_size * 1.25)
    max_lines = max(0, (h + panel_height - y_left - margin) // line_spacing)

    # Draw left (red) and right (green)
    for i, ln in enumerate(left_lines[:max_lines]):
        draw.text((left_x, y_left + i * line_spacing), ln, fill=(255, 120, 120), font=font)
    for i, ln in enumerate(right_lines[:max_lines]):
        draw.text((right_x, y_right + i * line_spacing), ln, fill=(120, 255, 170), font=font)

    # Faint separators
    draw.line([(w // 2, h), (w // 2, h + panel_height)], fill=(60, 60, 60), width=1)
    draw.rectangle([(0, h), (w - 1, h + panel_height - 1)], outline=(60, 60, 60), width=1)

    return canvas


def main() -> None:
    """Entry point."""
    args = parse_args()

    order_old, map_old = load_pred_map(args.old)
    order_new, map_new = load_pred_map(args.new)

    set_old = set(map_old.keys())
    set_new = set(map_new.keys())

    if args.union:
        images = list(dict.fromkeys(order_old + order_new))  # preserve old order, then add the rest
    else:
        # Only images present in both, preserving the order of --old
        images = [im for im in order_old if im in set_new]

    if not images:
        print("No overlapping images to visualize. Use --union to iterate over all.", file=sys.stderr)
        sys.exit(1)

    win = "Predictions: OLD (red) vs NEW (green)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    for img_name in images:
        img_path = args.images_dir / img_name
        if not img_path.is_file():
            print(f"[WARN] Missing image file: {img_path}", file=sys.stderr)
            # allow skip with space, else Esc to quit
            blank = np.zeros((512, 1024, 3), dtype=np.uint8)
            cv2.putText(blank, f"Missing image: {img_name}", (30, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            key = cv_show_and_wait(win, blank)
            if key in (27, ord('q')):
                break
            continue

        # Load and scale image to target height portion
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}", file=sys.stderr)
            continue

        # Compute display sizes
        total_h = max(400, args.height)
        panel_h = int(np.clip(args.panel_ratio, 0.0, 0.7) * total_h)
        img_h_target = total_h - panel_h

        w, h = pil_img.size
        scale = img_h_target / float(h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Prepare text
        old_pred = map_old.get(img_name)
        new_pred = map_new.get(img_name)
        if old_pred is None and new_pred is None:
            continue

        # Wrap width roughly proportional to half canvas width and font size
        # If user specified wrap, use it; otherwise approximate from width
        wrap = args.wrap
        if args.wrap <= 0:
            approx_chars = max(40, (new_w // 2) // 8)  # heuristic
            wrap = approx_chars

        left_lines = pretty_json(old_pred if old_pred is not None else {"_raw": "(no prediction)"}, wrap=wrap)
        right_lines = pretty_json(new_pred if new_pred is not None else {"_raw": "(no prediction)"}, wrap=wrap)

        # Base font size slightly scaled by width (heuristic)
        base_font = max(12, int(args.font_size * (new_w / 1000.0)))

        # Compose canvas
        canvas_pil = build_canvas(pil_img, left_lines, right_lines, panel_h, base_font)
        frame = pil_to_cv(canvas_pil)

        # Title bar (OpenCV window title already set), also overlay filename
        cv2.putText(frame, img_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)

        key = cv_show_and_wait(win, frame)
        if key in (27, ord('q')):  # Esc or q
            break
        # Space or any other key → next
        # (If you want strictly space, uncomment next line)
        # if key != 32: break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
