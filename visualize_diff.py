#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize side-by-side JSON predictions (old vs new) under each image.

- Сверху изображение.
- Снизу: слева (красным) предикт старой модели, справа (зелёным) предикт новой (LoRA).
- Управление: Space → следующее изображение; Esc или закрытие окна → выход.
- Пересечение имён из двух JSON (порядок как в --old). Флаг --union — объединение.

Парсер предиктов:
- Ожидает, что в conversations есть сообщение {"from":"gpt","value": "..."}.
- Поддерживает строки вида:
    "value": "```json\n{ ... }\n```"
  и просто "``` ... ```", а также “сырой” JSON без код-блоков.

Отображение:
- Всегда пытаемся сериализовать как корректный JSON.
- Если парсинг/сериализация не удались — показываем строку: "ошибка сериализации".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize old vs new JSON predictions under each image.")
    p.add_argument("--old", required=True, type=Path, help="Path to JSON with predictions from the old/original model.")
    p.add_argument("--new", required=True, type=Path, help="Path to JSON with predictions from the new/LoRA model.")
    p.add_argument("--images-dir", required=True, type=Path, help="Root folder with images referenced in JSON.")
    p.add_argument("--height", type=int, default=1000, help="Target display height (image+panel). Default: 1000 px.")
    p.add_argument("--panel-ratio", type=float, default=0.35,
                   help="Fraction of total height reserved for text panel (0..0.7). Default: 0.35.")
    p.add_argument("--font-size", type=int, default=18, help="Base font size for text panel.")
    p.add_argument("--wrap", type=int, default=60, help="Soft wrap width (chars) for rendered JSON.")
    p.add_argument("--union", action="store_true",
                   help="Iterate over union of image sets (by default — intersection, ordered as --old).")
    return p.parse_args()


# ---------------- JSON utils ----------------

_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

def load_pred_map(json_path: Path) -> Tuple[List[str], Dict[str, Optional[dict]]]:
    """Load list JSON and convert to ordered names + map image->prediction dict (or None on parse error)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    order: List[str] = []
    mapping: Dict[str, Optional[dict]] = {}
    for item in data:
        img_name = item.get("image")
        if not img_name:
            continue
        gpt_msg = None
        for msg in item.get("conversations", []):
            if msg.get("from") == "gpt":
                gpt_msg = msg.get("value")
        if gpt_msg is None:
            # нет предсказания — считаем как ошибка сериализации
            order.append(img_name)
            mapping[img_name] = None
            continue

        pred_dict = _safe_json_parse(gpt_msg)
        # pred_dict == None -> ошибка парсинга/сериализации
        order.append(img_name)
        mapping[img_name] = pred_dict
    return order, mapping


def _extract_json_from_codeblock(s: str) -> Optional[str]:
    """Return JSON object string found inside ```json ... ``` (or ``` ... ```), else None."""
    m = _CODEBLOCK_RE.search(s)
    if m:
        return m.group(1)
    return None


def _safe_json_parse(s: str) -> Optional[dict]:
    """Parse the model's JSON string robustly, incl. code-fenced blocks. Return dict or None on failure."""
    if not isinstance(s, str):
        return None
    s = s.strip()

    # 1) Точный захват JSON из код-блока ```json ... ```
    inner = _extract_json_from_codeblock(s)
    if inner:
        try:
            obj = json.loads(inner)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass  # попробуем другие варианты

    # 2) Код-блок без 'json'
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`").strip()
        # Если первая строка — слово json
        if body.lower().startswith("json"):
            body = body[4:].lstrip("\n\r\t :")
        try:
            obj = json.loads(body)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    # 3) Стереть префиксы типа "JSON:", "Output:"
    for prefix in ("JSON:", "json:", "Json:", "Output:", "Answer:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break

    # 4) Попытаться распарсить как есть
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def serialize_lines(d: Optional[dict], wrap: int = 60) -> List[str]:
    """Try to serialize dict to formatted JSON lines. If fails or d is None -> 'ошибка сериализации'."""
    if not isinstance(d, dict):
        return ["ошибка сериализации"]
    try:
        # Стандартный сериализатор JSON: экранирует спецсимволы в строках по стандарту JSON.
        text = json.dumps(d, ensure_ascii=False, indent=2)
    except Exception:
        return ["ошибка сериализации"]

    # Разбиваем на строки и мягко переносим
    lines: List[str] = []
    for ln in text.splitlines():
        lines.extend(_wrap_text(ln, wrap))
    return lines


def _wrap_text(text: str, width: int) -> List[str]:
    if width <= 0 or len(text) <= width:
        return [text]
    res = []
    start = 0
    while start < len(text):
        res.append(text[start:start + width])
        start += width
    return res


# ---------------- Drawing utils ----------------

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Return (w, h) for text using modern Pillow APIs with fallbacks."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:
        try:
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        except Exception:
            size = getattr(font, "size", 16)
            return int(len(text) * size * 0.6), int(size * 1.2)


def build_canvas(
    pil_img: Image.Image,
    left_lines: List[str],
    right_lines: List[str],
    panel_height: int,
    base_font_size: int,
) -> Image.Image:
    """Compose a canvas with image on top and two-column text panel below."""
    w, h = pil_img.size
    canvas = Image.new("RGB", (w, h + panel_height), (20, 20, 20))
    canvas.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=base_font_size)
        bold = ImageFont.truetype("DejaVuSansMono-Bold.ttf", size=base_font_size)
    except Exception:
        font = ImageFont.load_default()
        bold = font

    margin = 8
    col_gap = 20
    col_w = (w - 3 * margin) // 2
    left_x = margin
    right_x = left_x + col_w + col_gap
    top_y = h + margin

    # Headers
    draw.text((left_x, top_y), "OLD (original)", fill=(220, 80, 80), font=bold)
    draw.text((right_x, top_y), "NEW (LoRA)", fill=(80, 220, 120), font=bold)

    _, header_h = _measure_text(draw, "NEW (LoRA)", bold)
    y_left = top_y + header_h + 6
    y_right = y_left

    line_spacing = int(base_font_size * 1.25)
    max_lines = max(0, (h + panel_height - y_left - margin) // line_spacing)

    for i, ln in enumerate(left_lines[:max_lines]):
        draw.text((left_x, y_left + i * line_spacing), ln, fill=(255, 120, 120), font=font)
    for i, ln in enumerate(right_lines[:max_lines]):
        draw.text((right_x, y_right + i * line_spacing), ln, fill=(120, 255, 170), font=font)

    # Separators
    draw.line([(w // 2, h), (w // 2, h + panel_height)], fill=(60, 60, 60), width=1)
    draw.rectangle([(0, h), (w - 1, h + panel_height - 1)], outline=(60, 60, 60), width=1)

    return canvas


# ---------------- Display backends (OpenCV with fallback) ----------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _init_cv2_window(win: str) -> bool:
    try:
        import cv2
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        return True
    except Exception:
        return False


def cv_show_and_wait(win_name: str, frame: np.ndarray) -> int:
    import cv2
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(0) & 0xFF
    try:
        visible = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
        if visible < 1:
            return 27  # Esc
    except Exception:
        pass
    return key


def mpl_show_and_wait(frame_bgr: np.ndarray) -> str:
    """Matplotlib fallback: returns 'space' or 'esc'."""
    import matplotlib.pyplot as plt
    frame_rgb = frame_bgr[:, :, ::-1]
    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)
    ax.axis("off")
    pressed = {"key": None, "closed": False}

    def on_key(event):
        pressed["key"] = event.key
        plt.close(fig)

    def on_close(event):
        pressed["closed"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()

    if pressed["closed"] or pressed["key"] in ("escape", "esc"):
        return "esc"
    if pressed["key"] in (" ", "space"):
        return "space"
    return "space"


# ---------------- Main ----------------

def main() -> None:
    args = parse_args()

    order_old, map_old = load_pred_map(args.old)
    order_new, map_new = load_pred_map(args.new)

    set_old = set(map_old.keys())
    set_new = set(map_new.keys())

    if args.union:
        images = list(dict.fromkeys(order_old + order_new))
    else:
        images = [im for im in order_old if im in set_new]

    if not images:
        print("No images to visualize (try --union).", file=sys.stderr)
        sys.exit(1)

    # Try OpenCV GUI; fallback to Matplotlib if not available
    USE_CV2 = False
    win_title = "Predictions: OLD (red) vs NEW (green)"
    try:
        import cv2  # noqa: F401
        USE_CV2 = _init_cv2_window(win_title)
    except Exception:
        USE_CV2 = False

    if not USE_CV2:
        print("[INFO] OpenCV GUI недоступен — используем Matplotlib fallback (Space/ESC).", file=sys.stderr)

    for img_name in images:
        img_path = args.images_dir / img_name
        if not img_path.is_file():
            msg = f"[WARN] Missing image file: {img_path}"
            print(msg, file=sys.stderr)
            # Заглушка
            h, w = 512, 1024
            pil = Image.new("RGB", (w, h), (30, 30, 30))
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((30, h // 2 - 12), f"Missing image: {img_name}", fill=(255, 80, 80), font=font)
            frame_bgr = np.array(pil)[:, :, ::-1]
            if USE_CV2:
                k = cv_show_and_wait(win_title, frame_bgr)
                if k in (27, ord('q')):
                    break
            else:
                k = mpl_show_and_wait(frame_bgr)
                if k == "esc":
                    break
            continue

        # Load + scale image to target height portion
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}", file=sys.stderr)
            continue

        total_h = max(400, args.height)
        panel_h = int(np.clip(args.panel_ratio, 0.0, 0.7) * total_h)
        img_h_target = total_h - panel_h

        w0, h0 = pil_img.size
        scale = img_h_target / float(h0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Prepare text blocks
        old_pred = map_old.get(img_name)   # dict | None
        new_pred = map_new.get(img_name)   # dict | None
        if old_pred is None and new_pred is None:
            left_lines = right_lines = ["ошибка сериализации"]
        else:
            wrap = args.wrap if args.wrap > 0 else max(40, (new_w // 2) // 8)
            left_lines = serialize_lines(old_pred, wrap=wrap)
            right_lines = serialize_lines(new_pred, wrap=wrap)

        base_font = max(12, int(args.font_size * (new_w / 1000.0)))
        canvas_pil = build_canvas(pil_img, left_lines, right_lines, panel_h, base_font)

        # Overlay filename
        try:
            draw = ImageDraw.Draw(canvas_pil)
            try:
                ui_font = ImageFont.truetype("DejaVuSansMono.ttf", size=max(12, int(base_font * 0.9)))
            except Exception:
                ui_font = ImageFont.load_default()
            draw.text((10, 10), img_name, fill=(230, 230, 230), font=ui_font)
        except Exception:
            pass

        if USE_CV2:
            frame = pil_to_cv(canvas_pil)
            k = cv_show_and_wait(win_title, frame)
            if k in (27, ord('q')):  # Esc/q
                break
        else:
            frame_bgr = np.array(canvas_pil)[:, :, ::-1]
            k = mpl_show_and_wait(frame_bgr)
            if k == "esc":
                break

    if USE_CV2:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
