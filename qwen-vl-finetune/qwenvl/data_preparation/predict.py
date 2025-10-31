#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate Qwen2.5-VL predictions in Qwen-VL finetune format (<root>/predictions.json).

Output JSON is a list of items like:
[
  {
    "image": "relative/path/to/img.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nReturn STRICT JSON with keys ..."},
      {"from": "gpt",   "value": "{\"watermarks\": 2, \"text\": [], \"main object\": \"logo\", \"style\": \"photo\"}"}
    ]
  },
  ...
]

Usage:
  python generate_predictions_qwen_ftformat.py \
      --root /path/to/images_root \
      --device cuda \
      --save-every 10 \
      --max-new-tokens 256 \
      --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

STYLES = [
    "photo", "screenshot", "portrait", "landscape", "clipart",
    "comic", "anime", "render", "concept", "illustration",
    "minimalist", "vintage", "poster", "meme", "infographic",
    "sketch", "watercolor", "digital", "fashion", "documentary",
]

WATERMARK_PROMPT = (
    "You are a dict generator. Count visible watermarks (logos or text overlays). "
    "Ignore natural scene text. Return strictly: {\"watermarks\": <integer>}."
)
MAIN_OBJECT_PROMPT = (
    "You are a dict generator. Name the single main object (1–3 lowercase words). "
    "Return strictly: {\"main object\": \"<string>\"}."
)
STYLE_PROMPT_TEMPLATE = (
    "You are a dict generator. Classify visual style. "
    "Choose strictly one from this list: {choices}. "
    "Return strictly: {{\"style\": \"<string>\"}}."
)
TEXT_PROMPT = (
    "You are a dict generator. Read all CLEARLY READABLE WORDS in the image. "
    "Normalize to lowercase ASCII where possible, strip punctuation except hyphens inside words. "
    "Return strictly: {\"text\": [\"word1\", \"word2\", ...]}. If none: {\"text\": []}."
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions with Qwen2.5-VL into <root>/predictions.json")
    p.add_argument("--root", type=Path, required=True, help="Images root (scanned recursively)")
    p.add_argument("--device", type=str, default="cuda", help="Device: 'cuda', 'cuda:0' or 'cpu'")
    p.add_argument("--save-every", type=int, default=10, help="Save JSON every N predictions (0 disables)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    return p.parse_args(argv)


def scan_images(image_root: Path) -> List[str]:
    rel_paths: List[str] = []
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel_paths.append(path.relative_to(image_root).as_posix())
    return rel_paths


def open_image_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RuntimeError(f"Cannot open image {path}: {exc}") from exc


def atomic_save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def errors_path(root: Path) -> Path:
    return root / "errors.txt"


def log_error(root: Path, rel_image: str, exc: BaseException) -> None:
    with errors_path(root).open("a", encoding="utf-8") as f:
        f.write(f"[{rel_image}] {type(exc).__name__}: {exc}\n")


class QwenPredictor:
    """Lightweight Qwen2.5-VL predictor for four JSON tasks per image."""

    WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

    def __init__(self, device: str, max_new_tokens: int, temperature: float):
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=self._select_dtype(device),
        ).to(device)
        self.model.eval()
        tok = self.processor.tokenizer
        self.pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        self.autocast_dtype = self._autocast_dtype(device)

    @staticmethod
    def _select_dtype(device: str) -> torch.dtype:
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    @staticmethod
    def _autocast_dtype(device: str) -> Optional[torch.dtype]:
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None

    def _autocast_ctx(self):
        return torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()

    def _run_prompt(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            with self._autocast_ctx():
                out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, input_len:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _parse_json_key(raw: str, key: str) -> Any:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for key '{key}': {exc}: {raw}") from exc
        if key not in parsed:
            raise ValueError(f"Missing '{key}' in: {raw}")
        return parsed[key]

    @classmethod
    def _normalize_words(cls, words: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for w in words:
            if not isinstance(w, str):
                w = str(w)
            w = w.strip().lower()
            w = re.sub(r"^[^a-z0-9-]+|[^a-z0-9-]+$", "", w)
            if not w:
                continue
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out[:200]

    @classmethod
    def _parse_text_list(cls, raw: str) -> List[str]:
        try:
            maybe = json.loads(raw)
            if isinstance(maybe, dict) and "text" in maybe:
                val = maybe["text"]
                if isinstance(val, list):
                    return cls._normalize_words(val)
                if isinstance(val, str):
                    return cls._normalize_words(cls.WORD_RE.findall(val.lower()))
        except json.JSONDecodeError:
            pass
        try:
            maybe_list = json.loads(raw)
            if isinstance(maybe_list, list):
                return cls._normalize_words(maybe_list)
        except json.JSONDecodeError:
            pass
        return cls._normalize_words(cls.WORD_RE.findall(raw.lower()))

    def predict_fields(self, img: Image.Image) -> Dict[str, Any]:
        w_raw = self._run_prompt(img, WATERMARK_PROMPT)
        m_raw = self._run_prompt(img, MAIN_OBJECT_PROMPT)
        s_raw = self._run_prompt(img, STYLE_PROMPT_TEMPLATE.format(choices="{" + ", ".join(STYLES) + "}"))
        t_raw = self._run_prompt(img, TEXT_PROMPT)

        watermarks = self._parse_json_key(w_raw, "watermarks")
        if isinstance(watermarks, str):
            watermarks = int(watermarks) if watermarks.isdigit() else int(float(watermarks))
        if not isinstance(watermarks, int):
            watermarks = int(watermarks)

        main_object = self._parse_json_key(m_raw, "main object")
        main_object = str(main_object).strip().lower()

        style = self._parse_json_key(s_raw, "style")
        style = str(style).strip().lower()
        if style not in STYLES:
            style = "unknown"

        words = self._parse_text_list(t_raw)

        return {"watermarks": watermarks, "text": words, "main object": main_object, "style": style}


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    out_path = root / "predictions.json"
    rel_images = scan_images(root)
    predictor = QwenPredictor(args.device, args.max_new_tokens, args.temperature)

    items: List[Dict[str, Any]] = []
    since_save = 0

    for rel in tqdm(rel_images, desc="Predicting", unit="img"):
        try:
            img = open_image_rgb(root / rel)
            fields = predictor.predict_fields(img)
            gpt_payload = json.dumps(fields, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            log_error(root, rel, e)
            gpt_payload = json.dumps(
                {"watermarks": None, "text": [], "main object": "", "style": "unknown"},
                ensure_ascii=False,
            )

        item = {
            "image": rel,
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
                {"from": "gpt", "value": gpt_payload},
            ],
        }
        items.append(item)

        since_save += 1
        if args.save_every > 0 and since_save >= args.save_every:
            atomic_save_json(out_path, items)
            since_save = 0

    atomic_save_json(out_path, items)


if __name__ == "__main__":
    main()
