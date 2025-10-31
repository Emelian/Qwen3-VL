#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate Qwen2.5-VL predictions (one prompt, raw string output) in Qwen-VL finetune format.

Output JSON (<root>/predictions.json) is a list like:
[
  {
    "image": "relative/path/to/img.jpg",
    "conversations": [
      {"from": "human", "value": "<image>"},
      {"from": "gpt",   "value": "{\"watermarks\": 2, \"text\": [], \"main object\": \"logo\", \"style\": \"photo\"}"}
    ]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions with Qwen2.5-VL into <root>/predictions.json (one prompt, raw output)")
    p.add_argument("--root", type=Path, required=True, help="Images root (scanned recursively)")
    # p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF model id")
    p.add_argument("--model-id", type=str, default="output", help="HF model id or local path")
    p.add_argument("--device", type=str, default="cuda", help="Device: 'cuda', 'cuda:0' or 'cpu'")
    p.add_argument("--save-every", type=int, default=10, help="Save JSON every N predictions (0 disables)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    return p.parse_args(argv)


def scan_images(image_root: Path) -> List[str]:
    return [
        p.relative_to(image_root).as_posix()
        for p in sorted(image_root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


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


def log_error(root: Path, rel_image: str, exc: BaseException) -> None:
    with (root / "errors.txt").open("a", encoding="utf-8") as f:
        f.write(f"[{rel_image}] {type(exc).__name__}: {exc}\n")


class QwenOnePromptRaw:
    """Single-prompt Qwen2.5-VL inference, returns raw model output as string."""

    def __init__(self, model_id: str, device: str, max_new_tokens: int, temperature: float):
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
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

    def run_prompt(self, image: Image.Image, prompt: str) -> str:
        """Return raw model text output for a single image."""
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]
        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.pad_token_id,
            "temperature": self.temperature if do_sample else None,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            with self._autocast_ctx():
                out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[:, input_len:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    out_path = root / "predictions.json"
    rel_images = scan_images(root)

    predictor = QwenOnePromptRaw(args.model_id, args.device, args.max_new_tokens, args.temperature)

    items: List[Dict[str, Any]] = []
    since_save = 0

    for rel in tqdm(rel_images, desc="Predicting", unit="img"):
        try:
            img = open_image_rgb(root / rel)
            raw_output = predictor.run_prompt(img, "<image>")
        except Exception as e:  # noqa: BLE001
            log_error(root, rel, e)
            raw_output = ""

        item = {
            "image": rel,
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": raw_output},
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
