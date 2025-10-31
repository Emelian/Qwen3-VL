#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Complement QWEN annotations.json with Florence-2 OCR/caption and CLIP style.

Input:
  <root>/
    images/...
    annotations.json   # produced by the sampling script

This script updates each item in annotations.json:
  - Keeps existing "watermarks"
  - Fills:
      "text": list[str] from Florence-2 <OCR> (normalized, unique, truncated to 200 tokens)
      "main object": 1 word extracted from Florence-2 <CAPTION>
      "style": single label from STYLES via CLIP prompt ensembling
Errors are appended to <root>/errors.txt.
Periodically saves annotations.json every --save-every items (default: 10) without printing save messages.

Usage:
  python complement_annotations.py \
      --root "/path/to/sampled_output_dir" \
      --device "cuda" \
      --florence-model-id "microsoft/Florence-2-base" \
      --clip-model-id "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
      --clip-threshold 0.05 \
      --save-every 10
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
STYLES = [
    "photo", "screenshot", "portrait", "landscape", "clipart",
    "comic", "anime", "render", "concept", "illustration",
    "minimalist", "vintage", "poster", "meme", "infographic",
    "sketch", "watercolor", "digital", "fashion", "documentary",
]
CLIP_TEMPLATES = ("a {style} image", "image in the style of {style}")
FLORENCE_MAX_NEW_TOKENS = 128


def errors_path(root: Path) -> Path:
    """Path to error log file."""
    return root / "errors.txt"


def log_error(root: Path, rel_image: str, exc: BaseException) -> None:
    """Append an error line to errors.txt."""
    with errors_path(root).open("a", encoding="utf-8") as f:
        f.write(f"[{rel_image}] {type(exc).__name__}: {exc}\n")


def open_image_rgb(path: Path) -> Image.Image:
    """Open an image and convert to RGB."""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RuntimeError(f"Cannot open image: {exc}") from exc


def normalize_words(text: str) -> List[str]:
    """Lowercase, keep alnum/dash, deduplicate preserving order."""
    lowered = text.lower()
    cleaned = [ch if (ch.isalnum() or ch.isspace() or ch == "-") else " " for ch in lowered]
    no_punct = "".join(cleaned)
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", no_punct)
    seen, out = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def strip_loc_tokens(text: str) -> str:
    """Remove Florence <loc_###> tokens."""
    return re.sub(r"<loc_\d+>", "", text).strip()


def select_one_word(caption: str) -> str:
    """Pick a simple noun-like token from a caption."""
    text = strip_loc_tokens(caption).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    stop = {
        "a","an","the","of","in","on","at","with","and","or","for","to","from",
        "this","that","these","those","is","are","was","were","be","being","been",
        "by","as","it","its","into","over","under","near","next","up","down",
    }
    for t in tokens:
        if len(t) >= 3 and t not in stop:
            return t
    return tokens[0] if tokens else ""


@dataclass
class FlorenceParams:
    """Florence-2 loading parameters."""
    model_id: str
    device: str


class Florence:
    """Minimal Florence-2 wrapper for OCR and caption prompts."""

    def __init__(self, p: FlorenceParams):
        self.processor = AutoProcessor.from_pretrained(p.model_id, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(p.model_id, trust_remote_code=True)
        setattr(cfg, "attn_implementation", "eager")
        setattr(cfg, "_attn_implementation", "eager")
        setattr(cfg, "use_cache", False)
        self.model = AutoModelForCausalLM.from_pretrained(
            p.model_id, trust_remote_code=True, config=cfg, attn_implementation="eager"
        ).to(p.device)
        self.model.eval()
        self.device = p.device
        self.autocast_dtype = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.no_cast_ctx = nullcontext()

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Generate Florence-2 response for a single image and prompt token."""
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.autocast_dtype:
                ctx = torch.autocast("cuda", dtype=self.autocast_dtype)
            else:
                ctx = self.no_cast_ctx
            with ctx:
                out_ids = self.model.generate(
                    **inputs, max_new_tokens=FLORENCE_MAX_NEW_TOKENS, use_cache=False, return_dict_in_generate=False
                )
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        text = text.replace(prompt, "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def close(self) -> None:
        """Free Florence-2 resources."""
        del self.model
        del self.processor


def build_style_bank(proc: CLIPProcessor, model: CLIPModel, device: str) -> Tuple[torch.Tensor, List[str]]:
    """Build averaged CLIP text embeddings bank per style."""
    prompts, owners = [], []
    for s in STYLES:
        for t in CLIP_TEMPLATES:
            prompts.append(t.format(style=s))
            owners.append(s)
    txt_inputs = proc(text=prompts, padding=True, truncation=True, return_tensors="pt")
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
    with torch.no_grad():
        txt = model.get_text_features(**txt_inputs)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    per_style: Dict[str, List[torch.Tensor]] = {}
    for feat, owner in zip(txt, owners):
        per_style.setdefault(owner, []).append(feat)
    avg_feats, ordered = [], []
    for s, feats in per_style.items():
        m = torch.stack(feats, 0).mean(0)
        m = m / m.norm()
        avg_feats.append(m)
        ordered.append(s)
    bank = torch.stack(avg_feats, 0)
    return bank, ordered


@dataclass
class PredictArgs:
    """Prediction settings."""
    device: str
    florence_model_id: str
    clip_model_id: str
    clip_threshold: float


class Predictor:
    """Predict 'text', 'main object', and 'style' for a single image."""

    def __init__(self, pargs: PredictArgs):
        self.pargs = pargs
        self.fl = Florence(FlorenceParams(model_id=pargs.florence_model_id, device=pargs.device))
        self.clip_proc = CLIPProcessor.from_pretrained(pargs.clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(pargs.clip_model_id).to(pargs.device)
        self.clip_model.eval()
        self.style_bank, self.ordered_styles = build_style_bank(self.clip_proc, self.clip_model, pargs.device)

    def close(self) -> None:
        """Free heavy models."""
        self.fl.close()
        del self.clip_model
        del self.clip_proc
        if self.pargs.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_fields(self, image_path: Path) -> Dict[str, Any]:
        """Compute OCR words, main object (1 token), and style label."""
        img = open_image_rgb(image_path)

        ocr_raw = self.fl.generate(img, "<OCR>")
        words = normalize_words(ocr_raw)[:200]

        cap = self.fl.generate(img, "<CAPTION>")
        main_obj = select_one_word(cap)

        img_inputs = self.clip_proc(images=img, return_tensors="pt")
        img_inputs = {k: v.to(self.pargs.device) for k, v in img_inputs.items()}
        with torch.no_grad():
            img_feat = self.clip_model.get_image_features(**img_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        bank = self.style_bank.to(device=img_feat.device, dtype=img_feat.dtype)
        logits = img_feat @ bank.T
        probs = torch.softmax(logits, dim=-1)[0]
        conf, idx = torch.max(probs, dim=0)
        style = self.ordered_styles[int(idx)] if float(conf) >= self.pargs.clip_threshold else "unknown"

        return {"text": words, "main object": main_obj, "style": style}


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    """Load annotations.json (list of items)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("annotations.json must be a list of items")
    return data


def parse_gpt_json(value: str) -> Dict[str, Any]:
    """Parse the GPT JSON payload; recover from minor formatting issues."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        candidate = value.strip()
        candidate = candidate.replace("'", '"')
        candidate = re.sub(r"(\w+)\s*:", r'"\1":', candidate)
        return json.loads(candidate)


def save_annotations(path: Path, anns: List[Dict[str, Any]]) -> None:
    """Atomically save annotations.json."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(anns, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Complement QWEN annotations.json with text/main object/style")
    p.add_argument("--root", type=Path, required=True, help="Dataset root containing images/ and annotations.json")
    p.add_argument("--device", type=str, default="cuda", help="Device string, e.g. 'cuda', 'cuda:0', or 'cpu'")
    p.add_argument("--florence-model-id", type=str, default="microsoft/Florence-2-base")
    p.add_argument("--clip-model-id", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    p.add_argument("--clip-threshold", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=10, help="Save annotations every N predictions (0 disables)")
    return p.parse_args()


def main() -> None:
    """CLI entry point with tqdm progress and periodic saving."""
    args = parse_args()
    root = args.root.resolve()
    ann_path = root / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json not found at: {ann_path}")

    anns = load_annotations(ann_path)
    predictor = Predictor(PredictArgs(
        device=args.device,
        florence_model_id=args.florence_model_id,
        clip_model_id=args.clip_model_id,
        clip_threshold=args.clip_threshold,
    ))

    ok, fail = 0, 0
    try:
        for i, item in enumerate(tqdm(anns, desc="Complementing", unit="img")):
            try:
                rel_img = item.get("image")
                if not isinstance(rel_img, str):
                    raise ValueError("Item has no 'image' string field")
                img_path = (root / rel_img).resolve()
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                convs = item.get("conversations") or []
                if len(convs) < 2 or convs[1].get("from") != "gpt":
                    raise ValueError("Item lacks gpt response at conversations[1]")
                gpt_val = convs[1].get("value")
                if not isinstance(gpt_val, str):
                    raise ValueError("conversations[1].value must be a JSON string")

                gpt_json = parse_gpt_json(gpt_val)
                watermarks = gpt_json.get("watermarks", None)

                pred = predictor.predict_fields(img_path)
                new_payload = {
                    "watermarks": watermarks,
                    "text": pred["text"],
                    "main object": pred["main object"],
                    "style": pred["style"],
                }
                convs[1]["value"] = json.dumps(new_payload, ensure_ascii=False)
                ok += 1
            except Exception as e:  # noqa: BLE001
                rel_img = item.get("image", "<unknown>")
                log_error(root, str(rel_img), e)
                fail += 1

            if args.save_every > 0 and (i + 1) % args.save_every == 0:
                save_annotations(ann_path, anns)

        save_annotations(ann_path, anns)
        print(f"Updated: {ok}, failed: {fail}")
        if fail > 0:
            print(f"See errors: {errors_path(root)}")
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
