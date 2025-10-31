#!/usr/bin/env python3
"""Utility for generating pre-annotations for image datasets.

Changes vs previous version:
- OCR now uses Florence-2 (<OCR>) and returns a list of normalized words.
- Main object scoring improved: objectness * area_norm * center_prior.
- If no detections/captions, fallback: Florence-2 <CAPTION> with "one word" constraint.

The script performs three independent passes over a dataset of images:
1) OCR with Florence-2 <OCR>.
2) Region proposal + dense captions with Florence-2 to pick a main object.
3) Style classification with CLIP using prompt ensembling.

Outputs:
- texts.json: { rel_path: [word, ...] }
- main_objects.json: { rel_path: "<string>" }
- styles.json: { rel_path: "<style>" }
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from contextlib import nullcontext

import torch
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DEFAULT_CLIP_TEMPLATES = (
    "a {style} image",
    "image in the style of {style}",
)


@dataclass
class Config:
    image_root: Path
    output_dir: Path
    device: str
    batch_size_clip: int
    save_every: int
    florence_model_id: str
    clip_model_id: str
    clip_styles: Sequence[str]
    clip_threshold: float
    florence_max_new_tokens: int
    use_bf16: bool
    use_fp16: bool
    timeout_per_image_seconds: int


def load_config(config_path: Path, override_device: Optional[str]) -> Config:
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    image_root = Path(raw_cfg["image_root"]).expanduser().resolve()
    output_dir = Path(raw_cfg["output_dir"]).expanduser().resolve()
    device = override_device or raw_cfg.get("device") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    clip_styles = raw_cfg.get("clip_styles") or []
    if not clip_styles:
        raise ValueError("Config must provide at least one CLIP style candidate.")

    cfg = Config(
        image_root=image_root,
        output_dir=output_dir,
        device=device,
        batch_size_clip=int(raw_cfg.get("batch_size_clip", 4)),
        save_every=int(raw_cfg.get("save_every", 20)),
        florence_model_id=raw_cfg.get("florence_model_id", "microsoft/Florence-2-base"),
        clip_model_id=raw_cfg.get("clip_model_id", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
        clip_styles=clip_styles,
        clip_threshold=float(raw_cfg.get("clip_threshold", 0.25)),
        florence_max_new_tokens=int(raw_cfg.get("florence_max_new_tokens", 512)),
        use_bf16=bool(raw_cfg.get("use_bf16", False)),
        use_fp16=bool(raw_cfg.get("use_fp16", False)),
        timeout_per_image_seconds=int(raw_cfg.get("timeout_per_image_seconds", 30)),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("generate_annotation")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    error_log_path = output_dir / "errors.log"
    file_handler = logging.FileHandler(error_log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)

    return logger


def log_exception(logger: logging.Logger, rel_path: str, exc: BaseException) -> None:
    logger.error("Failed to process %s: %s", rel_path, exc)
    logger.debug("Traceback for %s", rel_path, exc_info=exc)
    error_log_handler = next(
        (handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)),
        None,
    )
    if error_log_handler is not None:
        error_log_path = Path(error_log_handler.baseFilename)
        with error_log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {rel_path}\n")
            f.write("".join(traceback.format_exception(exc)))
            f.write("\n")


def scan_images(image_root: Path) -> List[str]:
    image_paths: List[str] = []
    for root, _, files in os.walk(image_root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                abs_path = Path(root) / name
                rel_path = abs_path.relative_to(image_root).as_posix()
                image_paths.append(rel_path)
    image_paths.sort()
    return image_paths


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def run_with_timeout(func, timeout: int, *args, **kwargs):
    if timeout <= 0:
        return func(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)


def open_image_rgb(path: Path) -> Image.Image:
    try:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RuntimeError(f"Cannot open image: {exc}") from exc


def normalize_text(text: str) -> List[str]:
    lowered = text.lower()
    cleaned_chars = [char if (char.isalnum() or char.isspace()) else " " for char in lowered]
    no_punct = "".join(cleaned_chars)
    compressed = re.sub(r"\s+", " ", no_punct).strip()
    if not compressed:
        return []
    return re.findall(r"\w+", compressed)


def torch_autocast(device: str, use_fp16: bool, use_bf16: bool):
    if device.startswith("cuda") and torch.cuda.is_available():
        if use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if use_fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cuda", enabled=False)
    return nullcontext()


def cleanup_torch(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def strip_loc_tokens(text: str) -> str:
    """Удалить все <loc_###> из строки."""
    return re.sub(r"<loc_\d+>", "", text).strip()


def first_word(text: str) -> str:
    m = re.search(r"[A-Za-z0-9]+", text)
    return m.group(0).lower() if m else (text.strip().split()[0] if text.strip() else "")


def parse_florence_dense_pairs(raw: str) -> List[RegionCaption]:
    """
    Парсит последовательность вида:
      caption <loc_i><loc_j><loc_k><loc_l> caption2 <loc_...> ...
    Возвращает список RegionCaption(caption, bbox_norm[0..1]).
    """
    res: List[RegionCaption] = []
    i = 0
    n = len(raw)
    while i < n:
        # 1) набираем текст до первого <loc_...>
        m = re.search(r"<loc_(\d+)>", raw[i:])
        if not m:
            break
        start_loc = i + m.start()
        caption = raw[i:start_loc].strip()
        # 2) читаем 4 loc подряд
        locs = re.findall(r"<loc_(\d+)>", raw[start_loc:start_loc+200])  # локально
        if len(locs) < 4:
            break
        x1, y1, x2, y2 = [int(t) / 999.0 for t in locs[:4]]
        # 3) сдвигаем указатель за эти 4 loc-токена
        advance = 0
        cnt = 0
        for m2 in re.finditer(r"<loc_(\d+)>", raw[start_loc:]):
            advance = m2.end()
            cnt += 1
            if cnt == 4:
                break
        i = start_loc + advance
        # 4) приводим капшен в чистый вид и добавляем
        clean_caption = caption.strip(" .,:;|-").strip()
        if clean_caption:
            res.append(RegionCaption(bbox=(x1, y1, x2, y2), caption=clean_caption))
    return res


# -----------------------------
# Florence-2 helpers
# -----------------------------
class Florence:
    def __init__(self, model_id: str, device: str, max_new_tokens: int, use_fp16: bool, use_bf16: bool):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # robust eager attn for compatibility
        setattr(config, "attn_implementation", "eager")
        setattr(config, "_attn_implementation", "eager")
        setattr(config, "use_cache", "False")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=config,
            attn_implementation="eager",
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16

    def generate(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch_autocast(self.device, self.use_fp16, self.use_bf16):
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=False,
                    return_dict_in_generate=False,
                )
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        text = text.replace(prompt, "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def close(self):
        del self.model
        del self.processor


# -----------------------------
# OCR with Florence-2 (<OCR>)
# -----------------------------
def perform_ocr_florence(
    cfg: Config,
    logger: logging.Logger,
    rel_paths: Sequence[str],
    texts_path: Path,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    fl = Florence(cfg.florence_model_id, cfg.device, cfg.florence_max_new_tokens, cfg.use_fp16, cfg.use_bf16)

    results = dict(existing)
    since_save = 0
    timeout = cfg.timeout_per_image_seconds

    for rel_path in tqdm(rel_paths, desc="OCR (Florence-2)", unit="img"):
        if rel_path in results:
            continue
        abs_path = cfg.image_root / rel_path

        def _process_image() -> List[str]:
            image = open_image_rgb(abs_path)
            raw = fl.generate(image, "<OCR>")
            # Florence часто возвращает текст без спец-токенов — просто нормализуем
            return normalize_text(raw)

        try:
            words = run_with_timeout(_process_image, timeout)
        except Exception as exc:  # noqa: BLE001
            log_exception(logger, rel_path, exc)
            continue

        results[rel_path] = words
        since_save += 1
        if since_save >= cfg.save_every:
            save_json(texts_path, results)
            since_save = 0

    save_json(texts_path, results)
    fl.close()
    cleanup_torch(cfg.device)
    return results


# -----------------------------
# Florence-2 main object
# -----------------------------
@dataclass
class RegionCandidate:
    bbox: Tuple[float, float, float, float]  # absolute or normalized
    score: float  # objectness if present, else 1.0


@dataclass
class RegionCaption:
    bbox: Tuple[float, float, float, float]
    caption: str


def _extract_json_candidate(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    start_brace = text.find("{")
    start_bracket = text.find("[")
    candidates = []
    if start_brace != -1:
        end_brace = text.rfind("}")
        if end_brace != -1 and end_brace > start_brace:
            candidates.append(text[start_brace : end_brace + 1])
    if start_bracket != -1:
        end_bracket = text.rfind("]")
        if end_bracket != -1 and end_bracket > start_bracket:
            candidates.append(text[start_bracket : end_bracket + 1])
    for candidate in candidates:
        return candidate
    return None


def _load_json_lenient(text: str) -> Any:
    candidate = _extract_json_candidate(text)
    if candidate is None:
        raise ValueError("No JSON-like structure detected")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = candidate.replace("'", '"')
        cleaned = re.sub(r"(\w+)\s*:", r'"\1":', cleaned)
        return json.loads(cleaned)


def parse_region_captions(raw: str) -> List[RegionCaption]:
    try:
        parsed = _load_json_lenient(raw)
    except Exception:
        return []

    captions: List[RegionCaption] = []

    if isinstance(parsed, dict):
        entries = parsed.get("regions") or parsed.get("items") or parsed.get("captions")
        if isinstance(entries, list):
            for item in entries:
                bbox = _coerce_bbox(item.get("bbox") if isinstance(item, dict) else None)
                caption = (
                    item.get("caption")
                    if isinstance(item, dict)
                    else str(item)
                    if item is not None
                    else ""
                )
                if bbox is not None and caption:
                    captions.append(RegionCaption(bbox=bbox, caption=str(caption)))
        else:
            for key, value in parsed.items():
                bbox = _coerce_bbox(value) if key.lower().startswith("bbox") else None
                if bbox is not None:
                    captions.append(RegionCaption(bbox=bbox, caption=str(key)))
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                bbox = _coerce_bbox(item.get("bbox") or item.get("box"))
                caption = item.get("caption") or item.get("text") or ""
                if bbox is not None and caption:
                    captions.append(RegionCaption(bbox=bbox, caption=str(caption)))
            elif isinstance(item, (list, tuple)):
                bbox = _coerce_bbox(item)
                if bbox is not None:
                    captions.append(RegionCaption(bbox=bbox, caption=""))
            else:
                captions.append(RegionCaption(bbox=(0.0, 0.0, 0.0, 0.0), caption=str(item)))
    return captions


def _coerce_bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        keys = ["x1", "y1", "x2", "y2"]
        if all(k in value for k in keys):
            return tuple(float(value[k]) for k in keys)  # type: ignore[return-value]
        keys_xywh = ["x", "y", "w", "h"]
        if all(k in value for k in keys_xywh):
            x, y, w, h = (float(value[k]) for k in keys_xywh)
            return x, y, x + w, y + h
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            x1, y1, x2, y2 = map(float, value[:4])
            if x2 < x1 or y2 < y1:
                return None
            return x1, y1, x2, y2
        except (TypeError, ValueError):
            return None
    return None


def _coerce_score(scores: Any, idx: int) -> Optional[float]:
    if scores is None:
        return 1.0
    if isinstance(scores, (list, tuple)):
        if idx < len(scores):
            try:
                return float(scores[idx])
            except (TypeError, ValueError):
                return None
    try:
        return float(scores)
    except (TypeError, ValueError):
        return None


def to_absolute_bbox(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    # If coords look normalized (<=1.5 by magnitude), scale to absolute px.
    if max(abs(coord) for coord in bbox) <= 1.5:
        x1, y1, x2, y2 = bbox
        return x1 * width, y1 * height, x2 * width, y2 * height
    return bbox


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_prior(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> float:
    cx, cy = bbox_center(bbox)
    img_cx, img_cy = width / 2.0, height / 2.0
    dist = math.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
    max_dist = math.sqrt((img_cx) ** 2 + (img_cy) ** 2)  # corner distance
    if max_dist == 0:
        return 1.0
    return max(0.0, 1.0 - min(dist / max_dist, 1.0))


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_caption(
    candidate_bbox: Tuple[float, float, float, float],
    captions: Sequence[RegionCaption],
    width: int,
    height: int,
) -> str:
    if not captions:
        return ""
    cand_bbox = candidate_bbox
    best_caption = captions[0].caption if captions[0].caption else ""
    best_iou = -1.0
    for entry in captions:
        abs_bbox = to_absolute_bbox(entry.bbox, width, height)
        iou = bbox_iou(cand_bbox, abs_bbox)
        if iou > best_iou and entry.caption:
            best_caption = entry.caption
            best_iou = iou
    return best_caption


def parse_florence_locs(text: str) -> List[Tuple[float, float, float, float]]:
    """
    Parse <loc_*> tokens produced by Florence-2 region proposal output.

    Example input:
        "</s><s><loc_0><loc_0><loc_998><loc_998></s>"

    Returns list of (x1, y1, x2, y2) in [0,1] normalized coordinates.
    """
    tokens = re.findall(r"<loc_(\d+)>", text)
    if len(tokens) < 4:
        return []

    coords = [int(t) / 999.0 for t in tokens]
    boxes = []
    for i in range(0, len(coords) - 3, 4):
        x1, y1, x2, y2 = coords[i:i+4]
        if 0 <= x1 <= 1 and 0 <= x2 <= 1 and 0 <= y1 <= 1 and 0 <= y2 <= 1:
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


_EN_STOP = {
    "a","an","the","of","in","on","at","with","and","or","for","to","from",
    "this","that","these","those","is","are","was","were","be","being","been",
    "by","as","it","its","into","over","under","near","next","up","down",
}

def select_one_word(text: str) -> str:
    """Берёт обычный caption и возвращает одно осмысленное слово.
    Стратегия: первый токен [a-z0-9], не стоп-слово, длиной >=3."""
    text = strip_loc_tokens(text).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for t in tokens:
        if len(t) >= 3 and t not in _EN_STOP:
            return t
    # запасной вариант: первый алфанум-токен, если всё остальное отпало
    return tokens[0] if tokens else ""


def perform_florence_main_object(
    cfg: Config,
    logger: logging.Logger,
    rel_paths: Sequence[str],
    main_objects_path: Path,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    fl = Florence(cfg.florence_model_id, cfg.device, cfg.florence_max_new_tokens, cfg.use_fp16, cfg.use_bf16)

    results = dict(existing)
    since_save = 0
    timeout = cfg.timeout_per_image_seconds

    for rel_path in tqdm(rel_paths, desc="Main object (Florence-2)", unit="img"):
        if rel_path in results:
            continue
        abs_path = cfg.image_root / rel_path

        def _process_image() -> str:
            image = open_image_rgb(abs_path)
            width, height = image.size

            # 1) Proposals
            proposal_text = fl.generate(image, "<REGION_PROPOSAL>")
            loc_boxes = parse_florence_locs(proposal_text)
            candidates = [RegionCandidate(bbox=b, score=1.0) for b in loc_boxes]

            # 2) Dense captions (for label assignment)
            dense_text = fl.generate(image, "<DENSE_REGION_CAPTION>")
            # 1) пробуем структурно распарсить caption+4loc
            captions = parse_florence_dense_pairs(dense_text)
            # 2) если не получилось, берём глобальный капшен без loc-токенов
            if not captions and dense_text.strip():
                global_caption = strip_loc_tokens(dense_text)
                if global_caption:
                    captions = [RegionCaption(
                        bbox=(0.0, 0.0, float(width), float(height)),
                        caption=global_caption,
                    )]

            # 3) Score candidates with area_norm * center_prior * objectness
            best_caption = ""
            if candidates:
                img_area = float(width * height) if width > 0 and height > 0 else 1.0
                best_score = -1.0
                for cand in candidates:
                    abs_bbox = to_absolute_bbox(cand.bbox, width, height)
                    area = bbox_area(abs_bbox)
                    if area <= 0:
                        continue
                    area_norm = max(0.0, min(area / img_area, 1.0))
                    prior = center_prior(abs_bbox, width, height)
                    final_score = (cand.score or 1.0) * area_norm * prior
                    if final_score > best_score:
                        best_score = final_score
                        best_caption = match_caption(abs_bbox, captions, width, height)

            # 4) If still empty -> ask for one-word caption
            if not best_caption:
                caption_full = fl.generate(image, "<CAPTION>")
                best_caption = select_one_word(caption_full)

            return best_caption

        try:
            caption = run_with_timeout(_process_image, timeout)
        except Exception as exc:  # noqa: BLE001
            log_exception(logger, rel_path, exc)
            caption = ""

        results[rel_path] = strip_loc_tokens(caption or "")
        since_save += 1
        if since_save >= cfg.save_every:
            save_json(main_objects_path, results)
            since_save = 0

    save_json(main_objects_path, results)
    fl.close()
    cleanup_torch(cfg.device)
    return results


# -----------------------------
# CLIP (unchanged)
# -----------------------------
def preprocess_clip_text(
    processor: CLIPProcessor,
    styles: Sequence[str],
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    prompts: List[str] = []
    prompt_styles: List[str] = []
    for style in styles:
        for template in DEFAULT_CLIP_TEMPLATES:
            prompts.append(template.format(style=style))
            prompt_styles.append(style)
    inputs = processor(text=prompts, padding=True, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, prompt_styles


def compute_style_embeddings(
    model: CLIPModel,
    text_inputs: Dict[str, torch.Tensor],
    prompt_styles: Sequence[str],
) -> Tuple[torch.Tensor, List[str]]:
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    style_to_features: Dict[str, List[torch.Tensor]] = {}
    for feature, style in zip(text_features, prompt_styles):
        style_to_features.setdefault(style, []).append(feature)

    averaged_features: List[torch.Tensor] = []
    ordered_styles: List[str] = []
    for style, features in style_to_features.items():
        stacked = torch.stack(features, dim=0)
        averaged = stacked.mean(dim=0)
        averaged = averaged / averaged.norm()
        averaged_features.append(averaged)
        ordered_styles.append(style)

    return torch.stack(averaged_features, dim=0), ordered_styles


def perform_clip(
    cfg: Config,
    logger: logging.Logger,
    rel_paths: Sequence[str],
    styles_path: Path,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    processor = CLIPProcessor.from_pretrained(cfg.clip_model_id)
    model = CLIPModel.from_pretrained(cfg.clip_model_id)
    model.to(cfg.device)
    model.eval()

    text_inputs, prompt_styles = preprocess_clip_text(
        processor, cfg.clip_styles, cfg.device
    )
    style_embeddings, ordered_styles = compute_style_embeddings(
        model, text_inputs, prompt_styles
    )
    del text_inputs

    results = dict(existing)
    since_save = 0
    timeout = cfg.timeout_per_image_seconds

    batch: List[Tuple[str, Path]] = []

    def flush_batch():
        nonlocal since_save
        if not batch:
            return
        rels, paths = zip(*batch)

        def _process_batch() -> List[Tuple[str, float]]:
            images = [open_image_rgb(path) for path in paths]
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(cfg.device) for k, v in image_inputs.items()}
            with torch.no_grad():
                with torch_autocast(cfg.device, cfg.use_fp16, cfg.use_bf16):
                    image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            se = style_embeddings.to(device=image_features.device, dtype=image_features.dtype)
            logits = image_features @ se.T
            probs = torch.softmax(logits, dim=-1)
            results_local: List[Tuple[str, float]] = []
            for prob in probs:
                max_prob, idx = prob.max(dim=0)
                results_local.append((ordered_styles[int(idx)], float(max_prob)))
            return results_local

        try:
            batch_results = run_with_timeout(_process_batch, timeout)
        except Exception as exc:  # noqa: BLE001
            for rel_path in rels:
                log_exception(logger, rel_path, exc)
            batch.clear()
            return

        for rel_path, (style, confidence) in zip(rels, batch_results):
            predicted = style if confidence >= cfg.clip_threshold else "unknown"
            results[rel_path] = predicted
            since_save += 1
            if since_save >= cfg.save_every:
                save_json(styles_path, results)
                since_save = 0
        batch.clear()

    for rel_path in tqdm(rel_paths, desc="CLIP", unit="img"):
        if rel_path in results:
            continue
        abs_path = cfg.image_root / rel_path
        batch.append((rel_path, abs_path))
        if len(batch) >= cfg.batch_size_clip:
            flush_batch()

    flush_batch()
    save_json(styles_path, results)
    del model
    del processor
    cleanup_torch(cfg.device)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset annotations")
    parser.add_argument("--config", default="qwen-vl-finetune/qwenvl/generate_annotation.yaml", type=Path, help="Path to YAML config")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config, args.device)
    logger = setup_logging(cfg.output_dir)
    logger.info("Using device: %s", cfg.device)
    logger.info("Image root: %s", cfg.image_root)
    logger.info("Output dir: %s", cfg.output_dir)

    image_paths = scan_images(cfg.image_root)
    logger.info("Discovered %d images", len(image_paths))

    texts_path = cfg.output_dir / "texts.json"
    main_objects_path = cfg.output_dir / "main_objects.json"
    styles_path = cfg.output_dir / "styles.json"

    texts = load_json(texts_path)
    main_objects = load_json(main_objects_path)
    styles = load_json(styles_path)

    if image_paths:
        texts = perform_ocr_florence(cfg, logger, image_paths, texts_path, texts)
        main_objects = perform_florence_main_object(cfg, logger, image_paths, main_objects_path, main_objects)
        styles = perform_clip(cfg, logger, image_paths, styles_path, styles)
    else:
        logger.info("No images found; exiting.")

    logger.info(
        "Processing complete: %d OCR entries, %d main objects, %d styles",
        len(texts),
        len(main_objects),
        len(styles),
    )


if __name__ == "__main__":
    main()
