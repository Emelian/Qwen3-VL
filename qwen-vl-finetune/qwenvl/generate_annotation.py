#!/usr/bin/env python3
"""Utility for generating pre-annotations for image datasets.

The script performs three independent passes over a dataset of images:

1. OCR using a TrOCR model to extract normalized token lists.
2. Region proposal and dense captioning with Florence-2 to identify the
   main object per image.
3. Style classification with CLIP using prompt ensembling.

Results are stored as JSON dictionaries in the configured output
directory. Each stage skips images that already have annotations and
performs incremental saves to avoid large progress loss on failures.

All heavyweight models are loaded once per stage, used, and explicitly
released to free GPU memory.
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
    AutoModelForCausalLM,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DEFAULT_CLIP_TEMPLATES = (
    "a {style} style image",
    "an image in the style of {style}",
    "a {style} artwork",
    "a piece of art in the {style} style",
    "a photograph with {style} aesthetics",
)


@dataclass
class Config:
    image_root: Path
    output_dir: Path
    device: str
    batch_size_clip: int
    save_every: int
    ocr_model_id: str
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
        ocr_model_id=raw_cfg.get("ocr_model_id", "microsoft/trocr-large-printed"),
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


def perform_ocr(
    cfg: Config,
    logger: logging.Logger,
    rel_paths: Sequence[str],
    texts_path: Path,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    processor = TrOCRProcessor.from_pretrained(cfg.ocr_model_id)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.ocr_model_id)
    model.to(cfg.device)
    model.eval()

    results = dict(existing)
    since_save = 0
    timeout = cfg.timeout_per_image_seconds

    for rel_path in tqdm(rel_paths, desc="OCR", unit="img"):
        if rel_path in results:
            continue
        abs_path = cfg.image_root / rel_path

        def _process_image() -> List[str]:
            image = open_image_rgb(abs_path)
            pixel_values = processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(cfg.device)
            with torch.no_grad():
                with torch_autocast(cfg.device, cfg.use_fp16, cfg.use_bf16):
                    generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return normalize_text(text)

        try:
            words = run_with_timeout(_process_image, timeout)
        except Exception as exc:  # pylint: disable=broad-except
            log_exception(logger, rel_path, exc)
            continue

        results[rel_path] = words
        since_save += 1
        if since_save >= cfg.save_every:
            save_json(texts_path, results)
            since_save = 0

    save_json(texts_path, results)
    del model
    del processor
    cleanup_torch(cfg.device)
    return results


@dataclass
class RegionCandidate:
    bbox: Tuple[float, float, float, float]
    score: float


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
        # fall back to literal eval after normalising quotes
        cleaned = candidate.replace("'", '"')
        cleaned = re.sub(r"(\w+)\s*:", r'"\1":', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse JSON") from exc


def parse_region_candidates(raw: str) -> List[RegionCandidate]:
    try:
        parsed = _load_json_lenient(raw)
    except Exception:
        return []

    candidates: List[RegionCandidate] = []

    if isinstance(parsed, dict):
        items = parsed.get("regions") or parsed.get("bboxes") or parsed.get("boxes")
        scores = parsed.get("scores") or parsed.get("objectness")
        if isinstance(items, list):
            for idx, item in enumerate(items):
                bbox = _coerce_bbox(item)
                score = _coerce_score(scores, idx)
                if bbox is not None and score is not None:
                    candidates.append(RegionCandidate(bbox=bbox, score=score))
        else:
            for value in parsed.values():
                if isinstance(value, list):
                    for item in value:
                        bbox = _coerce_bbox(item)
                        if bbox is not None:
                            candidates.append(RegionCandidate(bbox=bbox, score=1.0))
    elif isinstance(parsed, list):
        for item in parsed:
            bbox = None
            score = 1.0
            if isinstance(item, dict):
                bbox = _coerce_bbox(item.get("bbox") or item.get("box") or item.get("b"))
                score_val = item.get("score") or item.get("objectness")
                if score_val is not None:
                    score = float(score_val)
            else:
                bbox = _coerce_bbox(item)
            if bbox is not None:
                candidates.append(RegionCandidate(bbox=bbox, score=score))
    if not candidates:
        number_pattern = re.compile(
            r"\[\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
            r"\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
            r"\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
            r"\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
            r"(?:\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?\s*\]"
        )
        for match in number_pattern.finditer(raw):
            groups = match.groups()
            if groups[0:4] and all(g is not None for g in groups[:4]):
                x1, y1, x2, y2 = map(float, groups[:4])
                score_group = groups[4]
                score = float(score_group) if score_group is not None else 1.0
                if x2 >= x1 and y2 >= y1:
                    candidates.append(RegionCandidate(bbox=(x1, y1, x2, y2), score=score))
    return candidates


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
    max_val = max(width, height)
    if max_val <= 0:
        return bbox
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
    max_dist = math.sqrt((img_cx) ** 2 + (img_cy) ** 2)
    if max_dist == 0:
        return 1.0
    return max(0.0, 1.0 - min(dist / max_dist, 1.0))


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


def perform_florence(
    cfg: Config,
    logger: logging.Logger,
    rel_paths: Sequence[str],
    main_objects_path: Path,
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    processor = AutoProcessor.from_pretrained(cfg.florence_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.florence_model_id,
        trust_remote_code=True,
    )
    model.to(cfg.device)
    model.eval()

    results = dict(existing)
    since_save = 0
    timeout = cfg.timeout_per_image_seconds

    for rel_path in tqdm(rel_paths, desc="Florence", unit="img"):
        if rel_path in results:
            continue
        abs_path = cfg.image_root / rel_path

        def _process_image() -> str:
            image = open_image_rgb(abs_path)
            width, height = image.size
            with torch.no_grad():
                inputs = processor(
                    text="<REGION_PROPOSAL>", images=image, return_tensors="pt"
                ).to(cfg.device)
                with torch_autocast(cfg.device, cfg.use_fp16, cfg.use_bf16):
                    proposal_ids = model.generate(
                        **inputs,
                        max_new_tokens=cfg.florence_max_new_tokens,
                    )
                proposal_text = processor.batch_decode(
                    proposal_ids, skip_special_tokens=False
                )[0]
                proposal_text = proposal_text.replace("<REGION_PROPOSAL>", "").strip()

                dense_inputs = processor(
                    text="<DENSE_REGION_CAPTION>", images=image, return_tensors="pt"
                ).to(cfg.device)
                with torch_autocast(cfg.device, cfg.use_fp16, cfg.use_bf16):
                    dense_ids = model.generate(
                        **dense_inputs,
                        max_new_tokens=cfg.florence_max_new_tokens,
                    )
                dense_text = processor.batch_decode(dense_ids, skip_special_tokens=False)[0]
                dense_text = dense_text.replace("<DENSE_REGION_CAPTION>", "").strip()

            candidates = parse_region_candidates(proposal_text)
            if not candidates:
                return ""
            captions = parse_region_captions(dense_text)
            if not captions and dense_text.strip():
                captions = [
                    RegionCaption(
                        bbox=(0.0, 0.0, float(width), float(height)),
                        caption=dense_text.strip(),
                    )
                ]

            best_caption = ""
            best_score = -1.0
            for candidate in candidates:
                abs_bbox = to_absolute_bbox(candidate.bbox, width, height)
                area = bbox_area(abs_bbox)
                if area <= 0:
                    continue
                prior = center_prior(abs_bbox, width, height)
                score = candidate.score * area * prior
                if score > best_score:
                    best_score = score
                    best_caption = match_caption(abs_bbox, captions, width, height)
            return best_caption

        try:
            caption = run_with_timeout(_process_image, timeout)
        except Exception as exc:  # pylint: disable=broad-except
            log_exception(logger, rel_path, exc)
            caption = ""

        results[rel_path] = caption or ""
        since_save += 1
        if since_save >= cfg.save_every:
            save_json(main_objects_path, results)
            since_save = 0

    save_json(main_objects_path, results)
    del model
    del processor
    cleanup_torch(cfg.device)
    return results


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
            logits = image_features @ style_embeddings.T
            probs = torch.softmax(logits, dim=-1)
            results_local: List[Tuple[str, float]] = []
            for prob in probs:
                max_prob, idx = prob.max(dim=0)
                results_local.append((ordered_styles[int(idx)], float(max_prob)))
            return results_local

        try:
            batch_results = run_with_timeout(_process_batch, timeout)
        except Exception as exc:  # pylint: disable=broad-except
            for rel_path in rels:
                log_exception(logger, rel_path, exc)
            batch.clear()
            return

        for rel_path, (style, confidence) in zip(rels, batch_results):
            if confidence < cfg.clip_threshold:
                predicted = "unknown"
            else:
                predicted = style
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
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override device", default=None)
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
        texts = perform_ocr(cfg, logger, image_paths, texts_path, texts)
        main_objects = perform_florence(cfg, logger, image_paths, main_objects_path, main_objects)
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
