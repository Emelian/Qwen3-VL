#!/usr/bin/env python3
"""Generate Qwen2.5-VL predictions for an image directory."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from contextlib import nullcontext

import torch
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

WATERMARK_PROMPT = (
    "You are a JSON generator. Count visible watermarks (logos or text overlays). "
    "Ignore natural scene text. Return JSON: {\"watermarks\": <integer>}"
)
MAIN_OBJECT_PROMPT = (
    "You are a JSON generator. Name the single main object (1–3 lowercase words). "
    "Return JSON: {\"main object\": <string>}"
)
STYLE_PROMPT_TEMPLATE = (
    "You are a JSON generator. Classify visual style. "
    "Choose strictly one from this list: {choices}. "
    "Return JSON: {\"style\": <string>}"
)


@dataclass
class Config:
    image_root: Path
    output_path: Path
    device: str
    save_every: int
    max_new_tokens: int
    temperature: float
    styles: List[str]
    skip_existing: bool


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_config(config_path: Path, override_device: Optional[str] = None) -> Config:
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    base_dir = config_path.parent.resolve()

    image_root = _resolve_path(base_dir, raw_cfg.get("image_root", "."))
    output_path = _resolve_path(base_dir, raw_cfg.get("output_path", "predicts.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if override_device:
        device = override_device
    else:
        cfg_device = raw_cfg.get("device")
        if cfg_device:
            device = str(cfg_device)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    save_every = int(raw_cfg.get("save_every", 20))
    if save_every <= 0:
        save_every = 1

    max_new_tokens = int(raw_cfg.get("max_new_tokens", 256))
    temperature = float(raw_cfg.get("temperature", 0.0))
    skip_existing = bool(raw_cfg.get("skip_existing", True))

    styles = raw_cfg.get("styles") or []
    if not isinstance(styles, list) or len(styles) == 0:
        raise ValueError("Config must provide a non-empty 'styles' list.")
    styles = [str(style) for style in styles]

    return Config(
        image_root=image_root,
        output_path=output_path,
        device=device,
        save_every=save_every,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        styles=styles,
        skip_existing=skip_existing,
    )


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("generate_predicts")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def scan_images(image_root: Path) -> List[str]:
    rel_paths: List[str] = []
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel_paths.append(path.relative_to(image_root).as_posix())
    return rel_paths


def load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_predictions(path: Path, data: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def open_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RuntimeError(f"Failed to open image {path}: {exc}") from exc


class Predictor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("generate_predicts")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        dtype = self._select_dtype(cfg.device)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        self.model.to(cfg.device)
        self.model.eval()

        tokenizer = self.processor.tokenizer
        pad_token = getattr(tokenizer, "pad_token_id", None)
        eos_token = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = pad_token if pad_token is not None else eos_token

        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")

        self.autocast_dtype = self._autocast_dtype(cfg.device)

    @staticmethod
    def _select_dtype(device: str) -> torch.dtype:
        if device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @staticmethod
    def _autocast_dtype(device: str) -> Optional[torch.dtype]:
        if device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return None

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    def _run_prompt(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        tensor_inputs = {
            key: value.to(self.cfg.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        input_length = tensor_inputs["input_ids"].shape[-1]
        do_sample = self.cfg.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.cfg.temperature

        with torch.no_grad():
            with self._autocast():
                generated = self.model.generate(**tensor_inputs, **generation_kwargs)

        new_tokens = generated[:, input_length:]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        return decoded.strip()

    def _parse_json(self, text: str, key: str) -> Any:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON for key '{key}': {exc}: {text}") from exc
        if key not in parsed:
            raise ValueError(f"JSON response missing '{key}': {text}")
        return parsed[key]

    def predict(self, image_path: Path) -> Dict[str, Any]:
        image = open_image(image_path)

        watermarks_raw = self._run_prompt(image, WATERMARK_PROMPT)
        main_object_raw = self._run_prompt(image, MAIN_OBJECT_PROMPT)
        choices_text = "{" + ", ".join(self.cfg.styles) + "}"
        style_prompt = STYLE_PROMPT_TEMPLATE.format(choices=choices_text)
        style_raw = self._run_prompt(image, style_prompt)

        watermarks = self._parse_json(watermarks_raw, "watermarks")
        if isinstance(watermarks, str) and watermarks.isdigit():
            watermarks = int(watermarks)
        if not isinstance(watermarks, int):
            try:
                watermarks = int(watermarks)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Watermarks value must be an integer: {watermarks_raw}"
                ) from exc

        main_object = self._parse_json(main_object_raw, "main object")
        if not isinstance(main_object, str):
            main_object = str(main_object)
        main_object = main_object.strip()

        style = self._parse_json(style_raw, "style")
        if not isinstance(style, str):
            style = str(style)
        style = style.strip()
        if style not in self.cfg.styles:
            raise ValueError(
                f"Predicted style '{style}' is not in configured styles list."
            )

        combined_text = json.dumps(
            {
                "watermarks": watermarks,
                "main object": main_object,
                "style": style,
            },
            ensure_ascii=False,
        )

        return {
            "watermarks": watermarks,
            "text": combined_text,
            "main object": main_object,
            "style": style,
        }


def is_complete(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if not isinstance(entry.get("watermarks"), int):
        return False
    if not isinstance(entry.get("main object"), str):
        return False
    if not isinstance(entry.get("style"), str):
        return False
    return True


def process_images(cfg: Config, logger: logging.Logger) -> None:
    if not cfg.image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {cfg.image_root}")

    predictor = Predictor(cfg)
    rel_paths = scan_images(cfg.image_root)
    logger.info("Found %d images in %s", len(rel_paths), cfg.image_root)

    predictions = load_existing(cfg.output_path)
    since_save = 0
    processed = 0

    for rel_path in tqdm(rel_paths, desc="Predicting", unit="img"):
        existing_entry = predictions.get(rel_path)
        if cfg.skip_existing and is_complete(existing_entry):
            continue

        abs_path = cfg.image_root / rel_path
        try:
            result = predictor.predict(abs_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to process %s: %s", rel_path, exc)
            continue

        predictions[rel_path] = result
        since_save += 1
        processed += 1

        if since_save >= cfg.save_every:
            save_predictions(cfg.output_path, predictions)
            logger.info("Progress saved after %d new items", since_save)
            since_save = 0

    if since_save > 0 or processed == 0:
        save_predictions(cfg.output_path, predictions)
        logger.info("Final predictions saved to %s", cfg.output_path)


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions with Qwen2.5-VL")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda, cuda:0, cpu)",
    )
    return parser.parse_args(args)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path, args.device)
    logger = setup_logging()
    logger.info("Using model %s on device %s", MODEL_ID, cfg.device)
    process_images(cfg, logger)


if __name__ == "__main__":
    main()
