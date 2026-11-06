"""Конфигурация, проверка параметров и расположение артефактов."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_VERSION = 1


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path=None):
    path = Path(path) if path else REPO_ROOT / "configs/baseline.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    return config


def validate_config(c):
    if not isinstance(c, dict):
        raise ValueError("Configuration must be a mapping")
    for key in ("schema_version", "resolution", "num_inference_steps", "seed",
                "canny_low_threshold", "canny_high_threshold"):
        if type(c.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    if c["schema_version"] != 1 or c["resolution"] != 512:
        raise ValueError("Stage 1 uses schema 1 and resolution 512")
    if c["num_inference_steps"] < 1 or not 0 <= c["seed"] < 2**32:
        raise ValueError("Invalid step count or seed")
    if not 0 <= c["canny_low_threshold"] < c["canny_high_threshold"] <= 255:
        raise ValueError("Invalid Canny thresholds")
    for key in ("guidance_scale", "controlnet_conditioning_scale", "ip_adapter_scale", "eta"):
        value = c.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if c.get("scheduler") != "DDIM" or c["eta"] != 0:
        raise ValueError("Baseline requires deterministic DDIM with eta=0")
    if c.get("dtype") != "float16" or type(c.get("cpu_offload")) is not bool:
        raise ValueError("Baseline requires float16 and a boolean cpu_offload")
    if not isinstance(c.get("negative_prompt"), str):
        raise ValueError("negative_prompt must be a string")
    for key in ("sd", "controlnet", "ip_adapter", "dino"):
        model = c["models"][key]
        if not isinstance(model.get("repo"), str) or not re.fullmatch(r"[0-9a-f]{40}", str(model.get("revision", ""))):
            raise ValueError(f"Pin a full model revision: {key}")
    for section in ("art", "coco"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(c[section].get("revision", ""))):
            raise ValueError(f"Pin dataset revision: {section}")
    art = c["art"]
    for key in ("total", "references", "scan_rows", "dedup_hamming"):
        if type(art.get(key)) is not int:
            raise ValueError(f"art.{key} must be an integer")
    if not 60 <= art["total"] <= 80 or not 3 <= art["references"] <= 5:
        raise ValueError("Expected 60–80 artworks with 3–5 held-out references")
    if art["scan_rows"] < art["total"] or not 0 <= art["dedup_hamming"] <= 64:
        raise ValueError("Invalid art candidate count or dedup tolerance")
    if c["coco"]["count"] != 24:
        raise ValueError("Stage 1 test set contains exactly 24 photos")
    radius = c["metrics"]["edge_tolerance_px"]
    if type(radius) is not int or not 0 <= radius <= 8:
        raise ValueError("Invalid edge tolerance")


def work_root():
    return Path(os.environ.get("DIFFUSION_STYLE_HOME", str(REPO_ROOT))).expanduser().resolve()


def data_settings(c):
    packages = {}
    for name in ("numpy", "Pillow", "opencv-python-headless", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {key: c[key] for key in ("resolution", "seed", "canny_low_threshold", "canny_high_threshold", "art", "coco")} | {"data_version": DATA_VERSION, "preprocessing_packages": packages}


def data_dir(c):
    return work_root() / "data" / fingerprint(data_settings(c))[:16]


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes artifact root: {relative}")
    return candidate
