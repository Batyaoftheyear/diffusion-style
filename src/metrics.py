"""Контуры с фиксированным допуском и CLS-признаки DINOv2."""
import math

import cv2
import numpy as np

from .config import data_dir, fingerprint, safe_path
from .data import validate_dataset
from .inference import complete_records, current_run
from .io import environment, file_sha, read_json, read_jsonl, write_csv, write_json
from .preprocessing import canny_map, load_rgb


def edge_metrics(source, generated, tolerance=2):
    source = np.asarray(source, dtype=bool)
    generated = np.asarray(generated, dtype=bool)
    if source.ndim != 2 or source.shape != generated.shape:
        raise ValueError("Expected two equally sized 2D edge maps")
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("Tolerance must be a nonnegative integer")
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    source_band = cv2.dilate(source.astype(np.uint8), kernel).astype(bool)
    generated_band = cv2.dilate(generated.astype(np.uint8), kernel).astype(bool)
    ns, ng = int(source.sum()), int(generated.sum())
    precision = float(np.logical_and(generated, source_band).sum() / ng) if ng else 0.0
    recall = float(np.logical_and(source, generated_band).sum() / ns) if ns else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"edge_precision": precision, "edge_recall": recall, "edge_f1": f1}


def cosine_similarity(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Expected equally sized finite vectors")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cosine is undefined for zero embeddings")
    return float(np.clip(np.dot(first, second) / denominator, -1, 1))


def summarize(rows):
    if not rows:
        raise ValueError("No measurements to summarize")
    keys = ("edge_precision", "edge_recall", "edge_f1", "dino_cosine", "generation_seconds", "peak_allocated_mib", "peak_reserved_mib")
    result = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite measurements: {key}")
        result[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "n": len(values)}
    return result


class DinoEncoder:
    def __init__(self, model):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if not torch.cuda.is_available():
            raise RuntimeError("Run DINOv2 evaluation on the Colab GPU")
        self.processor = AutoImageProcessor.from_pretrained(model["repo"], revision=model["revision"], use_fast=False)
        self.model = AutoModel.from_pretrained(model["repo"], revision=model["revision"], use_safetensors=True).eval().to("cuda")

    def embed(self, image):
        import torch
        inputs = {key: value.to("cuda") for key, value in self.processor(images=image, return_tensors="pt").items()}
        with torch.inference_mode():
            vector = self.model(**inputs).last_hidden_state[:, 0, :].float()
            vector = torch.nn.functional.normalize(vector, dim=-1)
        return vector[0].cpu().numpy()


def evaluate(c):
    validate_dataset(c)
    root, run = current_run(c)
    inputs = data_dir(c)
    photos = read_jsonl(inputs / "coco_manifest.jsonl")
    records = complete_records(root, photos, run["run_id"])
    if len(records) != 24 or not run.get("complete"):
        raise ValueError("Evaluation requires a complete 24-image baseline")
    env = environment(include_gpu=True)
    identity = {"run_id": run["run_id"], "model": c["models"]["dino"], "metrics": c["metrics"],
                "method": "DINOv2 last_hidden_state CLS, float32, L2 normalization, processor use_fast=False; edge many-to-many Chebyshev tolerance",
                "environment": env, "metrics_code_sha": file_sha(__file__)}
    evaluation_id = fingerprint(identity)[:20]
    metric_root = root / "evaluation" / evaluation_id
    if (metric_root / "protocol.json").exists():
        identity = read_json(metric_root / "protocol.json")
    photo_lookup = {row["id"]: row for row in photos}
    encoder, rows = None, []
    for record in records:
        destination = metric_root / f"{record['photo_id']}.json"
        cached = read_json(destination) if destination.exists() else None
        if cached and cached["output_sha256"] == record["output_sha256"] and cached["evaluation_id"] == evaluation_id:
            rows.append(cached)
            continue
        if encoder is None:
            encoder = DinoEncoder(c["models"]["dino"])
            identity["processor_config"] = encoder.processor.to_dict()
            write_json(metric_root / "protocol.json", identity)
        photo = photo_lookup[record["photo_id"]]
        source_image = load_rgb(safe_path(inputs, photo["prepared"]))
        generated_image = load_rgb(safe_path(root, record["output"]))
        generated_edges = np.asarray(canny_map(generated_image, c["canny_low_threshold"], c["canny_high_threshold"]).convert("L")) > 0
        source_edges = np.asarray(load_rgb(safe_path(inputs, photo["canny"])).convert("L")) > 0
        row = dict(photo_id=record["photo_id"], scene_group=photo["scene_group"], seed=record["seed"],
                   evaluation_id=evaluation_id, output_sha256=record["output_sha256"],
                   **edge_metrics(source_edges, generated_edges, c["metrics"]["edge_tolerance_px"]),
                   dino_cosine=cosine_similarity(encoder.embed(source_image), encoder.embed(generated_image)),
                   **{key: record[key] for key in ("generation_seconds", "peak_allocated_mib", "peak_reserved_mib")})
        write_json(destination, row)
        rows.append(row)
        print(f"Evaluated {row['photo_id']}", flush=True)
    summary = dict(run_id=run["run_id"], evaluation_id=evaluation_id, n=len(rows), metrics=summarize(rows),
                   std_ddof=1, scope="Between-image variation at one shared seed, not noise robustness", protocol=identity)
    write_csv(metric_root / "metrics.csv", rows)
    write_json(metric_root / "summary.json", summary)
    write_csv(root / "metrics.csv", rows)
    write_json(root / "summary.json", summary)
    return summary
