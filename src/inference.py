"""Фиксированный шум, проверяемый resume и сохранение реальных генераций."""
import io
import time
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, data_dir, fingerprint, safe_path, work_root
from .data import validate_dataset
from .io import atomic_write, environment, file_sha, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .preprocessing import load_rgb, save_png

LATENT_METHOD = "torch.randn(cpu_generator, seed, float32), saved as unscaled N(0,1) numpy; cast to fp16 at pipeline input"


def generation_code_sha():
    return fingerprint({name: file_sha(REPO_ROOT / "src" / name) for name in
                        ("config.py", "io.py", "preprocessing.py", "pipeline.py", "inference.py")})


def make_latents(seed, resolution=512):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, 4, resolution // 8, resolution // 8), generator=generator, dtype=torch.float32)


def save_latents(path, tensor):
    buffer = io.BytesIO()
    np.save(buffer, tensor.cpu().numpy(), allow_pickle=False)
    atomic_write(path, buffer.getvalue())


def data_identity(c):
    root = data_dir(c)
    return {name: file_sha(root / name) for name in ("manifest.json", "art_manifest.jsonl", "coco_manifest.jsonl")}


def current_run(c):
    pointer = read_json(work_root() / "results/current_run.json")
    root = safe_path(work_root() / "results", pointer["run_id"])
    run = read_json(root / "run.json")
    if run["config"] != c or run["data_identity"] != data_identity(c) or run["generation_code_sha"] != generation_code_sha():
        raise ValueError("Current run does not match config, data or generation code; run baseline with this config first")
    return root, run


def cached_record(root, photo_id, run_id):
    path = root / "metadata" / f"{photo_id}.json"
    if not path.exists():
        return None
    row = read_json(path)
    if row.get("photo_id") != photo_id or row.get("run_id") != run_id:
        raise ValueError("Result metadata identity mismatch")
    for key in ("output", "latents"):
        target = safe_path(root, row[key])
        if not target.is_file() or file_sha(target) != row[key + "_sha256"]:
            return None
    image = load_rgb(safe_path(root, row["output"]))
    if image.size != (512, 512):
        return None
    return row


def complete_records(root, photos, run_id):
    return [row for photo in photos if (row := cached_record(root, photo["id"], run_id)) is not None]


def export_records(root, records):
    if not records:
        return
    write_jsonl(root / "results.jsonl", records)
    keys = ("photo_id", "reference_id", "prompt", "negative_prompt", "seed", "output", "latents", "generation_seconds", "peak_allocated_mib", "peak_reserved_mib")
    write_csv(root / "results.csv", ({key: row[key] for key in keys} for row in records))


def run_baseline(c, limit=None):
    if limit is not None and not 1 <= limit <= 24:
        raise ValueError("limit must be between 1 and 24")
    validate_dataset(c)
    inputs = data_dir(c)
    photos = read_jsonl(inputs / "coco_manifest.jsonl")
    art = read_jsonl(inputs / "art_manifest.jsonl")
    info = read_json(inputs / "manifest.json")
    reference = next(row for row in art if row["id"] == info["main_reference_id"])
    env = environment(include_gpu=True)
    identity = dict(config=c, data_identity=data_identity(c), environment=env,
                    generation_code_sha=generation_code_sha(), latent_method=LATENT_METHOD)
    run_id = fingerprint(identity)[:20]
    root = work_root() / "results" / run_id
    root.mkdir(parents=True, exist_ok=True)
    run = dict(identity, run_id=run_id, expected_images=len(photos), main_reference_id=reference["id"])
    if (root / "run.json").exists():
        previous = read_json(root / "run.json")
        for key in identity:
            if previous[key] != identity[key]:
                raise ValueError("Existing run identity differs; refusing to overwrite")
        run = previous
    else:
        write_json(root / "run.json", run)
        write_json(root / "parameters.json", c)
    write_json(work_root() / "results/current_run.json", {"run_id": run_id})
    subset = photos[:limit] if limit else photos
    pending = [photo for photo in subset if cached_record(root, photo["id"], run_id) is None]
    if pending:
        import torch
        from .pipeline import load_pipeline
        start = time.perf_counter()
        pipe = load_pipeline(c)
        torch.cuda.synchronize()
        run["model_load_seconds"] = time.perf_counter() - start
        run["scheduler_config"] = dict(pipe.scheduler.config)
        write_json(root / "run.json", run)
        style_image = load_rgb(safe_path(inputs, reference["prepared"]))
        for photo in pending:
            latents = make_latents(c["seed"], c["resolution"])
            latent_path = root / "latents" / f"{photo['id']}.npy"
            save_latents(latent_path, latents)
            edge_image = load_rgb(safe_path(inputs, photo["canny"]))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                result = pipe(prompt=photo["prompt"], negative_prompt=c["negative_prompt"],
                              image=edge_image, ip_adapter_image=style_image,
                              latents=latents.to(device="cuda", dtype=torch.float16),
                              width=c["resolution"], height=c["resolution"],
                              num_inference_steps=c["num_inference_steps"], guidance_scale=c["guidance_scale"],
                              controlnet_conditioning_scale=c["controlnet_conditioning_scale"], eta=c["eta"])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if result.nsfw_content_detected and any(result.nsfw_content_detected):
                raise RuntimeError(f"Safety checker flagged {photo['id']}; result is not counted as successful")
            output = root / "images" / f"{photo['id']}.png"
            save_png(output, result.images[0])
            row = dict(run_id=run_id, photo_id=photo["id"], reference_id=reference["id"],
                       prompt=photo["prompt"], negative_prompt=c["negative_prompt"], seed=c["seed"],
                       models=c["models"], scheduler=c["scheduler"], scheduler_config=run["scheduler_config"],
                       num_inference_steps=c["num_inference_steps"], guidance_scale=c["guidance_scale"],
                       controlnet_conditioning_scale=c["controlnet_conditioning_scale"], ip_adapter_scale=c["ip_adapter_scale"],
                       canny_thresholds=[c["canny_low_threshold"], c["canny_high_threshold"]],
                       resolution=c["resolution"], dtype=c["dtype"], eta=c["eta"], cpu_offload=c["cpu_offload"],
                       environment=env, latent_method=LATENT_METHOD, data_identity=identity["data_identity"],
                       output=output.relative_to(root).as_posix(), output_sha256=file_sha(output),
                       latents=latent_path.relative_to(root).as_posix(), latents_sha256=file_sha(latent_path),
                       generation_seconds=elapsed, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                       timing_scope="pipeline call including CLIP image encoder, UNet, ControlNet, VAE and CPU transfers; model loading excluded; first image can include cold start overhead")
            write_json(root / "metadata" / f"{photo['id']}.json", row)
            export_records(root, complete_records(root, photos, run_id))
            print(f"Generated {photo['id']}: {elapsed:.1f}s", flush=True)
        del pipe
        torch.cuda.empty_cache()
    records = complete_records(root, photos, run_id)
    export_records(root, records)
    run.update(completed_images=len(records), complete=len(records) == len(photos))
    write_json(root / "run.json", run)
    from .visualization import comparison_grid
    comparison_grid(c, root, records, max_rows=6)
    print(f"Run {run_id}: {len(records)}/{len(photos)} images; {root}", flush=True)
    return root
