"""Атомарная запись и проверка сохранённых файлов."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path, rows):
    import io
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot export an empty table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode("utf-8-sig"))


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment(include_gpu=False):
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "Pillow", "opencv-python-headless", "opencv-python", "datasets", "huggingface-hub", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.system(), "packages": packages}
    if include_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required; run this command in Colab")
        result.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return result
