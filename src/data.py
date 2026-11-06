"""Подготовка небольших выборок WikiArt и COCO, кэш и manifests."""
from collections import Counter
import io
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from PIL import Image

from .config import data_dir, data_settings, fingerprint, safe_path
from .io import atomic_write, file_sha, read_json, read_jsonl, write_json, write_jsonl
from .preprocessing import canny_map, dhash, load_rgb, pixel_sha, prepare_image, save_png

VIEWER = "https://datasets-server.huggingface.co"
GROUPS = {
    "people": r"\b(man|woman|boy|girl|person|people|child|children)\b",
    "animals": r"\b(cat|dog|horse|bird|zebra|giraffe|elephant|cow|sheep|bear)\b",
    "urban": r"\b(bus|train|street|traffic|city|building|road|intersection)\b",
    "nature": r"\b(beach|mountain|forest|lake|ocean|field|river|snow|landscape)\b",
    "objects": r"\b(food|bowl|plate|fruit|clock|vase|cup|laptop|phone|bottle|pizza|sandwich)\b",
    "interiors": r"\b(room|kitchen|bathroom|bedroom|couch|sofa|bed|toilet|oven)\b",
}


def fetch(url, max_bytes=32 * 1024 * 1024):
    request = Request(url, headers={"User-Agent": "diffusion-style-stage1/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise ValueError(f"Response exceeds {max_bytes} bytes")
                return body, response.headers
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except URLError:
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable download state")


def viewer_page(c, root, offset, length=100):
    cache = root / "cache" / f"wikiart_{offset}_{length}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 300:
        response = read_json(cache)
    else:
        query = urlencode({"dataset": c["art"]["dataset"], "config": "default", "split": "train", "offset": offset, "length": length})
        body, headers = fetch(f"{VIEWER}/rows?{query}")
        response = {"revision": headers.get("x-revision"), "payload": json.loads(body)}
        write_json(cache, response)
    if response["revision"] != c["art"]["revision"]:
        raise ValueError("WikiArt viewer revision differs from pinned revision; do not silently change the data")
    return response["payload"]


def rank_candidates(rows, seed, id_key):
    return sorted(rows, key=lambda row: fingerprint([seed, row[id_key]]))


def select_coco(rows, seed, count=24):
    if count != 24:
        raise ValueError("The test protocol uses 24 images")
    ordered = rank_candidates(rows, seed, "image_id")
    pools = {name: [row for row in ordered if re.search(pattern, " ".join(row["captions"]).lower())]
             for name, pattern in GROUPS.items()}
    selected, used = [], set()
    for _ in range(4):
        for name, pool in pools.items():
            candidate = next((row for row in pool if row["image_id"] not in used), None)
            if candidate is None:
                raise ValueError(f"Not enough unique COCO candidates in group {name}")
            used.add(candidate["image_id"])
            selected.append(dict(candidate, scene_group=name))
    return selected


def coco_download_url(source_url):
    parsed = urlparse(source_url)
    if parsed.hostname != "images.cocodataset.org" or not re.fullmatch(r"/val2017/[0-9]{12}\.jpg", parsed.path):
        raise ValueError("Expected an official COCO val2017 image URL")
    # The images.cocodataset.org alias currently has a mismatched HTTPS certificate.
    # S3 path-style access retains TLS verification for the same public bucket.
    return "https://s3.amazonaws.com/images.cocodataset.org" + parsed.path


def download_image(url, path, expected_sha=None):
    path = Path(path)
    if path.exists() and (expected_sha is None or file_sha(path) == expected_sha):
        with Image.open(path) as image:
            image.verify()
        return
    body, _ = fetch(url)
    with Image.open(io.BytesIO(body)) as image:
        image.verify()
    if expected_sha is not None:
        import hashlib
        if hashlib.sha256(body).hexdigest() != expected_sha:
            raise ValueError("Remote image changed; refusing to replace a frozen input")
    atomic_write(path, body)


def valid_item(root, row):
    return all(safe_path(root, name).is_file() and file_sha(safe_path(root, name)) == sha
               for name, sha in row["files"].items())


def item_record(root, source_path, image_id, metadata, c, edges=False):
    original = load_rgb(source_path)
    prepared = prepare_image(original, c["resolution"])
    prepared_path = root / "prepared" / f"{image_id}.png"
    save_png(prepared_path, prepared)
    row = dict(metadata, id=image_id, original=source_path.relative_to(root).as_posix(),
               prepared=prepared_path.relative_to(root).as_posix(), original_size=list(original.size),
               pixel_sha=pixel_sha(original), prepared_pixel_sha=pixel_sha(prepared), dhash=f"{dhash(original):016x}")
    paths = [source_path, prepared_path]
    if edges:
        edge_path = root / "canny" / f"{image_id}.png"
        save_png(edge_path, canny_map(prepared, c["canny_low_threshold"], c["canny_high_threshold"]))
        row["canny"] = edge_path.relative_to(root).as_posix()
        paths.append(edge_path)
    if "caption" in metadata:
        caption_path = root / "captions" / f"{image_id}.txt"
        atomic_write(caption_path, (metadata["caption"] + "\n").encode("utf-8"))
        row["caption_file"] = caption_path.relative_to(root).as_posix()
        paths.append(caption_path)
    row["files"] = {path.relative_to(root).as_posix(): file_sha(path) for path in paths}
    write_json(root / "items" / f"{image_id}.json", row)
    return row


def prepare_art(c, root):
    catalogue_path = root / "art_catalogue.json"
    if catalogue_path.exists():
        catalogue = read_json(catalogue_path)
    else:
        rows, labels = [], None
        for offset in range(0, c["art"]["scan_rows"], 100):
            page = viewer_page(c, root, offset, min(100, c["art"]["scan_rows"] - offset))
            labels = {item["name"]: item["type"].get("names") for item in page["features"]}
            style_id = labels["style"].index(c["art"]["style"])
            rows.extend({"row_idx": entry["row_idx"], **{k: entry["row"][k] for k in ("artist", "genre", "style")}}
                        for entry in page["rows"] if entry["row"]["style"] == style_id)
            print(f"WikiArt metadata: {offset + len(page['rows'])} rows inspected", flush=True)
        catalogue = {"candidates": rank_candidates(rows, c["seed"], "row_idx"), "labels": labels}
        write_json(catalogue_path, catalogue)
    selected = []
    genres = {"landscape": "a landscape", "cityscape": "a city scene", "portrait": "a portrait",
              "still_life": "a still life", "genre_painting": "an everyday scene"}
    for candidate in catalogue["candidates"]:
        index = candidate["row_idx"]
        image_id = f"wikiart_{index:08d}"
        record_path = root / "items" / f"{image_id}.json"
        previous = read_json(record_path) if record_path.exists() else None
        if previous and valid_item(root, previous):
            row = previous
        else:
            page = viewer_page(c, root, index // 100 * 100)
            entry = next(entry for entry in page["rows"] if entry["row_idx"] == index)
            if any(entry["row"][key] != candidate[key] for key in ("artist", "genre", "style")):
                raise ValueError("WikiArt catalogue no longer matches the pinned rows")
            source_path = root / "originals" / f"{image_id}.jpg"
            expected = previous["files"].get(previous["original"]) if previous else None
            download_image(entry["row"]["image"]["src"], source_path, expected)
            genre = catalogue["labels"]["genre"][candidate["genre"]]
            artist = catalogue["labels"]["artist"][candidate["artist"]]
            caption = f"A painting in the {c['art']['style'].replace('_', ' ').lower()} style, depicting {genres.get(genre, 'a ' + genre.replace('_', ' '))}."
            metadata = dict(source_dataset=c["art"]["dataset"], source_revision=c["art"]["revision"],
                            source_row=index, artist=artist, genre=genre, style=c["art"]["style"],
                            source_url=f"https://huggingface.co/datasets/{c['art']['dataset']}/viewer/default/train?row={index}",
                            rights="Original authors; non-commercial research per dataset card; artwork-specific rights require review",
                            caption=caption, caption_method="genre template; manual review required before future training")
            row = item_record(root, source_path, image_id, metadata, c)
        if any(row["pixel_sha"] == old["pixel_sha"] or row["prepared_pixel_sha"] == old["prepared_pixel_sha"]
               or (int(row["dhash"], 16) ^ int(old["dhash"], 16)).bit_count() <= c["art"]["dedup_hamming"] for old in selected):
            continue
        row = dict(row, split="reference" if len(selected) < c["art"]["references"] else "future_train")
        selected.append(row)
        print(f"WikiArt: {len(selected)}/{c['art']['total']}", flush=True)
        if len(selected) == c["art"]["total"]:
            break
    if len(selected) != c["art"]["total"]:
        raise ValueError("Insufficient unique artworks; increase art.scan_rows in a new configuration")
    write_jsonl(root / "art_manifest.jsonl", selected)
    return selected


def prepare_coco(c, root):
    selection_path = root / "coco_selection.json"
    if selection_path.exists():
        selected = read_json(selection_path)
    else:
        from datasets import load_dataset
        metadata = load_dataset(c["coco"]["dataset"], split="validation", revision=c["coco"]["revision"], streaming=True)
        selected = select_coco(list(metadata), c["seed"], c["coco"]["count"])
        write_json(selection_path, selected)
    result = []
    for source in selected:
        image_id = f"coco_{source['image_id']:012d}"
        record_path = root / "items" / f"{image_id}.json"
        previous = read_json(record_path) if record_path.exists() else None
        if previous and valid_item(root, previous):
            result.append(previous)
            continue
        source_path = root / "originals" / f"{image_id}.jpg"
        expected = previous["files"].get(previous["original"]) if previous else None
        url = coco_download_url(source["coco_url"])
        download_image(url, source_path, expected)
        captions = sorted(zip(source["ids"], source["captions"]))
        metadata = dict(source_dataset=c["coco"]["dataset"], source_revision=c["coco"]["revision"],
                        coco_id=source["image_id"], source_url=source["coco_url"], download_url=url, flickr_url=source["flickr_url"],
                        license_id=source["license"], scene_group=source["scene_group"], prompt=captions[0][1].strip(),
                        caption_id=captions[0][0], all_captions=source["captions"], split="test")
        result.append(item_record(root, source_path, image_id, metadata, c, edges=True))
        print(f"COCO: {len(result)}/{c['coco']['count']}", flush=True)
    write_jsonl(root / "coco_manifest.jsonl", result)
    return result


def validate_dataset(c):
    root = data_dir(c)
    info = read_json(root / "manifest.json")
    if info["settings"] != data_settings(c):
        raise ValueError("Prepared data configuration mismatch")
    for name, sha in info["manifests"].items():
        if file_sha(safe_path(root, name)) != sha:
            raise ValueError(f"Manifest changed: {name}")
    art, photos = read_jsonl(root / "art_manifest.jsonl"), read_jsonl(root / "coco_manifest.jsonl")
    if len(art) != c["art"]["total"] or len(photos) != 24:
        raise ValueError("Unexpected dataset size")
    if len({row["id"] for row in art + photos}) != len(art) + len(photos):
        raise ValueError("Duplicate image IDs")
    if Counter(row["split"] for row in art) != {"reference": c["art"]["references"], "future_train": c["art"]["total"] - c["art"]["references"]}:
        raise ValueError("Invalid artwork split")
    if info["main_reference_id"] not in {row["id"] for row in art if row["split"] == "reference"}:
        raise ValueError("Main reference is not held out")
    if len({row["pixel_sha"] for row in art + photos}) != len(art) + len(photos):
        raise ValueError("Duplicate pixels across datasets")
    for row in art + photos:
        if not valid_item(root, row):
            raise ValueError(f"Missing or modified artifact: {row['id']}; rerun prepare_data")
        if load_rgb(safe_path(root, row["prepared"])).size != (c["resolution"], c["resolution"]):
            raise ValueError(f"Invalid image size: {row['id']}")
    for row in photos:
        prepared = load_rgb(safe_path(root, row["prepared"]))
        expected = canny_map(prepared, c["canny_low_threshold"], c["canny_high_threshold"])
        if pixel_sha(load_rgb(safe_path(root, row["canny"]))) != pixel_sha(expected):
            raise ValueError(f"Canny mismatch: {row['id']}")
        if not row["prompt"].strip():
            raise ValueError("Empty prompt")
    return {"artworks": len(art), "future_train": len(art) - c["art"]["references"], "references": c["art"]["references"],
            "photos": len(photos), "scene_groups": dict(Counter(row["scene_group"] for row in photos)), "data_dir": str(root)}


def prepare_data(c):
    root = data_dir(c)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        try:
            return validate_dataset(c)
        except (FileNotFoundError, ValueError) as error:
            print(f"Restoring prepared data: {error}", flush=True)
    art = prepare_art(c, root)
    prepare_coco(c, root)
    write_json(root / "manifest.json", {"settings": data_settings(c), "main_reference_id": art[0]["id"],
               "manifests": {name: file_sha(root / name) for name in ("art_manifest.jsonl", "coco_manifest.jsonl")}})
    return validate_dataset(c)
