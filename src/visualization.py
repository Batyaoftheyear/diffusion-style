"""Компактные таблицы изображений без изменения экспериментальных данных."""
from PIL import Image, ImageDraw

from .config import data_dir, safe_path
from .io import read_json, read_jsonl
from .preprocessing import load_rgb, prepare_image, save_png


def comparison_grid(c, run_root, records, max_rows=6, filename="comparisons.png"):
    if not records:
        return None
    root = data_dir(c)
    photos = {row["id"]: row for row in read_jsonl(root / "coco_manifest.jsonl")}
    art = {row["id"]: row for row in read_jsonl(root / "art_manifest.jsonl")}
    records = records[:max_rows]
    tile, header, footer = 256, 24, 24
    grid = Image.new("RGB", (tile * 4, header + len(records) * (tile + footer)), "white")
    draw = ImageDraw.Draw(grid)
    for index, title in enumerate(("Original", "Canny", "Style Reference", "Generated")):
        draw.text((index * tile + 8, 6), title, fill="black")
    for index, row in enumerate(records):
        photo = photos[row["photo_id"]]
        ref = art[row["reference_id"]]
        paths = [safe_path(root, photo["prepared"]), safe_path(root, photo["canny"]),
                 safe_path(root, ref["prepared"]), safe_path(run_root, row["output"])]
        y = header + index * (tile + footer)
        for column, path in enumerate(paths):
            image = load_rgb(path).resize((tile, tile), Image.Resampling.LANCZOS)
            grid.paste(image, (column * tile, y))
        draw.text((8, y + tile + 4), f"{row['photo_id']} | {photo['scene_group']} | seed={row['seed']}", fill="black")
    path = run_root / "figures" / filename
    save_png(path, grid)
    return path


def data_preview(c):
    root = data_dir(c)
    art = read_jsonl(root / "art_manifest.jsonl")
    photos = read_jsonl(root / "coco_manifest.jsonl")
    rows = [("Reference", item) for item in art[:c["art"]["references"]]] + [("Photo", item) for item in photos[:6]]
    columns, tile, footer = 4, 192, 28
    grid = Image.new("RGB", (columns * tile, ((len(rows) + columns - 1) // columns) * (tile + footer)), "white")
    draw = ImageDraw.Draw(grid)
    for index, (kind, row) in enumerate(rows):
        x, y = index % columns * tile, index // columns * (tile + footer)
        grid.paste(prepare_image(load_rgb(safe_path(root, row["prepared"])), tile), (x, y))
        draw.text((x + 4, y + tile + 4), f"{kind}: {row['id']}", fill="black")
    path = root / "preview.png"
    save_png(path, grid)
    return path
