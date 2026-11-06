"""PDF из редактируемого Markdown и проверенных артефактов baseline."""
import csv
from html import escape
import math
import os
from pathlib import Path
import re

from .config import REPO_ROOT, data_dir, safe_path, work_root
from .data import validate_dataset
from .inference import complete_records, current_run
from .io import file_sha, read_json, read_jsonl, write_json
from .metrics import summarize


def register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    candidates = [Path(os.environ.get("REPORT_FONT_PATH", "__not_set__")),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    regular = next((path for path in candidates if path.is_file()), None)
    if regular is None:
        raise FileNotFoundError("Set REPORT_FONT_PATH to a TrueType font supporting Cyrillic")
    bold = regular.with_name("DejaVuSans-Bold.ttf" if regular.name == "DejaVuSans.ttf" else "arialbd.ttf")
    pdfmetrics.registerFont(TTFont("Body", str(regular)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(bold if bold.is_file() else regular)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")


def inline(text):
    result = escape(text).replace("`", "")
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    return re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#174e85">\1</link>', result)


def checked_materials(c, draft):
    if not draft:
        validate_dataset(c)
    try:
        root, run = current_run(c)
    except (FileNotFoundError, ValueError):
        if not draft:
            raise ValueError("No matching baseline. Run prepare_data, run_baseline and evaluate; use --draft only for an unfinished report")
        return None, None, [], None
    photos = read_jsonl(data_dir(c) / "coco_manifest.jsonl")
    records = complete_records(root, photos, run["run_id"])
    if draft:
        return root, run, records, None
    if len(records) != 24 or not run.get("complete"):
        raise ValueError("Final report requires 24 checked PNGs and latent files")
    summary = read_json(root / "summary.json")
    with (root / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    if summary["run_id"] != run["run_id"] or summary["n"] != 24 or len(metrics) != 24:
        raise ValueError("Metrics are not from this complete baseline")
    outputs = {row["photo_id"]: row["output_sha256"] for row in records}
    if {row["photo_id"] for row in metrics} != set(outputs):
        raise ValueError("Metric/photo IDs differ")
    for row in metrics:
        if row["output_sha256"] != outputs[row["photo_id"]] or row["evaluation_id"] != summary["evaluation_id"]:
            raise ValueError("Metrics refer to different generated images")
        for key in summary["metrics"]:
            row[key] = float(row[key])
    recalculated = summarize(metrics)
    for key, stats in recalculated.items():
        if any(not math.isclose(stats[field], summary["metrics"][key][field], rel_tol=1e-8, abs_tol=1e-10) for field in ("mean", "std", "n")):
            raise ValueError("Summary does not match per-image metrics")
    if summary["protocol"]["model"] != c["models"]["dino"] or summary["protocol"]["metrics"] != c["metrics"] or summary["protocol"]["metrics_code_sha"] != file_sha(Path(__file__).with_name("metrics.py")):
        raise ValueError("Evaluation protocol mismatch")
    return root, run, records, summary


def pipeline_drawing():
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
    from reportlab.lib import colors
    drawing = Drawing(510, 170)
    def box(x, y, width, text):
        drawing.add(Rect(x, y, width, 28, rx=4, ry=4, fillColor=colors.HexColor("#edf3fa"), strokeColor=colors.HexColor("#6b8197")))
        drawing.add(String(x + width / 2, y + 9, text, textAnchor="middle", fontName="Body", fontSize=10))
    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#486278")))
        drawing.add(Polygon([x2, y2, x2 - 5, y2 + 3, x2 - 5, y2 - 3], fillColor=colors.HexColor("#486278"), strokeColor=None))
    box(0, 115, 145, "Фото → Canny")
    box(175, 115, 130, "ControlNet")
    box(0, 70, 145, "Текст → CLIP")
    box(0, 25, 145, "Стиль → CLIP vision")
    box(175, 25, 130, "IP-Adapter")
    box(355, 70, 150, "SD 1.5 / UNet")
    arrow(145, 129, 175, 129)
    arrow(305, 129, 355, 90)
    arrow(145, 84, 355, 84)
    arrow(145, 39, 175, 39)
    arrow(305, 39, 355, 77)
    drawing.add(String(430, 145, "Начальный шум z_T", textAnchor="middle", fontName="Body", fontSize=10))
    drawing.add(Line(430, 136, 430, 102, strokeColor=colors.grey))
    drawing.add(Line(430, 70, 430, 42, strokeColor=colors.grey))
    drawing.add(String(430, 25, "VAE → результат", textAnchor="middle", fontName="Body", fontSize=10))
    return drawing


def build_report(c, draft=False, render=False):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
    import pymupdf

    register_fonts()
    root, run, records, summary = checked_materials(c, draft)
    styles = {
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.3, leading=14.0, spaceAfter=7, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", fontName="BodyBold", fontSize=14, leading=18, spaceBefore=2, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="BodyBold", fontSize=11.3, leading=15, spaceBefore=7, spaceAfter=7),
        "table": ParagraphStyle("table", fontName="Body", fontSize=10, leading=13, spaceAfter=0),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=10, leading=13, spaceAfter=8),
    }
    width = A4[0] - 72
    def table(rows, widths=None):
        cells = [[Paragraph(inline(str(value)), styles["table"]) for value in row] for row in rows]
        result = Table(cells, colWidths=widths or [width / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#899bae")),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d4dce5")),
        ]))
        return result
    config_rows = [["Параметр", "Значение"], ["Разрешение / шаги", f"{c['resolution']} × {c['resolution']} / {c['num_inference_steps']}"],
                   ["Scheduler / seed / eta", f"{c['scheduler']} / {c['seed']} / {c['eta']}"],
                   ["CFG / ControlNet / IP-Adapter", f"{c['guidance_scale']} / {c['controlnet_conditioning_scale']} / {c['ip_adapter_scale']}"],
                   ["Canny / точность", f"{c['canny_low_threshold']}, {c['canny_high_threshold']} / {c['dtype']}"],
                   ["Экономия VRAM", "CPU offload" if c["cpu_offload"] else "Все модели на GPU"]]
    if summary:
        labels = {"edge_precision": "Edge precision", "edge_recall": "Edge recall", "edge_f1": "Edge F1", "dino_cosine": "DINOv2 cosine", "generation_seconds": "Время, с", "peak_allocated_mib": "Пик allocated, МиБ"}
        metric_rows = [["Показатель", "Среднее ± SD", "n"]]
        for key, label in labels.items():
            stats = summary["metrics"][key]
            metric_rows.append([label, f"{stats['mean']:.3f} ± {stats['std']:.3f}", stats["n"]])
        metrics_block = [table(metric_rows, [width * .43, width * .37, width * .20]), Spacer(1, 8)]
    else:
        metrics_block = [Paragraph("Экспериментальные метрики отсутствуют. Воспроизведение baseline ещё не подтверждено; численные значения и выводы не подставляются.", styles["body"])]
    figures = []
    if records:
        from .visualization import comparison_grid
        path = comparison_grid(c, root, records, max_rows=2, filename="report_examples.png")
        from PIL import Image as PILImage
        with PILImage.open(path) as image:
            figures.append(Image(str(path), width=width, height=width * image.height / image.width))
        figures.append(Paragraph("Рис. 2. Первые два примера фиксированного отбора: исходное фото, Canny, референс, результат. Полная таблица первых шести сцен сохранена отдельно; отбор не основан на значениях метрик.", styles["caption"]))
    else:
        figures.append(Paragraph("Иллюстрации реальных генераций появятся после GPU-запуска. Макеты результатов не используются.", styles["body"]))
    data_valid = False
    if (data_dir(c) / "manifest.json").exists():
        try:
            validate_dataset(c)
            data_valid = True
        except (FileNotFoundError, ValueError):
            if not draft:
                raise
    data_status = (f"Подготовлены {c['art']['total']} картин: {c['art']['total'] - c['art']['references']} для будущего обучения и {c['art']['references']} отдельных референсов; 24 фотографии COCO."
                   if data_valid else
                   f"Планируется подготовить {c['art']['total']} картин ({c['art']['total'] - c['art']['references']} для будущего обучения, {c['art']['references']} референсов) и 24 фотографии COCO. Фактическая подготовка ещё не подтверждена.")
    run_status = (f"Прогон {run['run_id']}; сохранено {len(records)}/24 изображений. GPU: {run['environment']['gpu']}." if run else
                  "Реальный GPU-прогон ещё не выполнен. Этот документ является незавершённым черновиком.")
    if draft:
        run_status += " Документ собран в режиме --draft и не готов к сдаче."
    blocks = {"{{CONFIG_TABLE}}": [table(config_rows, [width * .56, width * .44]), Spacer(1, 8)],
              "{{METRICS_TABLE}}": metrics_block, "{{FIGURES}}": figures,
              "{{PIPELINE}}": [pipeline_drawing(), Paragraph("Рис. 1. Пайплайн baseline. В будущем IP-Adapter заменяется на LoRA в UNet; ControlNet и остальные условия сохраняются.", styles["caption"])]}
    substitutions = {"{{DATA_STATUS}}": data_status, "{{RUN_STATUS}}": run_status,
                     "{{EDGE_TOLERANCE}}": str(c["metrics"]["edge_tolerance_px"])}
    source = (REPO_ROOT / "docs/stage1_report.md").read_text(encoding="utf-8")
    for token, value in substitutions.items():
        source = source.replace(token, value)
    story, pending = [], []
    def flush():
        if pending:
            story.append(Paragraph(inline(" ".join(pending)), styles["body"]))
            pending.clear()
    lines, index = source.splitlines(), 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
        elif line == "<!-- PAGE -->":
            flush(); story.append(PageBreak())
        elif line in blocks:
            flush(); story.extend(blocks[line])
        elif line.startswith("#"):
            flush(); level = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(inline(line.lstrip("# ")), styles["h1" if level == 1 else "h2"]))
        elif line.startswith("|"):
            flush(); rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [item.strip() for item in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", item) for item in row):
                    rows.append(row)
                index += 1
            story.extend([table(rows), Spacer(1, 8)])
            continue
        elif line.startswith("- "):
            flush(); story.append(Paragraph("• " + inline(line[2:]), styles["body"]))
        elif re.match(r"^\d+\.\s", line):
            flush(); story.append(Paragraph(inline(line), styles["body"]))
        else:
            pending.append(line)
        index += 1
    flush()
    directory = work_root() / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ("stage1_report_draft.pdf" if draft else "stage1_report.pdf")
    temporary = destination.with_suffix(".building.pdf")
    def footer(canvas, document):
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.HexColor("#566273"))
        label = "ЧЕРНОВИК · экспериментальные данные не подтверждены для сдачи" if draft else f"Baseline {run['run_id']}"
        canvas.drawString(36, A4[1] - 25, label)
        canvas.drawRightString(A4[0] - 36, 23, str(document.page))
    document = SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=42, bottomMargin=36,
                                 title="Диффузионные методы переноса стиля — этап 1", author="Учебный исследовательский проект")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    with pymupdf.open(temporary) as pdf:
        page_count = len(pdf)
        texts = [page.get_text() for page in pdf]
        if not 4 <= page_count <= 6:
            raise ValueError(f"Report has {page_count} pages; required 4–6. Inspect {temporary}")
        if not any("Диффузионные" in text for text in texts) or any("\ufffd" in text for text in texts):
            raise ValueError("Cyrillic text extraction failed")
        if render:
            previews = directory / ("draft_preview" if draft else "preview")
            previews.mkdir(exist_ok=True)
            for number, page in enumerate(pdf, 1):
                page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(previews / f"page_{number}.png")
    os.replace(temporary, destination)
    write_json(destination.with_suffix(".json"), {"draft": draft, "page_count": page_count,
               "run_id": run["run_id"] if run else None, "evaluated_images": summary["n"] if summary else 0,
               "source": "docs/stage1_report.md"})
    print(f"{'DRAFT' if draft else 'FINAL'}: {destination}; {page_count} pages", flush=True)
    return destination
