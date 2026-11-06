"""CPU-проверки. Все синтетические файлы создаются только во временных каталогах."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from src.config import REPO_ROOT, data_dir, data_settings, fingerprint, load_config, safe_path, validate_config
from src.data import coco_download_url, item_record, prepare_data, rank_candidates, select_coco, validate_dataset
from src.inference import cached_record, current_run, data_identity, generation_code_sha, make_latents, save_latents
from src.io import atomic_write, file_sha, read_json, read_jsonl, write_csv, write_json, write_jsonl
from src.metrics import cosine_similarity, edge_metrics, summarize
from src.preprocessing import canny_map, load_rgb, prepare_image, save_png


class ConfigTests(unittest.TestCase):
    def test_baseline_defaults(self):
        c = load_config()
        self.assertEqual((c["resolution"], c["num_inference_steps"], c["seed"]), (512, 30, 42))
        self.assertEqual(c["models"]["ip_adapter"]["weight_name"], "ip-adapter_sd15.safetensors")

    def test_invalid_parameters(self):
        for key, value in [("resolution", 513), ("seed", True), ("eta", 1.0), ("ip_adapter_scale", float("nan")),
                           ("canny_low_threshold", 201), ("scheduler", "PNDM"), ("dtype", "float32")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                c = load_config(); c[key] = value; validate_config(c)

    def test_revisions_must_be_pinned(self):
        c = load_config(); c["models"]["sd"]["revision"] = "main"
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_fingerprint_order_and_changes(self):
        self.assertEqual(fingerprint({"x": 1, "y": 2}), fingerprint({"y": 2, "x": 1}))
        self.assertNotEqual(fingerprint({"seed": 42}), fingerprint({"seed": 43}))

    def test_data_identity_ignores_inference_weight(self):
        first = load_config(); second = copy.deepcopy(first); second["ip_adapter_scale"] = 0.9
        self.assertEqual(data_dir(first), data_dir(second))
        second["canny_low_threshold"] = 80
        self.assertNotEqual(data_dir(first), data_dir(second))


class PreprocessingTests(unittest.TestCase):
    def test_crop_keeps_circle_round(self):
        image = Image.new("RGB", (400, 200), "black")
        ImageDraw.Draw(image).ellipse((160, 60, 240, 140), fill="white")
        output = prepare_image(image)
        ys, xs = np.where(np.asarray(output.convert("L")) > 200)
        self.assertEqual(output.size, (512, 512))
        self.assertLessEqual(abs((xs.max() - xs.min()) - (ys.max() - ys.min())), 2)

    def test_canny_finds_real_boundary(self):
        image = Image.new("RGB", (512, 512), "black")
        ImageDraw.Draw(image).rectangle((100, 100, 400, 400), fill="white")
        edges = np.asarray(canny_map(image))
        self.assertGreater(np.count_nonzero(edges[:, :, 0]), 100)
        self.assertEqual(np.count_nonzero(edges[200:300, 200:300]), 0)
        np.testing.assert_array_equal(edges[:, :, 0], edges[:, :, 2])
        np.testing.assert_array_equal(edges, np.asarray(canny_map(image)))

    def test_exif_orientation_and_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (80, 40), "red")
            exif = image.getexif(); exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(load_rgb(path).size, (40, 80))
            self.assertEqual(prepare_image(Image.new("RGBA", (30, 60))).mode, "RGB")

    def test_invalid_canny(self):
        with self.assertRaises(ValueError):
            canny_map(Image.new("RGB", (16, 16)), 200, 100)


class MetricTests(unittest.TestCase):
    def test_precision_and_recall_have_different_denominators(self):
        source = np.zeros((12, 12), dtype=bool); generated = source.copy()
        source[3, 3] = source[8, 8] = True; generated[3, 4] = True
        result = edge_metrics(source, generated, tolerance=1)
        self.assertEqual(result["edge_precision"], 1)
        self.assertEqual(result["edge_recall"], 0.5)
        self.assertAlmostEqual(result["edge_f1"], 2 / 3)

    def test_edge_tolerance_and_empty(self):
        a = np.zeros((10, 10), dtype=bool); b = a.copy()
        a[3, 3] = True; b[3, 5] = True
        self.assertEqual(edge_metrics(a, b, 1)["edge_f1"], 0)
        self.assertEqual(edge_metrics(a, b, 2)["edge_f1"], 1)
        self.assertEqual(edge_metrics(a & False, b & False)["edge_f1"], 0)
        with self.assertRaises(ValueError):
            edge_metrics(a, b[:5])

    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [2, 4]), 1)
        self.assertAlmostEqual(cosine_similarity([1, 2], [-2, -4]), -1)
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0)
        for a, b in [([0, 0], [1, 0]), ([1], [1, 2]), ([float("nan")], [1])]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                cosine_similarity(a, b)

    def test_summary_uses_sample_std(self):
        keys = ("edge_precision", "edge_recall", "edge_f1", "dino_cosine", "generation_seconds", "peak_allocated_mib", "peak_reserved_mib")
        result = summarize([{key: value for key in keys} for value in (1, 3)])
        self.assertEqual(result["edge_f1"]["mean"], 2)
        self.assertAlmostEqual(result["edge_f1"]["std"], 2**0.5)
        self.assertEqual(result["edge_f1"]["n"], 2)


class SelectionTests(unittest.TestCase):
    def test_coco_secure_download_url_preserves_bucket_and_image_id(self):
        self.assertEqual(coco_download_url("http://images.cocodataset.org/val2017/000000397133.jpg"),
                         "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000397133.jpg")
        for url in ("https://example.com/val2017/000000397133.jpg", "http://images.cocodataset.org/train2017/000000397133.jpg"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                coco_download_url(url)

    def test_selection_is_unique_balanced_and_order_independent(self):
        words = ["a person walking", "a dog running", "a city street", "a beach landscape", "a plate of food", "a kitchen room"]
        rows = [{"image_id": group * 100 + index, "captions": [caption]} for group, caption in enumerate(words) for index in range(6)]
        selected = select_coco(rows, 42)
        self.assertEqual(selected, select_coco(list(reversed(rows)), 42))
        self.assertEqual(len({row["image_id"] for row in selected}), 24)
        for group in {row["scene_group"] for row in selected}:
            self.assertEqual(sum(row["scene_group"] == group for row in selected), 4)
        self.assertEqual(len({row["scene_group"] for row in selected[:6]}), 6)
        self.assertNotEqual(selected, select_coco(rows, 123))

    def test_insufficient_candidates_fail(self):
        with self.assertRaises(ValueError):
            select_coco([{"image_id": 1, "captions": ["a cat"]}], 42)


class ArtifactTests(unittest.TestCase):
    def test_atomic_roundtrip_and_path_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = {"text": "Кириллица", "seed": 42}
            write_json(root / "config.json", value)
            self.assertEqual(read_json(root / "config.json"), value)
            write_jsonl(root / "manifest.jsonl", [value])
            self.assertEqual(read_jsonl(root / "manifest.jsonl"), [value])
            write_csv(root / "table.csv", [value])
            self.assertIn("Кириллица", (root / "table.csv").read_text(encoding="utf-8-sig"))
            first = file_sha(root / "config.json")
            write_json(root / "config.json", {"seed": 43})
            self.assertNotEqual(first, file_sha(root / "config.json"))
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(safe_path(root, "images/test.png"), root / "images/test.png")
            for path in ("../outside", str(root.parent / "absolute")):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    safe_path(root, path)

    def test_latents_are_fixed_and_do_not_change_global_rng(self):
        import torch
        before = torch.random.get_rng_state().clone()
        first = make_latents(42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, make_latents(42)))
        self.assertFalse(torch.equal(first, make_latents(123)))
        self.assertEqual(tuple(first.shape), (1, 4, 64, 64))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "noise.npy"
            save_latents(destination, first)
            np.testing.assert_array_equal(np.load(destination, allow_pickle=False), first.numpy())

    def test_resume_rejects_damaged_files_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); photo_id = "test_fixture"; run_id = "fixture_only"
            save_png(root / "image.png", Image.new("RGB", (512, 512), "red"))
            save_latents(root / "noise.npy", make_latents(42))
            row = {"photo_id": photo_id, "run_id": run_id, "output": "image.png", "latents": "noise.npy",
                   "output_sha256": file_sha(root / "image.png"), "latents_sha256": file_sha(root / "noise.npy")}
            write_json(root / "metadata" / f"{photo_id}.json", row)
            self.assertEqual(cached_record(root, photo_id, run_id), row)
            with self.assertRaises(ValueError):
                cached_record(root, photo_id, "different_run")
            atomic_write(root / "noise.npy", b"broken")
            self.assertIsNone(cached_record(root, photo_id, run_id))


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {"DIFFUSION_STYLE_HOME": cls.temporary.name}); cls.env.start()
        cls.config = load_config(); cls.config["art"]["total"] = 60
        cls.root = data_dir(cls.config)
        art, photos = [], []
        for index in range(84):
            image_id = f"fixture_{index:03d}"
            source = cls.root / "originals" / f"{image_id}.png"
            save_png(source, Image.new("RGB", (80, 60), (index * 17 % 256, index * 47 % 256, index * 73 % 256)))
            if index < 60:
                metadata = {"split": "reference" if index < 5 else "future_train", "caption": "Synthetic unit-test fixture"}
                art.append(item_record(cls.root, source, image_id, metadata, cls.config))
            else:
                metadata = {"split": "test", "prompt": "Synthetic unit-test fixture", "scene_group": "fixture"}
                photos.append(item_record(cls.root, source, image_id, metadata, cls.config, edges=True))
        write_jsonl(cls.root / "art_manifest.jsonl", art); write_jsonl(cls.root / "coco_manifest.jsonl", photos)
        write_json(cls.root / "manifest.json", {"settings": data_settings(cls.config), "main_reference_id": art[0]["id"],
                   "manifests": {name: file_sha(cls.root / name) for name in ("art_manifest.jsonl", "coco_manifest.jsonl")}})
        cls.art, cls.photos = art, photos

    @classmethod
    def tearDownClass(cls):
        cls.env.stop(); cls.temporary.cleanup()

    def test_validate_real_saved_manifest_artifacts(self):
        result = validate_dataset(self.config)
        self.assertEqual((result["references"], result["photos"]), (5, 24))

    def test_complete_preparation_resumes_without_network(self):
        with patch("src.data.fetch", side_effect=AssertionError("Completed data must not touch the network")):
            self.assertEqual(prepare_data(self.config)["photos"], 24)

    def test_modified_manifest_is_rejected(self):
        path = self.root / "coco_manifest.jsonl"; original = path.read_bytes()
        try:
            atomic_write(path, original + b"\n")
            with self.assertRaises(ValueError):
                validate_dataset(self.config)
        finally:
            atomic_write(path, original)

    def test_missing_prepared_image_is_rejected(self):
        path = self.root / self.photos[0]["prepared"]; original = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(ValueError):
                validate_dataset(self.config)
        finally:
            atomic_write(path, original)

    def make_saved_run_fixture(self, count=23):
        run_id = "unit_test_" + self._testMethodName
        root = Path(self.temporary.name) / "results" / run_id
        save_latents(root / "noise.npy", make_latents(42))
        for photo in self.photos[:count]:
            path = root / "images" / f"{photo['id']}.png"
            atomic_write(path, (self.root / photo["prepared"]).read_bytes())
            row = {"photo_id": photo["id"], "run_id": run_id, "output": path.relative_to(root).as_posix(),
                   "latents": "noise.npy", "output_sha256": file_sha(path), "latents_sha256": file_sha(root / "noise.npy"),
                   "reference_id": self.art[0]["id"], "prompt": "Synthetic CPU unit-test fixture", "negative_prompt": "",
                   "seed": 42, "generation_seconds": 0.1, "peak_allocated_mib": 1.0, "peak_reserved_mib": 1.0}
            write_json(root / "metadata" / f"{photo['id']}.json", row)
        run = {"run_id": run_id, "config": self.config, "data_identity": data_identity(self.config),
               "generation_code_sha": generation_code_sha(), "main_reference_id": self.art[0]["id"],
               "expected_images": 24, "completed_images": 2, "complete": False}
        write_json(root / "run.json", run)
        write_json(Path(self.temporary.name) / "results/current_run.json", {"run_id": run_id})
        return root

    def test_finalize_skip_preserves_saved_images_and_run_id(self):
        from src.inference import complete_records
        from src.run_status import finalize_run, reuse_finalized_run, verified_outcomes
        root = self.make_saved_run_fixture()
        before = {path: file_sha(path) for path in list((root / "images").glob("*.png")) + [root / "noise.npy"]}
        old_run_id = read_json(root / "run.json")["run_id"]
        self.assertEqual(finalize_run(self.config, [self.photos[-1]["id"]], "safety_checker"), root)
        run = read_json(root / "run.json")
        self.assertEqual(run["run_id"], old_run_id)
        self.assertEqual((run["completed_images"], run["skipped_images"], run["processed_images"]), (23, 1, 24))
        self.assertFalse(run["complete"])
        self.assertTrue(run["processed_complete"])
        self.assertTrue(all(file_sha(path) == sha for path, sha in before.items()))
        records = complete_records(root, self.photos, old_run_id)
        skipped, identity = verified_outcomes(root, run, self.photos, records)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(identity, run["outcomes_id"])
        self.assertTrue(reuse_finalized_run(self.config))
        self.assertEqual(finalize_run(self.config, [self.photos[-1]["id"]], "safety_checker"), root)

    def test_unexplained_missing_photo_cannot_finalize(self):
        from src.run_status import finalize_run
        root = self.make_saved_run_fixture()
        before = (root / "run.json").read_bytes()
        with self.assertRaises(ValueError):
            finalize_run(self.config)
        self.assertEqual((root / "run.json").read_bytes(), before)

    def test_successful_photo_cannot_be_replaced_by_skip(self):
        from src.run_status import finalize_run
        self.make_saved_run_fixture()
        with self.assertRaises(ValueError):
            finalize_run(self.config, [self.photos[0]["id"]], "safety_checker")

    def test_modified_skip_annotation_is_rejected(self):
        from src.inference import complete_records
        from src.run_status import finalize_run, verified_outcomes
        root = self.make_saved_run_fixture()
        finalize_run(self.config, [self.photos[-1]["id"]], "safety_checker")
        run = read_json(root / "run.json")
        path = root / run["skips"][0]["annotation"]
        atomic_write(path, path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            verified_outcomes(root, run, self.photos, complete_records(root, self.photos, run["run_id"]))

    def test_current_run_rejects_changed_config(self):
        run_id = "synthetic_identity_test"
        run = {"config": self.config, "data_identity": data_identity(self.config), "generation_code_sha": generation_code_sha()}
        write_json(Path(self.temporary.name) / "results" / run_id / "run.json", run)
        write_json(Path(self.temporary.name) / "results/current_run.json", {"run_id": run_id})
        self.assertEqual(current_run(self.config)[0].name, run_id)
        changed = copy.deepcopy(self.config); changed["guidance_scale"] = 8.0
        with self.assertRaises(ValueError):
            current_run(changed)


class NotebookAndReportTests(unittest.TestCase):
    def test_packaged_report_font_has_cyrillic_without_os_fonts(self):
        from matplotlib import get_data_path
        from reportlab.pdfbase import pdfmetrics
        from src.report import register_fonts
        with patch.dict(os.environ, {"REPORT_FONT_PATH": ""}):
            path = register_fonts()
        self.assertTrue(path.resolve().is_relative_to(Path(get_data_path()).resolve()))
        for name in ("Body", "BodyBold"):
            glyphs = pdfmetrics.getFont(name).face.charToGlyph
            for character in "ПриветЁёжйα":
                self.assertIn(ord(character), glyphs)

    def test_one_notebook_and_valid_python_cells(self):
        notebooks = list(REPO_ROOT.glob("*.ipynb"))
        self.assertEqual([path.name for path in notebooks], ["colab_stage1.ipynb"])
        notebook = json.loads(notebooks[0].read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), "colab_cell", "exec")

    def test_final_report_refuses_absent_experiments(self):
        from src.report import checked_materials
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DIFFUSION_STYLE_HOME": directory}):
            with self.assertRaises((ValueError, FileNotFoundError)):
                checked_materials(load_config(), draft=False)
            self.assertFalse((Path(directory) / "reports/stage1_report.pdf").exists())

    def test_draft_pdf_pages_and_cyrillic(self):
        import pymupdf
        from src.report import build_report
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DIFFUSION_STYLE_HOME": directory}):
            destination = build_report(load_config(), draft=True)
            self.assertEqual(destination.name, "stage1_report_draft.pdf")
            with pymupdf.open(destination) as pdf:
                self.assertTrue(4 <= len(pdf) <= 6)
                self.assertIn("Диффузионные", pdf[0].get_text())
                self.assertTrue(all("ЧЕРНОВИК" in page.get_text() for page in pdf))
            self.assertFalse((destination.parent / "stage1_report.pdf").exists())


if __name__ == "__main__":
    unittest.main()
