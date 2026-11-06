"""Opt-in test настоящего пайплайна, без моков и без скачивания моделей в CPU-тестах."""
import os
import unittest

from src.config import load_config
from src.inference import run_baseline
from src.io import read_json


@unittest.skipUnless(os.environ.get("RUN_GPU_SMOKE") == "1", "requires prepared data and a real Colab CUDA GPU")
class GpuSmokeTest(unittest.TestCase):
    def test_real_ip_adapter_and_controlnet(self):
        root = run_baseline(load_config(), limit=2)
        run = read_json(root / "run.json")
        self.assertGreaterEqual(run["completed_images"], 2)
        self.assertIsNotNone(run["environment"]["cuda"])
        self.assertTrue(run["environment"]["gpu"])
        self.assertTrue((root / "figures/comparisons.png").is_file())


if __name__ == "__main__":
    unittest.main()
