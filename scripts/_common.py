"""Общие аргументы независимых точек входа."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, load_config


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, default=REPO_ROOT / "configs/baseline.yaml")
    return result
