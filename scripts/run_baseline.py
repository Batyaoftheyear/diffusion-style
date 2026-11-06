"""Реальный GPU baseline; --limit 2 запускает smoke test."""
from _common import load_config, parser
from src.inference import run_baseline


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--limit", type=int, help="Use 2–4 images for the initial GPU smoke test")
    args = arguments.parse_args()
    run_baseline(load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
