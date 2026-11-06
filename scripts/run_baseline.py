"""Реальный GPU baseline; --limit 2 запускает smoke test."""
from _common import load_config, parser
from src.inference import run_baseline
from src.run_status import reuse_finalized_run


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--limit", type=int, help="Use 2–4 images for the initial GPU smoke test")
    args = arguments.parse_args()
    config = load_config(args.config)
    if not reuse_finalized_run(config):
        run_baseline(config, limit=args.limit)


if __name__ == "__main__":
    main()
