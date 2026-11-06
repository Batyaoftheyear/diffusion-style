"""Проверить готовые результаты и явно отметить пропущенные фото без GPU."""
from _common import load_config, parser
from src.run_status import finalize_run


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--skip-photo", action="append", default=[])
    arguments.add_argument("--reason", help="Explicit reason, e.g. safety_checker")
    args = arguments.parse_args()
    finalize_run(load_config(args.config), args.skip_photo, args.reason)


if __name__ == "__main__":
    main()
