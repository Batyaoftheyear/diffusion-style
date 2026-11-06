"""Загрузка, подготовка и проверка данных."""
import json
from _common import load_config, parser
from src.data import prepare_data, validate_dataset
from src.visualization import data_preview


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--check-only", action="store_true")
    args = arguments.parse_args()
    config = load_config(args.config)
    summary = validate_dataset(config) if args.check_only else prepare_data(config)
    if not args.check_only:
        summary["preview"] = str(data_preview(config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
