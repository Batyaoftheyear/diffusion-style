"""Расчёт метрик полного baseline."""
import json
from _common import load_config, parser
from src.metrics import evaluate


def main():
    args = parser(__doc__).parse_args()
    summary = evaluate(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
