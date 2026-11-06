"""Сборка PDF с проверкой данных и числа страниц."""
from _common import load_config, parser
from src.report import build_report


def main():
    arguments = parser(__doc__)
    arguments.add_argument("--draft", action="store_true", help="Create an explicitly unfinished report")
    arguments.add_argument("--render", action="store_true", help="Render PDF pages to PNG for inspection")
    args = arguments.parse_args()
    build_report(load_config(args.config), draft=args.draft, render=args.render)


if __name__ == "__main__":
    main()
