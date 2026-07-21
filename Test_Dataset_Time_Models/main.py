"""CLI entrypoint for the men's time-model test-dataset pipeline.

Usage:
  python main.py              # preprocess → validate → research stats
  python main.py preprocess
  python main.py validate
  python main.py research
  python main.py all
"""

from __future__ import annotations

import argparse

from preprocess import run_preprocess
from research_stats import run_research_stats
from validate import run_validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the men's test dataset, validate outdoor time models, "
        "and run research-grade statistical tests."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("preprocess", "validate", "research", "all"),
        help="Pipeline step to run (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command in ("preprocess", "all"):
        run_preprocess()
    if args.command in ("validate", "all"):
        # validate already invokes research_stats at the end
        run_validate()
    elif args.command == "research":
        run_research_stats()


if __name__ == "__main__":
    main()
