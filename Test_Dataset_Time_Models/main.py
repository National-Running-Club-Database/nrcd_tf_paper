"""CLI entrypoint for the men's time-model test-dataset pipeline.

Usage:
  python main.py              # preprocess → validate → research stats
  python main.py preprocess
  python main.py validate
  python main.py research
  python main.py d1_only         # NCAA D1 schools only → D1_Only_* artifacts
  python main.py all_divisions   # all docs (D1/D3/NAIA) → output/All_Divisions/
  python main.py all_divisions_2plus  # All_Divisions season-PB with ≥2 results in A and B
  python main.py all_divisions_2plus_from  # ≥2 in A only (B unrestricted)
  python main.py leakage         # quantify feature/protocol leakage on test outputs
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
        choices=(
            "preprocess",
            "validate",
            "research",
            "d1_only",
            "all_divisions",
            "all_divisions_2plus",
            "all_divisions_2plus_from",
            "leakage",
            "all",
        ),
        help="Pipeline step to run (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "d1_only":
        from run_d1_only import main as d1_main

        d1_main(["all"])
        return
    if args.command == "all_divisions":
        from run_all_divisions import main as all_div_main

        all_div_main(["all"])
        return
    if args.command == "all_divisions_2plus":
        from run_all_divisions import main as all_div_main

        all_div_main(["2plus"])
        return
    if args.command == "all_divisions_2plus_from":
        from run_all_divisions import main as all_div_main

        all_div_main(["2plus_from"])
        return
    if args.command == "leakage":
        from analyze_test_data_leakage import main as leakage_main

        leakage_main(["--dataset", "both"])
        return
    if args.command in ("preprocess", "all"):
        run_preprocess()
    if args.command in ("validate", "all"):
        # validate already invokes research_stats at the end
        run_validate()
    elif args.command == "research":
        run_research_stats()


if __name__ == "__main__":
    main()
