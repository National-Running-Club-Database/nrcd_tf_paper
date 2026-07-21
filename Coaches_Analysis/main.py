"""CLI entrypoint for the coaches_analysis module.

Document-only: coaching-recommendation text reports synthesized from RQ1B/RQ1C.
No analysis scripts to run.

Usage:
  python main.py
  python main.py all
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

REPORTS = [
    "Nationals_Scoring_Focus_Men_Women.txt",
    "Nationals_Scoring_Focus_Clearing_Roster.txt",
    "Nationals_Points_Maximization_Clearing_Roster.txt",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="coaches_analysis is document-only (no analysis scripts to run)."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all",),
        help="Accepted for orchestrator compatibility (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    print("coaches_analysis is a document-only module — there are no scripts to run.")
    print("Pre-built coaching reports in this folder:")
    for name in REPORTS:
        path = MODULE_ROOT / name
        marker = "found" if path.exists() else "MISSING"
        print(f"  - {name} ({marker})")
    print("\nSee README.md and Summary_findings.md for a synthesis of these reports.")


if __name__ == "__main__":
    main()
