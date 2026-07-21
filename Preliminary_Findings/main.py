"""CLI entrypoint for the preliminary_findings module.

Document-only early-stage snapshot; superseded by dedicated modules elsewhere.
No analysis scripts to run.

Usage:
  python main.py
  python main.py all
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

SUCCESSOR_MAP = {
    "NIRCA_Nationals_Strategy/": "../relays_findings/RQ1B_Nationals/, ../new_steeplechase_data/RQ1B_Nationals/",
    "Number_Of_Events/": "../number_of_events_question/, ../causal_analysis/",
    "Pairwise_Events/": "../non_relays_findings/*_Events_Counting/, ../indoor_analysis/RQ1A_Best_Event/",
    "Indoor_Outdoor_Interplay/": "../indoor_analysis/Indoor_Outdoor_Interplay/",
    "Time_Models/": "../time_models/, ../indoor_analysis/Feature_Importance_Point_Band_Time_Models/",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="preliminary_findings is document-only (superseded snapshot)."
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
    print("preliminary_findings is a document-only, early-stage snapshot module.")
    print("Its subfolders have been superseded by dedicated modules:\n")
    for subfolder, successor in SUCCESSOR_MAP.items():
        marker = "found" if (MODULE_ROOT / subfolder).exists() else "missing"
        print(f"  - {subfolder} ({marker}) -> see {successor}")
    print("\nSee README.md and Summary_findings.md for details.")


if __name__ == "__main__":
    main()
