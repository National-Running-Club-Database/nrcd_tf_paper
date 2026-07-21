"""CLI entrypoint for the time_models module (root-level build scripts).

This module's real content lives in self-contained subfolders (Point_Bands_Time_Models/,
specialized_time_models/, Short_Long_Specialization_Time_Models/, new_factors/,
Combined_Specialization_Time_Models/, six_to_ten_races_time_models/, model_search/), each
with its own analyze_*.py / build_*.py entrypoints — see README.md for the full list.
main.py wires the two root-level build scripts plus the inferential research layer.

Usage:
  python main.py                     # run both root-level build scripts + research
  python main.py cross_event         # build_cross_event_time_models.py
  python main.py higher_races_report # generate_higher_races_formula_reports.py
  python main.py research            # inferential layer on FI CV tables (research/)
  python main.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "cross_event": "build_cross_event_time_models.py",
    "higher_races_report": "generate_higher_races_formula_reports.py",
    "research": "research_stats.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the root-level time_models build scripts and research "
        "stats. See README.md for subfolder pipelines not wired here."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=(*COMMANDS.keys(), "all"),
        help="Script to run (default: all).",
    )
    return parser


def run_script(script_name: str) -> None:
    script_path = MODULE_ROOT / script_name
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(script_path)], cwd=MODULE_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    to_run = COMMANDS.keys() if args.command == "all" else [args.command]
    for key in to_run:
        run_script(COMMANDS[key])


if __name__ == "__main__":
    main()
