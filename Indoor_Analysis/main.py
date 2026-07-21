"""CLI entrypoint for the indoor_analysis module.

Usage:
  python main.py               # run both top-level indoor analyses
  python main.py rq1a          # indoor best-event specialization (RQ1A)
  python main.py point_bands   # indoor point-band time-model width/placement search
  python main.py research      # inferential-stats layer (research/) on top of rq1a + opener WA
  python main.py all           # rq1a + point_bands + research
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "rq1a": "analyze_indoor_rq1a.py",
    "point_bands": "analyze_indoor_point_bands.py",
    "research": "research_stats.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the top-level indoor track analyses: RQ1A specialization, "
        "indoor point-band time-model search, and the inferential-stats research layer."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=(*COMMANDS.keys(), "all"),
        help="Analysis to run (default: all).",
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
