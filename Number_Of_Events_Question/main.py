"""CLI entrypoint for the number_of_events_question module.

Usage:
  python main.py                 # run both analyses (default: all)
  python main.py point_jump      # point jump vs. competition volume (chains research_stats.py)
  python main.py best_event      # best-event (season-peak) timing
  python main.py research        # re-run only the inferential-stats layer (research/)
  python main.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "point_jump": "analyze_point_jump_by_competition_count.py",
    "best_event": "analyze_best_event_proportion.py",
    "research": "research_stats.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 'does racing more help / when do athletes peak' analyses."
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
    # "point_jump" already chains research_stats.py at the end of its own main(),
    # so "all" doesn't need to invoke "research" separately.
    to_run = ["point_jump", "best_event"] if args.command == "all" else [args.command]
    for key in to_run:
        run_script(COMMANDS[key])


if __name__ == "__main__":
    main()
