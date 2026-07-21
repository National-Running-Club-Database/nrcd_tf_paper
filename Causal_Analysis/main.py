"""CLI entrypoint for the causal_analysis module.

Usage:
  python main.py              # run the causal competition-volume analysis
  python main.py causal       # FE/OLS/delta models + research_stats.py (chained)
  python main.py research     # re-run only the inferential-stats layer (research/)
  python main.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "causal": "analyze_causal_competition_volume.py",
    "research": "research_stats.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the within-athlete causal analysis of competition volume "
        "vs seasonal World Athletics (WA) point improvement."
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
    # "causal" already chains research_stats.py at the end of its own main(), so
    # "all" only needs to invoke it once (running "research" again would just
    # recompute the same inferential outputs).
    to_run = ["causal"] if args.command == "all" else [args.command]
    for key in to_run:
        run_script(COMMANDS[key])


if __name__ == "__main__":
    main()
