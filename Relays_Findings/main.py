"""CLI entrypoint for the relays_findings module (relay-inclusive RQ1 analysis).

Usage:
  python main.py                # run all top-level relay analyses + research layer
  python main.py rq1b_nationals # nationals top-8 distributions (individual + relay events)
  python main.py rq1c           # most-competitive-event recommendation, relay-inclusive
  python main.py pecking_orders # event pecking orders from pairwise best-event comparisons
  python main.py research       # inferential-stats layer (research/) on top of rq1b/rq1c
  python main.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "rq1b_nationals": "analyze_rq1b_nationals_relays.py",
    "rq1c": "analyze_rq1c_competitiveness_relays.py",
    "pecking_orders": "build_pecking_orders.py",
    "research": "research_stats.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the relay-inclusive RQ1B/RQ1C analyses, the event "
        "pecking-order builder, and the inferential-stats research layer for "
        "relays_findings."
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
    if not script_path.exists():
        print(f"Skipping {script_name}: not found at {script_path}")
        return
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(script_path)], cwd=MODULE_ROOT, check=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    to_run = COMMANDS.keys() if args.command == "all" else [args.command]
    for key in to_run:
        run_script(COMMANDS[key])


if __name__ == "__main__":
    main()
