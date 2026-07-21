"""CLI entrypoint for the non_relays_findings module (RQ1 individual-event analysis).

Usage:
  python main.py                # run all four top-level RQ1 analyses
  python main.py rq1_improved   # RQ1A/B/C with bootstrap CIs, chi-square, Mann-Whitney, etc.
  python main.py rq1b_nationals # nationals top-8 distributions and 8th-place thresholds
  python main.py rq1c           # most-competitive-event recommendation (RQ1C)
  python main.py rq3            # population-level WA point dispersion (IQR)
  python main.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "rq1_improved": "analyze_rq1_improved.py",
    "rq1b_nationals": "analyze_rq1b_nationals.py",
    "rq1c": "analyze_rq1c_competitiveness.py",
    "rq3": "analyze_rq3_population_iqr.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RQ1 (specialization/competitiveness) and RQ3 (population "
        "dispersion) analyses for individual (non-relay) events."
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
