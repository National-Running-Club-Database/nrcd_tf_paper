"""CLI entrypoint for the new_steeplechase_data module.

This module re-runs the Distance RQ1/RQ1B/RQ1C pipeline with corrected 3000m
Steeplechase World Athletics (WA) scoring (see DATA_NOTES.txt), and rebuilds the
point-band time models with steeplechase included.

Usage:
  python main.py                  # full distance RQ pipeline (default: all)
  python main.py distance_rq       # prepare_distance_data -> RQ1 -> RQ1B -> RQ1C
  python main.py rankings           # update_rq1b_all_events_rankings.py (legacy vs corrected)
  python main.py feature_importance # analyze_feature_importance_with_steeple.py
  python main.py research           # inferential-stats layer (research/) on top of RQ1/RQ1B
  python main.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "distance_rq": "run_distance_rq_analyses.py",
    "rankings": "update_rq1b_all_events_rankings.py",
    "feature_importance": "Feature_Importance_Point_Band_Time_Models/analyze_feature_importance_with_steeple.py",
    "research": "research_stats.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the steeplechase-corrected distance RQ pipeline, the "
        "point-band feature-importance models, and the inferential-stats research layer."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=(*COMMANDS.keys(), "all"),
        help="Step to run (default: all).",
    )
    return parser


def run_script(script_rel_path: str) -> None:
    script_path = MODULE_ROOT / script_rel_path
    if not script_path.exists():
        print(f"Skipping {script_rel_path}: not found at {script_path}")
        return
    print(f"\n=== Running {script_rel_path} ===")
    subprocess.run([sys.executable, str(script_path)], cwd=script_path.parent, check=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    to_run = COMMANDS.keys() if args.command == "all" else [args.command]
    for key in to_run:
        run_script(COMMANDS[key])


if __name__ == "__main__":
    main()
