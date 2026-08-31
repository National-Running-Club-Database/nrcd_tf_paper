"""CLI for Indoor_to_Outdoor_Predictions.

Usage:
  python main.py              # baseline + feature-aware reports
  python main.py baseline     # linear indoor→outdoor models only
  python main.py feature      # feature-aware formula reports only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "baseline", "feature"),
    )
    args = parser.parse_args(argv)
    if args.command in ("baseline", "all"):
        from analyze_indoor_to_outdoor import main as baseline_main

        baseline_main()
    if args.command in ("feature", "all"):
        subprocess.run(
            [sys.executable, str(HERE / "generate_feature_aware_report.py")],
            cwd=str(HERE),
            check=True,
        )


if __name__ == "__main__":
    main()
