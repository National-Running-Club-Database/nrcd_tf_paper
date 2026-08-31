"""CLI for Meet_Calendar_Analysis.

Usage:
  python main.py              # outdoor-only April nationals calendar
  python main.py interplay    # indoor + outdoor combined calendar
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Meet calendar optimization")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "run", "interplay"),
    )
    args = parser.parse_args(argv)
    if args.command == "interplay":
        script = HERE / "Indoor_Outdoor_Calendar_Interplay" / "main.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))
    from analyze_meet_calendar import main as outdoor_main

    outdoor_main(["all"])


if __name__ == "__main__":
    main()
