"""CLI for Indoor_Outdoor_Calendar_Interplay.

Usage:
  python main.py           # combined indoor + pre-nats outdoor calendar
  python main.py post_nats # continuation after April nationals
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
        choices=("all", "run", "post_nats"),
    )
    args = parser.parse_args(argv)
    if args.command == "post_nats":
        script = HERE / "Post_Nationals_Continuation" / "main.py"
        raise SystemExit(subprocess.call([sys.executable, str(script)]))
    from analyze_interplay import main as interplay_main

    interplay_main(["all"])


if __name__ == "__main__":
    main()
