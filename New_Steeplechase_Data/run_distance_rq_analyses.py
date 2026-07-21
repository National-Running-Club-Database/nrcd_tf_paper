#!/usr/bin/env python3
"""Run RQ1 → RQ1B → RQ1C for new_steeplechase_data distance analyses."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n===== Running {script} =====\n")
    subprocess.check_call([sys.executable, str(ROOT / script)], cwd=str(ROOT))


def main() -> None:
    run("prepare_distance_data.py")
    run("analyze_rq1_best_event_distance.py")
    run("analyze_rq1b_nationals_distance.py")
    run("analyze_rq1c_competitiveness_distance.py")
    print(f"\nAll distance RQ analyses complete under {ROOT}")


if __name__ == "__main__":
    main()
