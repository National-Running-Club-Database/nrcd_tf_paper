"""Root CLI for the Cursor_AI_Track_Paper research workspace.

Usage:
  python main.py                  # list modules
  python main.py list
  python main.py run test_dataset_time_models
  python main.py run causal_analysis all
  python main.py enrich-scores    # append VDOT/Purdy/Mercier columns to CSVs
  python main.py compare-metrics  # RQ1A best-event agreement across 4 metrics
  python main.py run-all          # run every module's default `all` (long)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Display order follows the paper's analysis pipeline
MODULES: list[tuple[str, str]] = [
    ("non_relays_findings", "Outdoor RQ1A/B/C + RQ3 (individual events)"),
    ("relays_findings", "Outdoor RQ1 with relays + pecking orders"),
    ("new_steeplechase_data", "Canonical distance/steeple WA re-run"),
    ("indoor_analysis", "Indoor RQ1A + point-band/FI + indoor↔outdoor"),
    ("number_of_events_question", "Competition volume vs WA point jump"),
    ("causal_analysis", "Within-athlete FE dose–response for volume"),
    ("time_models", "Cross-event time models / point bands / FI"),
    ("test_dataset_time_models", "External D1 validation of pair models"),
    ("Meet_Calendar_Analysis", "Outdoor meet budget → nationals margin policies"),
    ("coaches_analysis", "Coach-facing nationals strategy (docs)"),
    ("preliminary_findings", "Curated presentation copies (docs)"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate NRCD track specialization analyses for the JQAS paper."
    )
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("list", help="List analysis modules (default).")

    run_p = sub.add_parser("run", help="Run one module's main.py")
    run_p.add_argument("module", choices=[m for m, _ in MODULES])
    run_p.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the module main.py (default: all).",
    )

    sub.add_parser(
        "run-all",
        help="Run every module's `all` command sequentially (can take a long time).",
    )
    sub.add_parser(
        "enrich-scores",
        help="Append VDOT / Purdy / Mercier columns to discipline CSVs.",
    )
    sub.add_parser(
        "compare-metrics",
        help="Cross-metric best-event comparison (scientific vs sports framing).",
    )
    return parser


def list_modules() -> None:
    print("Cursor_AI_Track_Paper — analysis modules\n")
    for name, desc in MODULES:
        main_py = ROOT / name / "main.py"
        status = "ready" if main_py.is_file() else "missing main.py"
        print(f"  {name:32s}  [{status}]  {desc}")
    print("\nScoring:")
    print("  Scientific framing: Gardner–Purdy + Mercier (1999)")
    print("  Sports / coaching:  World Athletics Points + VDOT")
    print("\nExamples:")
    print("  python main.py run test_dataset_time_models")
    print("  python main.py run causal_analysis research")
    print("  python main.py enrich-scores")
    print("  python main.py compare-metrics")
    print("  python main.py run non_relays_findings all")
    print("\nSee README.md and Summary_findings.md at the repo root.")


def run_module(module: str, module_args: list[str]) -> int:
    main_py = ROOT / module / "main.py"
    if not main_py.is_file():
        print(f"No main.py in {module}", file=sys.stderr)
        return 1
    args = module_args if module_args else ["all"]
    # Reminder absorbs leading '--' sometimes
    if args and args[0] == "--":
        args = args[1:] or ["all"]
    cmd = [sys.executable, str(main_py), *args]
    print(f"→ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(main_py.parent))


def run_all() -> int:
    rc = 0
    for name, _ in MODULES:
        code = run_module(name, ["all"])
        if code != 0:
            print(f"Module {name} exited with {code}", file=sys.stderr)
            rc = code
    return rc


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    action = args.action or "list"
    if action == "list":
        list_modules()
        return
    if action == "run":
        raise SystemExit(run_module(args.module, list(args.module_args or [])))
    if action == "run-all":
        raise SystemExit(run_all())
    if action == "enrich-scores":
        from scoring.enrich_csvs import main as enrich_main

        raise SystemExit(enrich_main([]))
    if action == "compare-metrics":
        from scoring.compare_metrics import main as compare_main

        raise SystemExit(compare_main([]))
    parser.print_help()


if __name__ == "__main__":
    main()
