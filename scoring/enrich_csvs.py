"""Enrich discipline CSVs with VDOT, Purdy, and Mercier columns.

Keeps existing World_Athletics_Points_* columns unchanged (legacy NRCD values).
Adds recomputed WA verification columns optionally; by default only appends
new metric columns.

Usage:
  python -m scoring.enrich_csvs
  python -m scoring.enrich_csvs --dry-run
  python main.py enrich-scores
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring.api import score
from scoring.columns import points_col
from scoring.marks import RUNNING_EVENT_ID_TO_CANONICAL, canonicalize_event

# Modules / folders that hold performance CSVs with result_time + running_event_id
SEARCH_ROOTS = [
    ROOT / "non_relays_findings",
    ROOT / "relays_findings",
    ROOT / "indoor_analysis",
    ROOT / "new_steeplechase_data",
    ROOT / "time_models",
    ROOT / "number_of_events_question",
]

NEW_SUFFIXES = (
    "VDOT_Men",
    "VDOT_Women",
    "Purdy_Points_Men",
    "Purdy_Points_Women",
    "Mercier_Points_Men",
    "Mercier_Points_Women",
)


def _gender_from_row(row: dict, path: Path) -> str | None:
    g = (row.get("gender") or "").strip()
    if g in {"M", "Men", "men", "male"}:
        return "men"
    if g in {"W", "Women", "women", "female", "F"}:
        return "women"
    name = path.name.lower()
    if "_men_" in name or name.startswith("men"):
        return "men"
    if "_women_" in name or name.startswith("women"):
        return "women"
    return None


def _is_target_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    # Skip non-result tables
    skip_bits = (
        "running_event",
        "summary",
        "cohort",
        "pair_results",
        "feature_",
        "predictions",
        "validation",
        "by_metric",
        "dose_response",
        "fe_regression",
        "within_",
        "opener_wa",
        "best_event",
        "point_band",
        "model_",
        "compare_",
        "research",
    )
    low = path.name.lower()
    if any(b in low for b in skip_bits):
        return False
    return "data" in low or "outdoor" in low or "indoor" in low


def discover_csvs() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.csv"):
            if _is_target_csv(path):
                found.append(path)
    return sorted(found)


def enrich_file(path: Path, *, dry_run: bool = False) -> dict:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {"path": str(path), "status": "empty"}
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required = {"result_time", "running_event_id"}
    if not required.issubset(set(fieldnames)):
        return {"path": str(path), "status": "skip_no_mark_cols"}

    # Ensure new columns exist
    for col in NEW_SUFFIXES:
        if col not in fieldnames:
            fieldnames.append(col)

    wa_legacy = 0
    wa_recomputed = 0
    wa_abs_diff = 0.0
    n_scored = 0
    n_rows = len(rows)

    for row in rows:
        gender = _gender_from_row(row, path)
        if gender is None:
            continue
        event_raw = row.get("running_event_id")
        try:
            event_id = int(float(event_raw)) if event_raw not in (None, "") else None
        except (TypeError, ValueError):
            event_id = None
        if event_id is None or event_id not in RUNNING_EVENT_ID_TO_CANONICAL:
            # Still clear new cols for consistency
            for col in NEW_SUFFIXES:
                row.setdefault(col, "")
            continue

        metrics = score(row.get("result_time"), event_id, gender)
        n_scored += 1

        # Write both genders' columns; fill the athlete gender, blank the other
        for metric_key, col_men, col_women in (
            ("vdot", "VDOT_Men", "VDOT_Women"),
            ("purdy", "Purdy_Points_Men", "Purdy_Points_Women"),
            ("mercier", "Mercier_Points_Men", "Mercier_Points_Women"),
        ):
            val = metrics.get(metric_key)
            if gender == "men":
                row[col_men] = "" if val is None else val
                row[col_women] = row.get(col_women, "") or ""
            else:
                row[col_women] = "" if val is None else val
                row[col_men] = row.get(col_men, "") or ""

        # WA verification vs legacy (do not overwrite)
        legacy_col = points_col(gender, "wa")
        legacy = row.get(legacy_col)
        try:
            legacy_f = float(legacy) if legacy not in (None, "", "nan") else None
        except ValueError:
            legacy_f = None
        if legacy_f is not None and metrics["wa"] is not None:
            wa_legacy += 1
            wa_recomputed += 1
            wa_abs_diff += abs(legacy_f - float(metrics["wa"]))

    if not dry_run:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    mean_diff = (wa_abs_diff / wa_legacy) if wa_legacy else None
    return {
        "path": str(path.relative_to(ROOT)),
        "status": "ok",
        "rows": n_rows,
        "scored": n_scored,
        "wa_compared": wa_legacy,
        "wa_mean_abs_diff": round(mean_diff, 3) if mean_diff is not None else None,
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich CSVs with multi-metric scores.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    args = parser.parse_args(argv)

    paths = discover_csvs()
    if args.limit:
        paths = paths[: args.limit]

    print(f"Enriching {len(paths)} CSV files (dry_run={args.dry_run})")
    ok = 0
    skipped = 0
    reports = []
    for path in paths:
        rep = enrich_file(path, dry_run=args.dry_run)
        reports.append(rep)
        if rep["status"] == "ok":
            ok += 1
            mad = rep.get("wa_mean_abs_diff")
            mad_s = f" WAΔ={mad}" if mad is not None else ""
            print(f"  ✓ {rep['path']}  scored={rep['scored']}/{rep['rows']}{mad_s}")
        else:
            skipped += 1

    out_report = ROOT / "scoring" / "data" / "enrichment_report.csv"
    if not args.dry_run:
        with open(out_report, "w", newline="", encoding="utf-8") as f:
            fields = [
                "path",
                "status",
                "rows",
                "scored",
                "wa_compared",
                "wa_mean_abs_diff",
            ]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in reports:
                w.writerow(r)
        print(f"Wrote {out_report.relative_to(ROOT)}")

    print(f"Done. ok={ok} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
