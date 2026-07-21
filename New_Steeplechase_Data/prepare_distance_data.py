"""Prepare New_Steeplechase_Data distance CSVs for RQ1/RQ1B/RQ1C.

Validates gender labels on Relays_Distance_* files. Women's files in
New_Steeplechase_Data currently appear to be men's datasets; when that
happens, fall back to Relays_Findings/Distance_Relays_Findings women CSVs
and document the issue.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LEGACY_DIST = PROJECT / "Relays_Findings" / "Distance_Relays_Findings"
OUT_DIST = ROOT / "Distance_Relays_Findings"

SEASONS = ("2024", "2025", "2026")
GENDERS = ("Men", "Women")
EXPECTED_GENDER = {"Men": {"M", "MALE", "MEN"}, "Women": {"F", "W", "FEMALE", "WOMEN"}}


def csv_name(gender: str, year: str) -> str:
    return f"Relays_Distance_{gender}_Outdoor_{year}_Data.csv"


def gender_match_rate(path: Path, gender: str) -> float:
    expected = EXPECTED_GENDER[gender]
    n = bad = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n += 1
            g = (row.get("gender") or "").strip().upper()
            if g not in expected:
                bad += 1
    return 0.0 if n == 0 else 1.0 - bad / n


def steeple_point_summary(path: Path, gender: str) -> str:
    col = (
        "World_Athletics_Points_Men"
        if gender == "Men"
        else "World_Athletics_Points_Women"
    )
    pts = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row.get("running_event_id") or -1)) != 20:
                continue
            pts.append(float(row.get(col) or 0))
    if not pts:
        return "no steeple rows"
    nz = sum(1 for p in pts if p > 0)
    return (
        f"n={len(pts)}, nonzero={nz}, mean={sum(pts)/len(pts):.1f}, max={max(pts):.1f}"
    )


def main() -> None:
    OUT_DIST.mkdir(parents=True, exist_ok=True)
    notes = [
        "New_Steeplechase_Data — Distance dataset notes",
        "==============================================",
        "",
        "Source CSVs live in New_Steeplechase_Data/ (Relays_Distance_*).",
        "Working copies for analysis are placed in Distance_Relays_Findings/.",
        "",
    ]
    for gender in GENDERS:
        for year in SEASONS:
            name = csv_name(gender, year)
            src_new = ROOT / name
            src_old = LEGACY_DIST / name
            dest = OUT_DIST / name
            if not src_new.exists():
                notes.append(f"[MISSING] {name} in New_Steeplechase_Data")
                if src_old.exists():
                    shutil.copy2(src_old, dest)
                    notes.append(f"  → copied legacy {name}")
                continue
            rate = gender_match_rate(src_new, gender)
            steeple = steeple_point_summary(src_new, gender)
            if rate < 0.5:
                notes.append(
                    f"[GENDER MISMATCH] {name}: only {100*rate:.1f}% rows match "
                    f"expected gender={gender}. Steeple summary: {steeple}."
                )
                notes.append(
                    "  → This file looks like the opposite gender’s dataset. "
                    "Using Relays_Findings legacy women/men file instead for this slot."
                )
                if src_old.exists():
                    shutil.copy2(src_old, dest)
                    notes.append(
                        f"  → fell back to {src_old.relative_to(PROJECT)} "
                        f"(steeple: {steeple_point_summary(src_old, gender)})"
                    )
                else:
                    notes.append("  → ERROR: no legacy fallback available")
            else:
                shutil.copy2(src_new, dest)
                notes.append(
                    f"[OK] {name} gender match {100*rate:.1f}%; steeple: {steeple}"
                )
                if src_old.exists():
                    # compare steeple means
                    old_s = steeple_point_summary(src_old, gender)
                    notes.append(f"  legacy steeple for comparison: {old_s}")

    notes.extend(
        [
            "",
            "If corrected Women Relays_Distance_* CSVs with female gender labels and",
            "resored steeple WA points become available, replace the files in",
            "New_Steeplechase_Data/ and re-run prepare_distance_data.py + the RQ scripts.",
            "",
        ]
    )
    (ROOT / "DATA_NOTES.txt").write_text("\n".join(notes).rstrip() + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
