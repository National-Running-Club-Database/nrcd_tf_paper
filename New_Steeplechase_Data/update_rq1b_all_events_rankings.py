"""Rebuild rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt with ALL event groups.

Uses Relays_Findings RQ1B eighth-place WA for every event except 3000m Steeplechase,
which is taken from New_Steeplechase_Data RQ1B (corrected steeplechase scoring).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "RQ1B_Nationals"
LEGACY_TOP8 = (
    ROOT.parent / "Relays_Findings" / "RQ1B_Nationals" / "rq1b_nationals_top8_all_seasons.csv"
)
NEW_TOP8 = OUTPUT / "rq1b_nationals_top8_all_seasons.csv"
SEASONS = ("2024", "2025", "2026")


def load_eighths(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["place"]) != 8:
                continue
            wa = float(row["world_athletics_points"])
            if wa <= 0:
                continue
            out[(row["gender"], row["event_name"])][row["season"]] = wa
    return out


def main() -> None:
    eighth = load_eighths(LEGACY_TOP8)
    new_eighth = load_eighths(NEW_TOP8)
    # Prefer corrected steeplechase; keep other distance events consistent with legacy
    # (they match) but overwrite steeple from new analysis.
    for key, yearly in new_eighth.items():
        if key[1] == "3000m Steeplechase":
            eighth[key] = yearly

    lines = [
        "Nationals Top-8 Eighth-Place World Athletics Points — 3-Year Average Rankings",
        "(All event groups; relay-inclusive. Steeplechase WA from New_Steeplechase_Data;",
        " other events from Relays_Findings RQ1B. Average of available 2024–2026 8th-place WA.)",
        "",
    ]
    thresholds: dict[tuple[str, str], float] = {}

    for gender in ("Men", "Women"):
        lines.append("=" * 60)
        lines.append(f"{gender} — 3-Year Average (2024–2026)")
        lines.append("=" * 60)
        event_avgs: list[tuple[str, float, dict[str, float]]] = []
        events = sorted({ev for g, ev in eighth if g == gender})
        for event in events:
            yearly = {
                y: round(eighth[(gender, event)][y], 1)
                for y in SEASONS
                if y in eighth[(gender, event)]
            }
            if not yearly:
                continue
            avg = round(sum(yearly.values()) / len(yearly), 1)
            thresholds[(gender, event)] = avg
            event_avgs.append((event, avg, yearly))
        for rank, (event, avg, yearly) in enumerate(
            sorted(event_avgs, key=lambda x: (-x[1], x[0])), 1
        ):
            yr_str = ", ".join(f"{y}={yearly[y]}" for y in sorted(yearly))
            lines.append(
                f"  {rank:>2}. {event:<24} avg 8th-place WA: {avg:>6.1f}   ({yr_str})"
            )
        lines.append("")

    out_path = OUTPUT / "rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt"
    out_path.write_text("\n".join(lines).rstrip() + "\n")

    # Also refresh all-event thresholds JSON used for cross-discipline competitiveness
    (OUTPUT / "rq1c_thresholds_all_events.json").write_text(
        json.dumps({f"{g}|{e}": v for (g, e), v in thresholds.items()}, indent=2) + "\n"
    )

    # Year-by-year all events
    by_year = [
        "Nationals 8th-Place WA by Year and Gender (All Event Groups)",
        "Steeplechase from New_Steeplechase_Data; other events from Relays_Findings.",
        "",
    ]
    for year in SEASONS:
        by_year.append(f"## {year}")
        for gender in ("Men", "Women"):
            by_year.append(f"### {gender}")
            ranked = [
                (ev, eighth[(gender, ev)][year])
                for (g, ev) in eighth
                if g == gender and year in eighth[(gender, ev)]
            ]
            for event, val in sorted(ranked, key=lambda x: -x[1]):
                by_year.append(f"  {event}: {val:.1f}")
            by_year.append("")
    (OUTPUT / "rq1b_nationals_8th_place_rankings_by_year_gender.txt").write_text(
        "\n".join(by_year).rstrip() + "\n"
    )

    print(f"Wrote {out_path}")
    print(f"Steeple Men avg={thresholds.get(('Men','3000m Steeplechase'))}")
    print(f"Steeple Women avg={thresholds.get(('Women','3000m Steeplechase'))}")
    print(f"Total ranked events: {len(thresholds)}")


if __name__ == "__main__":
    main()
