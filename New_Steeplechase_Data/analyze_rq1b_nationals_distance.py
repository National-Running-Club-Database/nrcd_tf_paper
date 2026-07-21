"""RQ1B: Nationals top-8 WA distributions — Distance only (new steeplechase data)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

RELAYS_FINDINGS = Path(__file__).resolve().parents[1] / "Relays_Findings"
sys.path.insert(0, str(RELAYS_FINDINGS))

from analyze_rq1b_nationals_relays import (  # noqa: E402
    dist_stats,
    filter_nationals_pool,
    rank_top8,
    write_season_distribution_plots,
)
from relay_rq1_data import (  # noqa: E402
    EVENT_MAP,
    FIELD_EVENT_IDS,
    SEASONS,
    STANDARD_RELAY_IDS,
    STANDARD_RELAY_MAP,
    parse_performance,
    points_col,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Distance_Relays_Findings"
OUTPUT = ROOT / "RQ1B_Nationals"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Distance individual + relays that appear in the distance relay-inclusive CSVs
DISTANCE_EVENT_IDS = {9, 11, 17, 20} | STANDARD_RELAY_IDS


def load_year(gender: str, year: str) -> list[dict]:
    path = DATA_DIR / f"Relays_Distance_{gender}_Outdoor_{year}_Data.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def event_jobs() -> list[tuple[str, int]]:
    """(event_name, event_id) for distance individuals then relays."""
    jobs = [
        (EVENT_MAP[eid], eid)
        for eid in (9, 11, 20, 17)  # 800, 1500, steeple, 5000
    ]
    for eid in sorted(STANDARD_RELAY_IDS):
        jobs.append((STANDARD_RELAY_MAP[eid], eid))
    return jobs


def analyze() -> None:
    all_top8_rows: list[dict] = []
    summary_lines_by_season: dict[str, list[str]] = {y: [] for y in SEASONS}
    eighth_by_key: dict[tuple[str, str, str], float] = {}

    for year in SEASONS:
        season_header = [
            f"RQ1B — Nationals Top-8 World Athletics Distributions ({year})",
            "[Distance only — New_Steeplechase_Data / corrected steeplechase WA]",
            "Ranking: Finals (or unlabeled nationals rows). Relays by team result_time.",
            "",
        ]
        summary_lines_by_season[year].extend(season_header)
        plot_groups: list[tuple] = []

        for gender in ("Men", "Women"):
            summary_lines_by_season[year].append(f"## {gender}")
            rows = load_year(gender, year)
            if not rows:
                summary_lines_by_season[year].append("  (no data)")
                continue
            pcol = points_col(gender)

            for event_name, event_id in event_jobs():
                season_event_rows = [
                    r for r in rows if int(r["running_event_id"]) == event_id
                ]
                if not season_event_rows:
                    continue
                season_points = [
                    float(r[pcol]) for r in season_event_rows if float(r[pcol]) > 0
                ]
                pool = filter_nationals_pool(rows, event_id)
                top8 = rank_top8(pool, event_id)
                nat_points = [float(r[pcol]) for r in top8 if float(r[pcol]) > 0]
                nat_stats = dist_stats(nat_points)
                season_stats = dist_stats(season_points)

                disc = "Relays" if event_id in STANDARD_RELAY_IDS else "Distance"
                summary_lines_by_season[year].append(
                    f"\n### {disc} — {gender} — {event_name}"
                )
                summary_lines_by_season[year].append(
                    f"Nationals pool: {len(pool)} | Season total: {len(season_event_rows)}"
                )
                summary_lines_by_season[year].append(
                    f"Nationals top-8 WA points — "
                    f"n={nat_stats['n']}, mean={nat_stats['mean']}, median={nat_stats['median']}, "
                    f"std={nat_stats['std']}, min={nat_stats['min']}, max={nat_stats['max']}"
                )
                summary_lines_by_season[year].append(
                    f"Full season WA points — "
                    f"n={season_stats['n']}, mean={season_stats['mean']}, "
                    f"median={season_stats['median']}, std={season_stats['std']}, "
                    f"min={season_stats['min']}, max={season_stats['max']}"
                )

                for place, r in enumerate(top8, 1):
                    wa = float(r[pcol])
                    summary_lines_by_season[year].append(
                        f"  {place}. {r['result_time']} | WA={wa:.0f} | athlete={r['athlete_id']}"
                    )
                    all_top8_rows.append(
                        {
                            "season": year,
                            "discipline": disc,
                            "gender": gender,
                            "event_id": event_id,
                            "event_name": event_name,
                            "place": place,
                            "result_time": r["result_time"],
                            "world_athletics_points": wa,
                            "athlete_id": r["athlete_id"],
                            "event_type_used": r["event_type"] or "(unlabeled)",
                            "meet_id": r["meet_id"],
                        }
                    )
                    if place == 8 and wa > 0:
                        eighth_by_key[(gender, event_name, year)] = wa

                if nat_points and season_points:
                    plot_groups.append((f"{gender} {event_name}", nat_points, season_points))

            summary_lines_by_season[year].append("")

        # Temporarily point plot writer at our OUTPUT
        import analyze_rq1b_nationals_relays as rq1b_mod

        old_out = rq1b_mod.OUTPUT
        rq1b_mod.OUTPUT = OUTPUT
        try:
            write_season_distribution_plots(plot_groups, year)
        finally:
            rq1b_mod.OUTPUT = old_out

    for year in SEASONS:
        (OUTPUT / f"rq1b_summary_{year}.txt").write_text(
            "\n".join(summary_lines_by_season[year]).rstrip() + "\n"
        )

    # CSV
    if all_top8_rows:
        path = OUTPUT / "rq1b_nationals_top8_all_seasons.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_top8_rows[0].keys()))
            w.writeheader()
            w.writerows(all_top8_rows)

    # Combined summary
    lines = [
        "RQ1B — Combined Nationals Top-8 Summary",
        "[Distance only — New_Steeplechase_Data]",
        "",
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_top8_rows:
        grouped[(r["season"], r["gender"], r["event_name"])].append(r)
    for key in sorted(grouped):
        season, gender, event = key
        pts = [
            r["world_athletics_points"]
            for r in grouped[key]
            if r["world_athletics_points"] > 0
        ]
        stats = dist_stats(pts)
        lines.append(
            f"{season} | {gender} | {event}: top-8 WA mean={stats['mean']}, "
            f"median={stats['median']}, range=[{stats['min']}, {stats['max']}]"
        )
    (OUTPUT / "rq1b_combined_top8_summary.txt").write_text("\n".join(lines) + "\n")

    # 8th-place rankings + thresholds
    rank_lines = [
        "Nationals Top-8 Eighth-Place World Athletics Points — 3-Year Average Rankings",
        "(Distance only — New_Steeplechase_Data; average of 2024–2026 8th-place WA)",
        "",
    ]
    thresholds: dict[tuple[str, str], float] = {}
    for gender in ("Men", "Women"):
        rank_lines.append("=" * 60)
        rank_lines.append(f"{gender} — 3-Year Average (2024–2026)")
        rank_lines.append("=" * 60)
        event_avgs = []
        events = sorted({ev for g, ev, _ in eighth_by_key if g == gender})
        for event in events:
            yearly = {}
            for year in SEASONS:
                val = eighth_by_key.get((gender, event, year))
                if val is not None:
                    yearly[year] = round(val, 1)
            if not yearly:
                continue
            avg = round(sum(yearly.values()) / len(yearly), 1)
            thresholds[(gender, event)] = avg
            event_avgs.append((event, avg, yearly))
        for rank, (event, avg, yearly) in enumerate(
            sorted(event_avgs, key=lambda x: (-x[1], x[0])), 1
        ):
            yr_str = ", ".join(f"{y}={yearly[y]}" for y in sorted(yearly))
            rank_lines.append(
                f"  {rank:>2}. {event:<24} avg 8th-place WA: {avg:>6.1f}   ({yr_str})"
            )
        rank_lines.append("")
    (OUTPUT / "rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt").write_text(
        "\n".join(rank_lines).rstrip() + "\n"
    )

    by_year_lines = [
        "Nationals 8th-Place WA by Year and Gender (Distance — New Steeplechase Data)",
        "",
    ]
    for year in SEASONS:
        by_year_lines.append(f"## {year}")
        for gender in ("Men", "Women"):
            by_year_lines.append(f"### {gender}")
            events = sorted(
                {ev for g, ev, y in eighth_by_key if g == gender and y == year}
            )
            ranked = [
                (ev, eighth_by_key[(gender, ev, year)])
                for ev in events
                if (gender, ev, year) in eighth_by_key
            ]
            for event, val in sorted(ranked, key=lambda x: -x[1]):
                by_year_lines.append(f"  {event}: {val:.1f}")
            by_year_lines.append("")
    (OUTPUT / "rq1b_nationals_8th_place_rankings_by_year_gender.txt").write_text(
        "\n".join(by_year_lines).rstrip() + "\n"
    )

    (OUTPUT / "rq1c_thresholds.json").write_text(
        json.dumps({f"{g}|{e}": v for (g, e), v in thresholds.items()}, indent=2) + "\n"
    )
    print(f"Wrote RQ1B Distance outputs to {OUTPUT}")


if __name__ == "__main__":
    analyze()
