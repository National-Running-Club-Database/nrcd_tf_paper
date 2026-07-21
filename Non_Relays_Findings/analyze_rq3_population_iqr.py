"""Population-level World Athletics point dispersion (IQR) by event and gender."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT / "RQ3_Population_Dispersion"
OUTPUT.mkdir(exist_ok=True)

SEASONS = ["2024", "2025", "2026"]
MIN_ATHLETES = 100
FIELD_EVENT_IDS = {38, 39, 40, 41, 42, 43, 44, 45, 46}

EVENT_MAP = {
    int(r["running_event_id"]): r["event_name"]
    for r in csv.DictReader(open(ROOT / "Distance_Events_Counting" / "running_event.csv"))
}

DISCIPLINES = [
    ("Sprints", "Sprints_Events_Counting", "Sprinters"),
    ("Distance", "Distance_Events_Counting", "Distance"),
    ("Hurdles", "Hurdles_Events_Counting", "Hurdles"),
    ("Jumps", "Jumps_Events_Counting", "Jumps"),
    ("Throws", "Throws_Events_Counting", "Throws"),
]


def points_col(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


def parse_performance(result_time: str, event_id: int) -> float:
    raw = (result_time or "").strip().lower().replace("m", "")
    if not raw:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")
    try:
        if ":" in raw:
            mins, secs = raw.split(":", 1)
            return float(mins) * 60 + float(secs)
        return float(raw)
    except ValueError:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")


def collect_athlete_best_marks(gender: str, event_id: int) -> list[tuple[float, float]]:
    """Personal-best (perf, WA points) per unique athlete, sorted best to worst."""
    pcol = points_col(gender)
    higher_better = event_id in FIELD_EVENT_IDS
    athlete_best: dict[str, tuple[float, float]] = {}

    for _, folder, prefix in DISCIPLINES:
        for year in SEASONS:
            for row in load_season_rows(folder, prefix, gender, year):
                if int(row["running_event_id"]) != event_id:
                    continue
                aid = row["athlete_id"]
                if not aid:
                    continue
                perf = parse_performance(row["result_time"], event_id)
                if higher_better:
                    if perf == float("-inf"):
                        continue
                elif perf == float("inf"):
                    continue
                pts = float(row[pcol])
                if pts <= 0:
                    continue

                if aid not in athlete_best:
                    athlete_best[aid] = (perf, pts)
                    continue

                best_perf, _ = athlete_best[aid]
                is_better = perf > best_perf if higher_better else perf < best_perf
                if is_better:
                    athlete_best[aid] = (perf, pts)

    return sorted(athlete_best.values(), key=lambda x: x[0], reverse=higher_better)


def nth_best_mark_wa(marks: list[tuple[float, float]], rank: int) -> float | None:
    """WA points for the rank-th unique athlete's personal-best mark (1-indexed)."""
    if len(marks) < rank:
        return None
    return round(marks[rank - 1][1], 1)


def load_season_rows(folder: str, prefix: str, gender: str, year: str) -> list[dict]:
    path = ROOT / folder / f"{prefix}_{gender}_Outdoor_{year}_Data.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def quartile_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    values = sorted(values)
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    if med <= 0:
        return None
    return {
        "n": len(values),
        "q1": round(q1, 1),
        "median": round(med, 1),
        "q3": round(q3, 1),
        "iqr": round(q3 - q1, 1),
        "min": round(values[0], 1),
        "max": round(values[-1], 1),
    }


def collect_athlete_bests(gender: str) -> tuple[dict[int, dict[str, float]], dict[int, set[str]]]:
    """Return per-event athlete personal bests and unique athlete sets."""
    bests: dict[int, dict[str, float]] = defaultdict(dict)
    athletes: dict[int, set[str]] = defaultdict(set)

    for _, folder, prefix in DISCIPLINES:
        for year in SEASONS:
            for row in load_season_rows(folder, prefix, gender, year):
                eid = int(row["running_event_id"])
                aid = row["athlete_id"]
                if not aid:
                    continue
                athletes[eid].add(aid)
                pts = float(row[points_col(gender)])
                if aid not in bests[eid] or pts > bests[eid][aid]:
                    bests[eid][aid] = pts

    return bests, athletes


def _fmt_wa(value: float | None) -> str:
    return f"{value:>8.1f}" if value is not None else "     N/A"


def format_table(rows: list[dict]) -> list[str]:
    lines = [
        f"{'Rank':<5} {'Event':<22} {'N':>6} {'Median':>8} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'10th':>8} {'25th':>8}",
        "-" * 96,
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"{i:<5} {row['event_name']:<22} {row['n']:>6} "
            f"{row['median']:>8.1f} {row['q1']:>8.1f} {row['q3']:>8.1f} {row['iqr']:>8.1f}"
            f"{_fmt_wa(row['tenth_best_wa'])}{_fmt_wa(row['twenty_fifth_best_wa'])}"
        )
    return lines


def format_marks_table(rows: list[dict]) -> list[str]:
    lines = [
        f"{'Rank':<5} {'Event':<22} {'N':>6} {'10th':>8} {'25th':>8} {'50th':>8}",
        "-" * 72,
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"{i:<5} {row['event_name']:<22} {row['n']:>6} "
            f"{_fmt_wa(row['tenth_best_wa'])}{_fmt_wa(row['twenty_fifth_best_wa'])}{_fmt_wa(row['fiftieth_best_wa'])}"
        )
    return lines


def build_event_stats(gender: str) -> list[dict]:
    bests, athletes = collect_athlete_bests(gender)
    event_stats = []

    for eid, athlete_pts in bests.items():
        n_athletes = len(athletes[eid])
        if n_athletes < MIN_ATHLETES:
            continue
        scored_pts = [p for p in athlete_pts.values() if p > 0]
        if len(scored_pts) < MIN_ATHLETES:
            continue
        stats = quartile_stats(scored_pts)
        if not stats:
            continue
        best_marks = collect_athlete_best_marks(gender, eid)
        event_stats.append({
            "event_id": eid,
            "event_name": EVENT_MAP.get(eid, str(eid)),
            "tenth_best_wa": nth_best_mark_wa(best_marks, 10),
            "twenty_fifth_best_wa": nth_best_mark_wa(best_marks, 25),
            "fiftieth_best_wa": nth_best_mark_wa(best_marks, 50),
            **stats,
        })

    return event_stats


def main():
    lines = [
        "RQ3 — Population-Level World Athletics Point Dispersion (IQR)",
        "Data: NRCD outdoor track 2024–2026 combined.",
        "Unit: one personal-best WA score per athlete per event (highest score across all seasons).",
        f"Inclusion rule: at least {MIN_ATHLETES} unique athletes competed in the event over 2024–2026,",
        "with valid World Athletics scoring (median personal best > 0).",
        "Ranking: highest median WA points first.",
        "10th/25th-best mark: WA points for the personal-best mark of the 10th- and 25th-ranked unique athletes (by result_time).",
        "",
    ]

    marks_lines = [
        "RQ3 — Population-Level Top Marks by Unique Athlete (World Athletics Points)",
        "Data: NRCD outdoor track 2024–2026 combined.",
        f"Inclusion rule: at least {MIN_ATHLETES} unique athletes competed in the event over 2024–2026,",
        "with valid World Athletics scoring (median personal best > 0).",
        "Mark ranks: personal-best mark per unique athlete, ranked by result_time (fastest/best mark first).",
        "Ranking: highest 25th-best mark WA points first.",
        "",
    ]

    csv_rows = []
    marks_csv_rows = []

    for gender in ("Men", "Women"):
        event_stats = build_event_stats(gender)
        by_median = sorted(event_stats, key=lambda r: (-r["median"], r["event_name"]))
        by_25th = sorted(
            event_stats,
            key=lambda r: (-(r["twenty_fifth_best_wa"] or -1), r["event_name"]),
        )

        lines.append("=" * 75)
        lines.append(f"## {gender}")
        lines.append("=" * 75)
        lines.extend(format_table(by_median))
        lines.append("")

        marks_lines.append("=" * 75)
        marks_lines.append(f"## {gender}")
        marks_lines.append("=" * 75)
        marks_lines.extend(format_marks_table(by_25th))
        marks_lines.append("")

        for rank, row in enumerate(by_median, 1):
            csv_rows.append({
                "gender": gender,
                "rank_by_median": rank,
                "running_event_id": row["event_id"],
                "event_name": row["event_name"],
                "unique_athletes": row["n"],
                "median_wa": row["median"],
                "q1_wa": row["q1"],
                "q3_wa": row["q3"],
                "iqr_wa": row["iqr"],
                "min_wa": row["min"],
                "max_wa": row["max"],
                "tenth_best_mark_wa": row["tenth_best_wa"],
                "twenty_fifth_best_mark_wa": row["twenty_fifth_best_wa"],
            })

        for rank, row in enumerate(by_25th, 1):
            marks_csv_rows.append({
                "gender": gender,
                "rank_by_25th_mark": rank,
                "running_event_id": row["event_id"],
                "event_name": row["event_name"],
                "unique_athletes": row["n"],
                "tenth_best_mark_wa": row["tenth_best_wa"],
                "twenty_fifth_best_mark_wa": row["twenty_fifth_best_wa"],
                "fiftieth_best_mark_wa": row["fiftieth_best_wa"],
            })

    text = "\n".join(lines).rstrip() + "\n"
    txt_path = OUTPUT / "rq3_population_wa_iqr_by_event.txt"
    txt_path.write_text(text)

    marks_text = "\n".join(marks_lines).rstrip() + "\n"
    marks_txt_path = OUTPUT / "rq3_population_wa_marks_by_25th.txt"
    marks_txt_path.write_text(marks_text)

    csv_path = OUTPUT / "rq3_population_wa_iqr_by_event.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    marks_csv_path = OUTPUT / "rq3_population_wa_marks_by_25th.csv"
    with open(marks_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(marks_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(marks_csv_rows)

    print(text)
    print(f"Saved to {txt_path}")
    print(f"Saved to {csv_path}")
    print()
    print(marks_text)
    print(f"Saved to {marks_txt_path}")
    print(f"Saved to {marks_csv_path}")


if __name__ == "__main__":
    main()
