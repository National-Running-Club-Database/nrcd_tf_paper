"""RQ1B: Nationals top-8 World Athletics score distributions vs full-season distributions."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

for lib in (
    Path(__file__).parent / "Distance_Events_Counting" / ".pylibs",
    Path(__file__).parent / "Sprints_Events_Counting" / ".pylibs",
):
    if lib.exists():
        sys.path.insert(0, str(lib))
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUTPUT = ROOT / "RQ1B_Nationals"
OUTPUT.mkdir(exist_ok=True)

PRELIM_EVENT_IDS = {3, 4}  # 100m, 200m
FIELD_EVENT_IDS = {38, 39, 40, 41, 42, 43, 44, 45, 46}

EVENT_MAP = {
    int(r["running_event_id"]): r["event_name"]
    for r in csv.DictReader(open(ROOT / "Distance_Events_Counting" / "running_event.csv"))
}

DISCIPLINES = [
    ("Sprints", "Sprints_Events_Counting", "Sprinters", {3, 4, 6}),
    ("Distance", "Distance_Events_Counting", "Distance", {9, 11, 17, 20}),
    ("Hurdles", "Hurdles_Events_Counting", "Hurdles", {35, 37, 34, 36}),
    ("Jumps", "Jumps_Events_Counting", "Jumps", {38, 39, 40}),
    ("Throws", "Throws_Events_Counting", "Throws", {41, 42}),
]

SEASONS = ["2024", "2025", "2026"]


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


def points_col(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_nationals_pool(rows: list[dict], event_id: int) -> list[dict]:
    nat = [r for r in rows if r["nationals"] == "True" and int(r["running_event_id"]) == event_id]
    if event_id in PRELIM_EVENT_IDS:
        return [r for r in nat if r["event_type"] == "Prelims"]
    return [r for r in nat if r["event_type"] in ("Finals", "")]


def rank_top8(pool: list[dict], event_id: int) -> list[dict]:
    if not pool:
        return []
    higher_better = event_id in FIELD_EVENT_IDS
    ranked = sorted(
        pool,
        key=lambda r: parse_performance(r["result_time"], event_id),
        reverse=higher_better,
    )
    if len(ranked) <= 8:
        return ranked
    eighth_perf = parse_performance(ranked[7]["result_time"], event_id)
    top = []
    for r in ranked:
        perf = parse_performance(r["result_time"], event_id)
        if higher_better:
            if perf >= eighth_perf:
                top.append(r)
        else:
            if perf <= eighth_perf:
                top.append(r)
    return top


def dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "std": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def season_rows(disc_folder: str, prefix: str, gender: str, year: str) -> list[dict]:
    path = ROOT / disc_folder / f"{prefix}_{gender}_Outdoor_{year}_Data.csv"
    if not path.exists():
        return []
    return load_csv(path)


def analyze():
    all_top8_rows = []
    summary_lines_by_season: dict[str, list[str]] = {y: [] for y in SEASONS}

    for year in SEASONS:
        season_header = [
            f"RQ1B — Nationals Top 8 World Athletics Distributions ({year})",
            "Ranking: 100m/200m from Prelims; all other events from Finals (or unlabeled nationals rows).",
            "Performance rank by result_time (fastest/lowest for running; best mark for field).",
            "",
        ]
        summary_lines_by_season[year].extend(season_header)

        plot_groups: list[tuple] = []

        for disc_name, disc_folder, prefix, event_ids in DISCIPLINES:
            summary_lines_by_season[year].append(f"## {disc_name}")
            for gender in ("Men", "Women"):
                pcol = points_col(gender)
                for event_id in sorted(event_ids):
                    event_name = EVENT_MAP.get(event_id, str(event_id))
                    rows = season_rows(disc_folder, prefix, gender, year)
                    if not rows:
                        continue

                    season_event_rows = [r for r in rows if int(r["running_event_id"]) == event_id]
                    season_points = [float(r[pcol]) for r in season_event_rows]

                    pool = filter_nationals_pool(rows, event_id)
                    top8 = rank_top8(pool, event_id)
                    nat_points = [float(r[pcol]) for r in top8]

                    nat_stats = dist_stats(nat_points)
                    season_stats = dist_stats(season_points)

                    summary_lines_by_season[year].append(
                        f"\n### {gender} — {event_name}"
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
                        f"n={season_stats['n']}, mean={season_stats['mean']}, median={season_stats['median']}, "
                        f"std={season_stats['std']}, min={season_stats['min']}, max={season_stats['max']}"
                    )

                    for place, r in enumerate(top8, 1):
                        summary_lines_by_season[year].append(
                            f"  {place}. {r['result_time']} | WA={float(r[pcol]):.0f} | athlete={r['athlete_id']}"
                        )
                        all_top8_rows.append(
                            {
                                "season": year,
                                "discipline": disc_name,
                                "gender": gender,
                                "event_id": event_id,
                                "event_name": event_name,
                                "place": place,
                                "result_time": r["result_time"],
                                "world_athletics_points": float(r[pcol]),
                                "athlete_id": r["athlete_id"],
                                "event_type_used": r["event_type"] or "(unlabeled)",
                                "meet_id": r["meet_id"],
                            }
                        )

                    if nat_points and season_points:
                        plot_groups.append(
                            (
                                f"{gender} {event_name}",
                                nat_points,
                                season_points,
                                year,
                                disc_name,
                            )
                        )

            summary_lines_by_season[year].append("")

        write_season_distribution_plots(plot_groups, year)

    for year in SEASONS:
        (OUTPUT / f"rq1b_summary_{year}.txt").write_text(
            "\n".join(summary_lines_by_season[year]).rstrip() + "\n"
        )

    write_top8_csv(all_top8_rows)
    write_combined_summary(all_top8_rows)
    print(f"Wrote outputs to {OUTPUT}")


def write_top8_csv(rows: list[dict]):
    if not rows:
        return
    path = OUTPUT / "rq1b_nationals_top8_all_seasons.csv"
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_combined_summary(rows: list[dict]):
    lines = [
        "RQ1B — Combined Nationals Top-8 vs Full-Season Summary",
        "",
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["season"], r["gender"], r["event_name"])].append(r)

    for key in sorted(grouped):
        season, gender, event = key
        pts = [r["world_athletics_points"] for r in grouped[key]]
        stats = dist_stats(pts)
        lines.append(
            f"{season} | {gender} | {event}: top-8 WA mean={stats['mean']}, "
            f"median={stats['median']}, range=[{stats['min']}, {stats['max']}]"
        )
    (OUTPUT / "rq1b_combined_top8_summary.txt").write_text("\n".join(lines) + "\n")


def write_season_distribution_plots(plot_groups: list[tuple], year: str):
    if not plot_groups:
        return

    n = len(plot_groups)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5, rows_n * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, nat_pts, season_pts, _, _) in zip(axes, plot_groups):
        bp = ax.boxplot(
            [nat_pts, season_pts],
            tick_labels=["Nationals\nTop 8", "Full\nSeason"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#C73E1D")
        bp["boxes"][1].set_facecolor("#2E86AB")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("World Athletics Points")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[len(plot_groups) :]:
        ax.axis("off")

    fig.suptitle(
        f"RQ1B: Nationals Top-8 vs Full-Season WA Point Distributions ({year})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / f"rq1b_distributions_{year}.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    analyze()
