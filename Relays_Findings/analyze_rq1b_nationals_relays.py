"""RQ1B: Nationals top-8 WA distributions for relay-inclusive outdoor data."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from relay_rq1_data import (
    DISCIPLINE_INDIVIDUAL,
    EVENT_MAP,
    FIELD_EVENT_IDS,
    PRELIM_EVENT_IDS,
    RELAY_SOURCE,
    ROOT,
    SEASONS,
    STANDARD_RELAY_IDS,
    hurdles_event_ids,
    parse_performance,
    points_col,
    season_rows,
)

for lib in (
    NON_RELAYS_ROOT := ROOT.parent / "non_relays_findings" / "Sprints_Events_Counting" / ".pylibs",
):
    if lib.exists():
        sys.path.insert(0, str(lib))
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = ROOT / "RQ1B_Nationals"
OUTPUT.mkdir(exist_ok=True)


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


def event_jobs(gender: str) -> list[tuple[str, str, str, int]]:
    """Unique (discipline, folder, prefix, event_id) jobs."""
    jobs: list[tuple[str, str, str, int]] = []
    for disc, folder, prefix, ids, _ in DISCIPLINE_INDIVIDUAL:
        for eid in sorted(ids):
            jobs.append((disc, folder, prefix, eid))
    h_folder, h_prefix = "Hurdles_Relays_Findings", "Hurdles"
    for eid in sorted(hurdles_event_ids(gender)):
        jobs.append(("Hurdles", h_folder, h_prefix, eid))
    r_folder, r_prefix = RELAY_SOURCE
    for eid in sorted(STANDARD_RELAY_IDS):
        jobs.append(("Relays", r_folder, r_prefix, eid))
    return jobs


def analyze() -> None:
    all_top8_rows: list[dict] = []
    summary_lines_by_season: dict[str, list[str]] = {y: [] for y in SEASONS}
    eighth_by_key: dict[tuple[str, str, str], float] = {}

    for year in SEASONS:
        season_header = [
            f"RQ1B — Nationals Top-8 World Athletics Distributions ({year}) [Relay-Inclusive Data]",
            "Ranking: 100m/200m from Prelims; all other events from Finals (or unlabeled nationals rows).",
            "Relay events ranked by team result_time; WA points are team scores.",
            "",
        ]
        summary_lines_by_season[year].extend(season_header)
        plot_groups: list[tuple] = []

        for gender in ("Men", "Women"):
            summary_lines_by_season[year].append(f"## {gender}")
            for disc, folder, prefix, event_id in event_jobs(gender):
                if disc == "Hurdles" and event_id not in hurdles_event_ids(gender):
                    continue
                event_name = EVENT_MAP.get(event_id, str(event_id))
                rows = season_rows(folder, prefix, gender, year)
                if not rows:
                    continue

                season_event_rows = [r for r in rows if int(r["running_event_id"]) == event_id]
                pcol = points_col(gender)
                season_points = [float(r[pcol]) for r in season_event_rows if float(r[pcol]) > 0]

                pool = filter_nationals_pool(rows, event_id)
                top8 = rank_top8(pool, event_id)
                nat_points = [float(r[pcol]) for r in top8 if float(r[pcol]) > 0]

                nat_stats = dist_stats(nat_points)
                season_stats = dist_stats(season_points)

                summary_lines_by_season[year].append(f"\n### {disc} — {gender} — {event_name}")
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

        write_season_distribution_plots(plot_groups, year)

    for year in SEASONS:
        (OUTPUT / f"rq1b_summary_{year}.txt").write_text(
            "\n".join(summary_lines_by_season[year]).rstrip() + "\n"
        )

    write_top8_csv(all_top8_rows)
    write_combined_summary(all_top8_rows)
    write_eighth_place_rankings(eighth_by_key)
    print(f"Wrote outputs to {OUTPUT}")


def write_top8_csv(rows: list[dict]) -> None:
    if not rows:
        return
    path = OUTPUT / "rq1b_nationals_top8_all_seasons.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_combined_summary(rows: list[dict]) -> None:
    lines = ["RQ1B — Combined Nationals Top-8 vs Full-Season Summary [Relay-Inclusive]", ""]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["season"], r["gender"], r["event_name"])].append(r)
    for key in sorted(grouped):
        season, gender, event = key
        pts = [r["world_athletics_points"] for r in grouped[key] if r["world_athletics_points"] > 0]
        stats = dist_stats(pts)
        lines.append(
            f"{season} | {gender} | {event}: top-8 WA mean={stats['mean']}, "
            f"median={stats['median']}, range=[{stats['min']}, {stats['max']}]"
        )
    (OUTPUT / "rq1b_combined_top8_summary.txt").write_text("\n".join(lines) + "\n")


def write_eighth_place_rankings(eighth_by_key: dict[tuple[str, str, str], float]) -> None:
    lines = [
        "Nationals Top-8 Eighth-Place World Athletics Points — 3-Year Average Rankings",
        "(Relay-inclusive data; average of 2024, 2025, and 2026 8th-place WA scores)",
        "Source: RQ1B nationals top-8 analysis (relays_findings)",
        "",
    ]
    thresholds: dict[tuple[str, str], float] = {}

    for gender in ("Men", "Women"):
        lines.append("=" * 60)
        lines.append(f"{gender} — 3-Year Average (2024–2026)")
        lines.append("=" * 60)
        event_avgs: list[tuple[str, float, dict[str, float]]] = []

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
            lines.append(f"  {rank:>2}. {event:<24} avg 8th-place WA: {avg:>6.1f}   ({yr_str})")
        lines.append("")

    (OUTPUT / "rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt").write_text(
        "\n".join(lines).rstrip() + "\n"
    )

    by_year_lines = ["Nationals 8th-Place WA by Year and Gender (Relay-Inclusive)", ""]
    for year in SEASONS:
        by_year_lines.append(f"## {year}")
        for gender in ("Men", "Women"):
            by_year_lines.append(f"### {gender}")
            events = sorted({ev for g, ev, y in eighth_by_key if g == gender and y == year})
            ranked = []
            for event in events:
                val = eighth_by_key.get((gender, event, year))
                if val is not None:
                    ranked.append((event, val))
            for event, val in sorted(ranked, key=lambda x: -x[1]):
                by_year_lines.append(f"  {event}: {val:.1f}")
            by_year_lines.append("")
    (OUTPUT / "rq1b_nationals_8th_place_rankings_by_year_gender.txt").write_text(
        "\n".join(by_year_lines).rstrip() + "\n"
    )

    thresh_path = OUTPUT / "rq1c_thresholds.json"
    import json
    thresh_path.write_text(
        json.dumps({f"{g}|{e}": v for (g, e), v in thresholds.items()}, indent=2) + "\n"
    )


def write_season_distribution_plots(plot_groups: list[tuple], year: str) -> None:
    if not plot_groups:
        return
    n = len(plot_groups)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5, rows_n * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, (title, nat_pts, season_pts) in zip(axes, plot_groups):
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
        f"RQ1B: Nationals Top-8 vs Full-Season WA Distributions ({year}) [Relay-Inclusive]",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / f"rq1b_distributions_{year}.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    analyze()
