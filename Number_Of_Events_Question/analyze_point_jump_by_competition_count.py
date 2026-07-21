"""Point jump vs. competition volume analysis (relay-inclusive combined dataset)."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELAYS_ROOT = PROJECT_ROOT / "relays_findings"
OUTPUT_ROOT = Path(__file__).resolve().parent
PYLIBS = PROJECT_ROOT / "non_relays_findings" / "Sprints_Events_Counting" / ".pylibs"
if PYLIBS.exists():
    sys.path.insert(0, str(PYLIBS))

# Writable cache for headless / sandboxed runs (avoid unwritable ~/.matplotlib)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEASONS = ["2024", "2025", "2026"]
MIN_ATHLETES_PER_BIN = 10
MARCH1 = {year: datetime(int(year), 3, 1) for year in SEASONS}

DISCIPLINE_FOLDERS = [
    ("Sprinters_Relays_Findings", "Sprinters"),
    ("Distance_Relays_Findings", "Distance"),
    ("Hurdles_Relays_Findings", "Hurdles"),
    ("Jumps_Relays_Findings", "Jumps"),
    ("Throws_Relays_Findings", "Throws"),
]

EVENT_MAP = {
    int(r["running_event_id"]): r["event_name"]
    for r in csv.DictReader(
        open(PROJECT_ROOT / "non_relays_findings" / "Distance_Events_Counting" / "running_event.csv")
    )
}

STANDARD_RELAY_IDS = {21, 22, 24, 26, 29, 30, 31}

EVENT_GROUPS = {
    "Sprints": {3, 4, 6} | STANDARD_RELAY_IDS,
    "Distance": {9, 11, 17, 20} | STANDARD_RELAY_IDS,
    "Jumps": {38, 39, 40} | STANDARD_RELAY_IDS,
    "Throws": {41, 42, 43, 44, 45, 46} | STANDARD_RELAY_IDS,
}
EVENT_GROUPS["Hurdles"] = {34, 35, 36, 37} | STANDARD_RELAY_IDS  # all hurdle ids; gender filtered later

HURDLES_BY_GENDER = {
    "Men": {35, 37},
    "Women": {34, 37},
}


@dataclass(frozen=True)
class AthleteResult:
    result_id: str
    athlete_id: str
    gender: str
    season: str
    start_date: str
    meet_id: str
    running_event_id: int
    event_name: str
    wa_points: float


def parse_athlete_id(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    return str(int(float(raw)))


def leg_athlete_ids(row: dict) -> list[str]:
    ids = []
    for col in ("athlete_id", "athlete_id_2", "athlete_id_3", "athlete_id_4"):
        aid = parse_athlete_id(row.get(col, ""))
        if aid:
            ids.append(aid)
    return ids


def gender_label(raw: str) -> str | None:
    raw = (raw or "").strip().upper()
    if raw == "M":
        return "Men"
    if raw == "F":
        return "Women"
    return None


def points_for_row(row: dict, gender: str, metric: str = "wa") -> float:
    from scoring.columns import points_col

    col = points_col(gender, metric)
    try:
        return float(row[col])
    except (TypeError, ValueError, KeyError):
        return 0.0


def in_outdoor_season(date_str: str, season: str) -> bool:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.year == int(season) and d >= MARCH1[season]


def data_path(folder: str, prefix: str, gender: str, season: str) -> Path:
    return RELAYS_ROOT / folder / f"Relays_{prefix}_{gender}_Outdoor_{season}_Data.csv"


def load_combined_dataset(metric: str = "wa") -> tuple[list[dict], list[AthleteResult]]:
    """Load all relay CSVs, deduplicate by result_id, expand relay legs to athletes."""
    seen_result_ids: set[str] = set()
    deduped_rows: list[dict] = []
    athlete_results: list[AthleteResult] = []

    for folder, prefix in DISCIPLINE_FOLDERS:
        for season in SEASONS:
            for gender in ("Men", "Women"):
                path = data_path(folder, prefix, gender, season)
                if not path.exists():
                    continue
                for row in csv.DictReader(open(path, newline="")):
                    result_id = (row.get("result_id") or "").strip()
                    if not result_id or result_id in seen_result_ids:
                        continue
                    date_str = (row.get("start_date") or "").strip()
                    if not date_str or not in_outdoor_season(date_str, season):
                        continue

                    gender_val = gender_label(row.get("gender", ""))
                    if gender_val != gender:
                        continue

                    event_id = int(row["running_event_id"])
                    wa = points_for_row(row, gender, metric)
                    if wa <= 0:
                        continue

                    seen_result_ids.add(result_id)
                    row_copy = dict(row)
                    row_copy["_source_folder"] = folder
                    deduped_rows.append(row_copy)

                    event_name = EVENT_MAP.get(event_id, str(event_id))
                    meet_id = (row.get("meet_id") or "").strip()
                    if event_id in STANDARD_RELAY_IDS:
                        athlete_ids = leg_athlete_ids(row)
                    else:
                        aid = parse_athlete_id(row.get("athlete_id", ""))
                        athlete_ids = [aid] if aid else []

                    for aid in athlete_ids:
                        athlete_results.append(
                            AthleteResult(
                                result_id=result_id,
                                athlete_id=aid,
                                gender=gender,
                                season=season,
                                start_date=date_str,
                                meet_id=meet_id,
                                running_event_id=event_id,
                                event_name=event_name,
                                wa_points=wa,
                            )
                        )

    return deduped_rows, athlete_results


def event_allowed_in_group(event_id: int, group: str, gender: str) -> bool:
    if group == "Hurdles":
        return event_id in HURDLES_BY_GENDER[gender] or event_id in STANDARD_RELAY_IDS
    return event_id in EVENT_GROUPS[group]


def compute_point_jumps(
    athlete_results: list[AthleteResult],
) -> list[dict]:
    """Return rows: athlete_id, gender, season, event_group, result_count, first_wa, max_wa, point_jump."""
    grouped: dict[tuple[str, str, str, str], list[AthleteResult]] = defaultdict(list)
    for result in athlete_results:
        for group in EVENT_GROUPS:
            if event_allowed_in_group(result.running_event_id, group, result.gender):
                key = (result.athlete_id, result.gender, result.season, group)
                grouped[key].append(result)

    output: list[dict] = []
    for (athlete_id, gender, season, group), results in grouped.items():
        # Deduplicate athlete results within group-season (same result_id once)
        by_result: dict[str, AthleteResult] = {}
        for r in results:
            by_result[r.result_id] = r
        unique_results = list(by_result.values())
        if not unique_results:
            continue

        unique_results.sort(key=lambda r: (r.start_date, r.meet_id, r.result_id))
        first_date = unique_results[0].start_date
        first_day_results = [r for r in unique_results if r.start_date == first_date]
        first_wa = max(r.wa_points for r in first_day_results)
        max_wa = max(r.wa_points for r in unique_results)
        point_jump = max_wa - first_wa

        output.append(
            {
                "athlete_id": athlete_id,
                "gender": gender,
                "season": season,
                "event_group": group,
                "result_count": len(unique_results),
                "first_wa": round(first_wa, 1),
                "max_wa": round(max_wa, 1),
                "point_jump": round(point_jump, 1),
                "first_date": first_date,
            }
        )
    return output


def aggregate_for_plots(point_jumps: list[dict]) -> list[dict]:
    """Average point jump by gender, season, event_group, result_count (n>=10)."""
    bins: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in point_jumps:
        key = (row["gender"], row["season"], row["event_group"], row["result_count"])
        bins[key].append(row["point_jump"])

    summary = []
    for (gender, season, group, result_count), values in sorted(bins.items()):
        n = len(values)
        if n < MIN_ATHLETES_PER_BIN:
            continue
        summary.append(
            {
                "gender": gender,
                "season": season,
                "event_group": group,
                "result_count": result_count,
                "athlete_count": n,
                "avg_point_jump": round(sum(values) / n, 2),
                "median_point_jump": round(
                    sorted(values)[len(values) // 2]
                    if len(values) % 2
                    else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2,
                    2,
                ),
            }
        )
    return summary


def save_combined_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_point_jumps_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "athlete_id",
        "gender",
        "season",
        "event_group",
        "result_count",
        "first_wa",
        "max_wa",
        "point_jump",
        "first_date",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_averaged_summaries(summary: list[dict]) -> list[dict]:
    """Average seasonal plot values per gender, event group, and result count."""
    buckets: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in summary:
        key = (row["gender"], row["event_group"], row["result_count"])
        buckets[key].append(row)

    averaged = []
    for (gender, group, result_count), rows in sorted(buckets.items()):
        seasons = sorted({r["season"] for r in rows})
        avg_point_jump = round(sum(r["avg_point_jump"] for r in rows) / len(rows), 2)
        total_athletes = sum(r["athlete_count"] for r in rows)
        averaged.append(
            {
                "gender": gender,
                "event_group": group,
                "result_count": result_count,
                "seasons_averaged": len(seasons),
                "seasons": ",".join(seasons),
                "avg_point_jump": avg_point_jump,
                "total_athlete_count": total_athletes,
            }
        )
    return averaged


def plot_summaries(summary: list[dict], plot_dir: Path, *, title_suffix: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    groups = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]
    for gender in ("Men", "Women"):
        for season in SEASONS:
            for group in groups:
                rows = [
                    r
                    for r in summary
                    if r["gender"] == gender and r["season"] == season and r["event_group"] == group
                ]
                if not rows:
                    continue
                rows.sort(key=lambda r: r["result_count"])
                x = [r["result_count"] for r in rows]
                y = [r["avg_point_jump"] for r in rows]
                ns = [r["athlete_count"] for r in rows]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x, y, marker="o", linewidth=2, color="#2E86AB")
                for xi, yi, n in zip(x, y, ns):
                    ax.annotate(f"n={n}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
                ax.set_xlabel("Number of results in season (event group)")
                ax.set_ylabel("Average point jump (WA points)")
                ax.set_title(f"{gender} {group} — {season}\n{title_suffix}")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                slug = f"point_jump_{gender.lower()}_{group.lower()}_{season}.png"
                plt.savefig(plot_dir / slug, dpi=150)
                plt.close()


def plot_averaged_summaries(averaged: list[dict]) -> None:
    plot_dir = OUTPUT_ROOT / "Averaged_Out_Plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    groups = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]
    for gender in ("Men", "Women"):
        for group in groups:
            rows = [r for r in averaged if r["gender"] == gender and r["event_group"] == group]
            if not rows:
                continue
            rows.sort(key=lambda r: r["result_count"])
            x = [r["result_count"] for r in rows]
            y = [r["avg_point_jump"] for r in rows]
            ns = [r["total_athlete_count"] for r in rows]
            season_counts = [r["seasons_averaged"] for r in rows]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, marker="o", linewidth=2, color="#C73E1D")
            for xi, yi, n, k in zip(x, y, ns, season_counts):
                ax.annotate(
                    f"n={n}\n({k} yr)",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
            ax.set_xlabel("Number of results in season (event group)")
            ax.set_ylabel("Average point jump (WA points)")
            ax.set_title(
                f"{gender} {group} — 2024–2026 Average\n"
                f"Mean of seasonal avg point jumps (each season bin ≥{MIN_ATHLETES_PER_BIN} athletes)"
            )
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            slug = f"point_jump_{gender.lower()}_{group.lower()}_averaged_2024_2026.png"
            plt.savefig(plot_dir / slug, dpi=150)
            plt.close()


def write_findings(
    deduped_rows: list[dict],
    point_jumps: list[dict],
    summary: list[dict],
    path: Path,
) -> None:
    lines = [
        "Point Jump vs. Competition Volume — Findings",
        "==========================================",
        "",
        "Research question: Does competing in more races during a track season",
        "lead to a stronger World Athletics score improvement (point jump)?",
        "",
        "Definitions:",
        "  • Combined dataset: all relay-inclusive discipline CSVs, deduplicated by result_id.",
        "  • Relay results counted once in the combined dataset; leg athletes each receive",
        "    the team WA score for that relay appearance.",
        "  • Season: outdoor year file (2024/2025/2026), results on or after March 1.",
        "  • Point jump: season max WA minus first-day WA (max WA if multiple events on first date).",
        "  • Event group: athlete-season rows built from results in that group's events only.",
        "  • Plots: average point jump by result-count bins with ≥10 athletes.",
        "",
        f"Combined deduplicated results: {len(deduped_rows):,}",
        f"Athlete-season-event_group records: {len(point_jumps):,}",
        f"Plot bins (≥{MIN_ATHLETES_PER_BIN} athletes): {len(summary):,}",
        "",
        "Important methodological note:",
        "  Athletes with exactly 1 result in a season-event group have point jump = 0 by",
        "  definition (first-day WA equals season max). The lowest-volume plotted bin is",
        "  therefore always 0.0; positive trends with more competitions partly reflect",
        "  this structure and partly reflect athletes who improve across multiple meets.",
        "  Interpret as associative, not necessarily causal (committed athletes may both",
        "  compete more and improve more).",
        "",
    ]

    for gender in ("Men", "Women"):
        lines.append(f"## {gender}")
        lines.append("")
        for season in SEASONS:
            lines.append(f"### {season}")
            lines.append("")
            for group in ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]:
                rows = [
                    r
                    for r in summary
                    if r["gender"] == gender and r["season"] == season and r["event_group"] == group
                ]
                if not rows:
                    lines.append(f"**{group}:** No bins with ≥{MIN_ATHLETES_PER_BIN} athletes.")
                    lines.append("")
                    continue
                rows.sort(key=lambda r: r["result_count"])
                low = rows[0]
                high = rows[-1]
                lines.append(f"**{group}** ({len(rows)} bins plotted)")
                lines.append(
                    f"  Lowest volume bin: {low['result_count']} results → "
                    f"avg point jump {low['avg_point_jump']} (n={low['athlete_count']})"
                )
                lines.append(
                    f"  Highest volume bin: {high['result_count']} results → "
                    f"avg point jump {high['avg_point_jump']} (n={high['athlete_count']})"
                )
                trend = high["avg_point_jump"] - low["avg_point_jump"]
                direction = "increases" if trend > 0 else "decreases" if trend < 0 else "is flat"
                lines.append(
                    f"  End-to-end trend (lowest to highest plotted bin): avg point jump {direction} "
                    f"by {abs(trend):.1f} WA points."
                )
                lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n")


def main(argv: list[str] | None = None) -> None:
    import argparse

    sys.path.insert(0, str(PROJECT_ROOT))
    from scoring.columns import ALL_METRICS

    parser = argparse.ArgumentParser(description="Point jump vs competition volume.")
    parser.add_argument(
        "--metric",
        default="wa",
        choices=list(ALL_METRICS),
        help="Scoring metric (default: wa).",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Run for each metric into by_metric/<metric>/.",
    )
    args = parser.parse_args(argv)

    metrics = list(ALL_METRICS) if args.all_metrics else [args.metric]
    for metric in metrics:
        if args.all_metrics:
            out = OUTPUT_ROOT / "by_metric" / metric
        elif metric == "wa":
            out = OUTPUT_ROOT
        else:
            out = OUTPUT_ROOT / "by_metric" / metric
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Point-jump analysis metric={metric} → {out} ===")
        print("Loading combined dataset...")
        deduped_rows, athlete_results = load_combined_dataset(metric=metric)
        print(f"  Deduped results: {len(deduped_rows):,}")
        print(f"  Athlete-result rows: {len(athlete_results):,}")

        save_combined_csv(deduped_rows, out / "combined_relay_dataset_deduped.csv")

        print("Computing point jumps...")
        point_jumps = compute_point_jumps(athlete_results)
        save_point_jumps_csv(point_jumps, out / "athlete_point_jumps_by_season.csv")

        summary = aggregate_for_plots(point_jumps)
        save_combined_csv(summary, out / "avg_point_jump_by_result_count.csv")

        averaged_summary = aggregate_averaged_summaries(summary)
        save_combined_csv(averaged_summary, out / "avg_point_jump_by_result_count_averaged.csv")

        print("Generating plots...")
        plot_summaries(
            summary,
            out / "plots",
            title_suffix=(
                f"Avg {metric} jump by competition volume "
                f"(bins with ≥{MIN_ATHLETES_PER_BIN} athletes)"
            ),
        )
        plot_averaged_summaries(averaged_summary)

        write_findings(deduped_rows, point_jumps, summary, out / "point_jump_findings.txt")
        print(f"Done. Outputs in {out}")

        if metric == "wa":
            print("\n--- Research statistics & inferential tests ---\n")
            from research_stats import run_research_stats

            run_research_stats(point_jump_rows=point_jumps)

    # When batching all metrics, also mirror WA to module root for compatibility
    if args.all_metrics:
        wa_dir = OUTPUT_ROOT / "by_metric" / "wa"
        for name in (
            "combined_relay_dataset_deduped.csv",
            "athlete_point_jumps_by_season.csv",
            "avg_point_jump_by_result_count.csv",
            "avg_point_jump_by_result_count_averaged.csv",
            "point_jump_findings.txt",
        ):
            src = wa_dir / name
            if src.exists():
                (OUTPUT_ROOT / name).write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
