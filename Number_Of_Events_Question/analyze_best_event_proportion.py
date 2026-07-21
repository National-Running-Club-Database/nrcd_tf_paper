"""Best-event proportion: when in the season an athlete records their top WA mark."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
POINT_JUMP_ROOT = PROJECT_ROOT / "Number_Of_Events_Question"
PYLIBS = PROJECT_ROOT / "Non_Relays_Findings" / "Sprints_Events_Counting" / ".pylibs"
if PYLIBS.exists():
    sys.path.insert(0, str(PYLIBS))

sys.path.insert(0, str(POINT_JUMP_ROOT))
from analyze_point_jump_by_competition_count import (  # noqa: E402
    load_combined_dataset,
    event_allowed_in_group,
)

EVENT_GROUPS = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]
MIN_REPORT_N = 10


def compute_best_event_proportions(athlete_results) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list] = defaultdict(list)
    for r in athlete_results:
        for group in EVENT_GROUPS:
            if event_allowed_in_group(r.running_event_id, group, r.gender):
                key = (r.athlete_id, r.gender, r.season, group)
                grouped[key].append(r)

    rows = []
    for (athlete_id, gender, season, group), results in grouped.items():
        by_rid = {r.result_id: r for r in results}
        unique = sorted(by_rid.values(), key=lambda x: (x.start_date, x.meet_id, x.result_id))
        n = len(unique)
        if n == 0:
            continue

        season_max = max(r.wa_points for r in unique)
        running_max = unique[0].wa_points
        best_meet_index = 1
        for i, r in enumerate(unique, start=1):
            running_max = max(running_max, r.wa_points)
            if running_max >= season_max - 1e-9:
                best_meet_index = i
                break

        proportion = best_meet_index / n
        rows.append(
            {
                "athlete_id": athlete_id,
                "gender": gender,
                "season": season,
                "event_group": group,
                "result_count": n,
                "best_meet_index": best_meet_index,
                "best_event_proportion": round(proportion, 4),
                "season_max_wa": round(season_max, 1),
                "best_meet_date": unique[best_meet_index - 1].start_date,
            }
        )
    return rows


def proportion_bucket(p: float) -> str:
    if p <= 0.25:
        return "early (≤25%)"
    if p <= 0.50:
        return "first half (26–50%)"
    if p <= 0.75:
        return "second half (51–75%)"
    return "late (76–100%)"


def summarize_group(rows: list[dict]) -> dict:
    props = [r["best_event_proportion"] for r in rows]
    counts = [r["result_count"] for r in rows]
    buckets = defaultdict(int)
    for p in props:
        buckets[proportion_bucket(p)] += 1

    late = sum(1 for p in props if p >= 0.75)
    early = sum(1 for p in props if p <= 0.5)
    return {
        "n": len(rows),
        "avg_proportion": round(mean(props), 3),
        "median_proportion": round(median(props), 3),
        "avg_result_count": round(mean(counts), 2),
        "pct_late_peak": round(100 * late / len(props), 1),
        "pct_early_peak": round(100 * early / len(props), 1),
        "buckets": dict(buckets),
    }


def summarize_by_result_count(rows: list[dict]) -> list[dict]:
    bins: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        bins[r["result_count"]].append(r["best_event_proportion"])

    out = []
    for rc in sorted(bins):
        vals = bins[rc]
        if len(vals) < MIN_REPORT_N:
            continue
        out.append(
            {
                "result_count": rc,
                "n": len(vals),
                "avg_proportion": round(mean(vals), 3),
                "median_proportion": round(median(vals), 3),
                "pct_late_peak": round(100 * sum(1 for v in vals if v >= 0.75) / len(vals), 1),
            }
        )
    return out


def append_group_summary_lines(lines: list[str], rows: list[dict], *, include_meet_breakdown: bool) -> None:
    if not rows:
        lines.append("  (no data)")
        lines.append("")
        return
    s = summarize_group(rows)
    lines.append(f"  n={s['n']:,}")
    lines.append(f"  Avg proportion: {s['avg_proportion']:.3f} | Median: {s['median_proportion']:.3f}")
    if include_meet_breakdown:
        lines.append(f"  Avg meets/season: {s['avg_result_count']:.1f}")
    lines.append(
        f"  Peaked in final 25% of schedule: {s['pct_late_peak']:.1f}% | "
        f"Peaked by midpoint (≤50%): {s['pct_early_peak']:.1f}%"
    )
    lines.append("  Distribution:")
    for label in ["early (≤25%)", "first half (26–50%)", "second half (51–75%)", "late (76–100%)"]:
        cnt = s["buckets"].get(label, 0)
        pct = 100 * cnt / s["n"]
        lines.append(f"    {label}: {cnt:,} ({pct:.1f}%)")
    if include_meet_breakdown:
        by_rc = summarize_by_result_count(rows)
        if by_rc:
            lines.append("  By number of results in season:")
            for row in by_rc:
                lines.append(
                    f"    {row['result_count']} meets: n={row['n']}, "
                    f"avg proportion={row['avg_proportion']:.3f}, "
                    f"median={row['median_proportion']:.3f}, "
                    f"late-peak rate={row['pct_late_peak']:.1f}%"
                )
    lines.append("")


def write_season_by_season_report(all_rows: list[dict], multi_rows: list[dict]) -> None:
    seasons = sorted({r["season"] for r in all_rows})
    lines = [
        "Best Event Proportion — Season-by-Season Report",
        "==============================================",
        "",
        "Same metric and definitions as best_event_proportion_report.txt, but each",
        "section is limited to a single outdoor season (2024, 2025, or 2026).",
        "",
        "  best_event_proportion = best_meet_index / total_results_in_season",
        "  (first meet reaching season-max WA / total meets in that season)",
        "",
        "Section A per season: all athlete-seasons (including 1-result).",
        "Section B per season: ≥2 results only (primary coaching read).",
        "",
    ]

    for season in seasons:
        season_all = [r for r in all_rows if r["season"] == season]
        season_multi = [r for r in multi_rows if r["season"] == season]
        lines.append(f"## Season {season}")
        lines.append("")
        lines.append(f"Records: {len(season_all):,} total | {len(season_multi):,} with ≥2 results")
        lines.append("")

        lines.append(f"### {season} — A. All seasons (including 1-result)")
        lines.append("")
        for gender in ("Men", "Women"):
            lines.append(f"#### {gender}")
            lines.append("")
            for group in EVENT_GROUPS:
                sub = [
                    r for r in season_all if r["gender"] == gender and r["event_group"] == group
                ]
                lines.append(f"**{group}**")
                append_group_summary_lines(lines, sub, include_meet_breakdown=False)

        lines.append(f"### {season} — B. Multi-meet only (≥2 results)")
        lines.append("")
        for gender in ("Men", "Women"):
            lines.append(f"#### {gender}")
            lines.append("")
            for group in EVENT_GROUPS:
                sub = [
                    r for r in season_multi if r["gender"] == gender and r["event_group"] == group
                ]
                if not sub:
                    continue
                lines.append(f"**{group}**")
                append_group_summary_lines(lines, sub, include_meet_breakdown=True)

        lines.append(f"### {season} — C. Cross-group comparison (≥2 results)")
        lines.append("")
        combined = []
        for group in EVENT_GROUPS:
            sub = [r for r in season_multi if r["event_group"] == group]
            if len(sub) < MIN_REPORT_N:
                continue
            s = summarize_group(sub)
            combined.append((s["avg_proportion"], s["median_proportion"], s["pct_late_peak"], group, s["n"]))
        if combined:
            combined.sort(reverse=True)
            for avg_p, med_p, late, group, n in combined:
                lines.append(
                    f"  {group:10} n={n:5,}  avg={avg_p:.3f}  median={med_p:.3f}  late-peak={late:.1f}%"
                )
        else:
            lines.append("  (insufficient data)")
        lines.append("")

    lines.append("## Cross-season comparison (≥2 results)")
    lines.append("")
    lines.append("Men distance:")
    for season in seasons:
        sub = [
            r for r in multi_rows
            if r["season"] == season and r["gender"] == "Men" and r["event_group"] == "Distance"
        ]
        if sub:
            s = summarize_group(sub)
            lines.append(
                f"  {season}: n={s['n']:,}, avg={s['avg_proportion']:.3f}, "
                f"median={s['median_proportion']:.3f}, late-peak={s['pct_late_peak']:.1f}%"
            )
    lines.append("")
    lines.append("Men sprints:")
    for season in seasons:
        sub = [
            r for r in multi_rows
            if r["season"] == season and r["gender"] == "Men" and r["event_group"] == "Sprints"
        ]
        if sub:
            s = summarize_group(sub)
            lines.append(
                f"  {season}: n={s['n']:,}, avg={s['avg_proportion']:.3f}, "
                f"median={s['median_proportion']:.3f}, late-peak={s['pct_late_peak']:.1f}%"
            )
    lines.append("")
    lines.append("Women distance:")
    for season in seasons:
        sub = [
            r for r in multi_rows
            if r["season"] == season and r["gender"] == "Women" and r["event_group"] == "Distance"
        ]
        if sub:
            s = summarize_group(sub)
            lines.append(
                f"  {season}: n={s['n']:,}, avg={s['avg_proportion']:.3f}, "
                f"median={s['median_proportion']:.3f}, late-peak={s['pct_late_peak']:.1f}%"
            )
    lines.append("")
    lines.append("Women sprints:")
    for season in seasons:
        sub = [
            r for r in multi_rows
            if r["season"] == season and r["gender"] == "Women" and r["event_group"] == "Sprints"
        ]
        if sub:
            s = summarize_group(sub)
            lines.append(
                f"  {season}: n={s['n']:,}, avg={s['avg_proportion']:.3f}, "
                f"median={s['median_proportion']:.3f}, late-peak={s['pct_late_peak']:.1f}%"
            )
    lines.extend(
        [
            "",
            "Source: combined relay-inclusive dataset (deduplicated by result_id),",
            "outdoor March 1+ per season.",
        ]
    )

    (OUTPUT_ROOT / "best_event_proportion_report_by_season.txt").write_text(
        "\n".join(lines).rstrip() + "\n"
    )


def write_report(all_rows: list[dict], multi_rows: list[dict]) -> None:
    lines = [
        "Best Event Proportion — When Do Athletes Hit Their Season Best?",
        "================================================================",
        "",
        "Definition:",
        "  For each athlete × season × event group, results are ordered chronologically",
        "  (by date, meet, result_id). The best-event proportion is:",
        "",
        "    best_meet_index / total_results_in_season",
        "",
        "  where best_meet_index is the first meet at which the athlete reaches their",
        "  season-maximum World Athletics score.",
        "",
        "  Example: 4 meets, season best at meet 3 → proportion = 3/4 = 0.75",
        "",
        "  Interpretation:",
        "    • 1.00 = season best on the final (or only) competition",
        "    • 0.50 = season best by the midpoint of the schedule",
        "    • Low values = athlete peaked early relative to season length",
        "",
        "  Note: Athletes with 1 result always have proportion 1.0 by definition.",
        "  Section B excludes 1-result seasons for clearer championship-timing read.",
        "",
        f"Total athlete-season-event_group records: {len(all_rows):,}",
        f"Records with ≥2 results: {len(multi_rows):,}",
        "",
    ]

    lines.append("## A. All seasons (including 1-result)")
    lines.append("")
    for gender in ("Men", "Women"):
        lines.append(f"### {gender}")
        lines.append("")
        for group in EVENT_GROUPS:
            sub = [r for r in all_rows if r["gender"] == gender and r["event_group"] == group]
            if not sub:
                lines.append(f"**{group}:** no data")
                lines.append("")
                continue
            s = summarize_group(sub)
            lines.append(f"**{group}** (n={s['n']:,})")
            lines.append(f"  Avg proportion: {s['avg_proportion']:.3f} | Median: {s['median_proportion']:.3f}")
            lines.append(
                f"  Avg meets/season: {s['avg_result_count']:.1f} | "
                f"Peaked in final 25% of schedule: {s['pct_late_peak']:.1f}% | "
                f"Peaked by midpoint (≤50%): {s['pct_early_peak']:.1f}%"
            )
            lines.append("  Distribution:")
            for label in ["early (≤25%)", "first half (26–50%)", "second half (51–75%)", "late (76–100%)"]:
                cnt = s["buckets"].get(label, 0)
                pct = 100 * cnt / s["n"]
                lines.append(f"    {label}: {cnt:,} ({pct:.1f}%)")
            lines.append("")

    lines.append("## B. Multi-meet seasons only (≥2 results) — primary coaching read")
    lines.append("")
    for gender in ("Men", "Women"):
        lines.append(f"### {gender}")
        lines.append("")
        for group in EVENT_GROUPS:
            sub = [r for r in multi_rows if r["gender"] == gender and r["event_group"] == group]
            if not sub:
                continue
            s = summarize_group(sub)
            lines.append(f"**{group}** (n={s['n']:,})")
            lines.append(f"  Avg proportion: {s['avg_proportion']:.3f} | Median: {s['median_proportion']:.3f}")
            lines.append(
                f"  Peaked in final 25% of schedule: {s['pct_late_peak']:.1f}% | "
                f"Peaked by midpoint (≤50%): {s['pct_early_peak']:.1f}%"
            )
            by_rc = summarize_by_result_count(sub)
            if by_rc:
                lines.append("  By number of results in season:")
                for row in by_rc:
                    lines.append(
                        f"    {row['result_count']} meets: n={row['n']}, "
                        f"avg proportion={row['avg_proportion']:.3f}, "
                        f"median={row['median_proportion']:.3f}, "
                        f"late-peak rate={row['pct_late_peak']:.1f}%"
                    )
            lines.append("")

    lines.extend(
        [
            "## C. Cross-group comparison (≥2 results, all genders combined)",
            "",
        ]
    )
    combined = []
    for group in EVENT_GROUPS:
        sub = [r for r in multi_rows if r["event_group"] == group]
        if len(sub) < MIN_REPORT_N:
            continue
        s = summarize_group(sub)
        combined.append((s["avg_proportion"], s["median_proportion"], s["pct_late_peak"], group, s["n"]))
    combined.sort(reverse=True)
    lines.append("Ranked by avg best-event proportion (higher = later seasonal peak):")
    for avg_p, med_p, late, group, n in combined:
        lines.append(
            f"  {group:10} n={n:5,}  avg={avg_p:.3f}  median={med_p:.3f}  late-peak={late:.1f}%"
        )

    lines.extend(
        [
            "",
            "## D. Implications for race-volume vs championship timing",
            "",
        ]
    )

    # Pull key stats for narrative
    dist_m = summarize_group([r for r in multi_rows if r["gender"]=="Men" and r["event_group"]=="Distance"])
    sprint_m = summarize_group([r for r in multi_rows if r["gender"]=="Men" and r["event_group"]=="Sprints"])
    lines.append(
        "Among athletes with multiple meets, a substantial share still record their season "
        "best in the second half or final quarter of their schedule — higher volume does not "
        "automatically mean early peaking in this dataset."
    )
    lines.append("")
    if dist_m:
        lines.append(
            f"  Men distance (n={dist_m['n']:,}): avg proportion {dist_m['avg_proportion']:.3f}, "
            f"{dist_m['pct_late_peak']:.1f}% peak in final 25% of meets."
        )
    if sprint_m:
        lines.append(
            f"  Men sprints (n={sprint_m['n']:,}): avg proportion {sprint_m['avg_proportion']:.3f}, "
            f"{sprint_m['pct_late_peak']:.1f}% peak in final 25% of meets."
        )
    lines.extend(
        [
            "",
            "Caveats:",
            "  • Proportion depends on how many meets are scheduled; more meets create more",
            "    opportunities for a late peak (and for proportion to land below 1.0).",
            "  • Relay WA scores can shift 'best mark' timing for leg athletes.",
            "  • Athletes who miss late-season meets due to injury are not fully captured.",
            "",
            "Source: combined relay-inclusive dataset (deduplicated by result_id),",
            "outdoor 2024–2026, March 1+ per season.",
        ]
    )

    (OUTPUT_ROOT / "best_event_proportion_report.txt").write_text("\n".join(lines).rstrip() + "\n")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Loading combined dataset...")
    _, athlete_results = load_combined_dataset()
    print("Computing best-event proportions...")
    rows = compute_best_event_proportions(athlete_results)
    multi = [r for r in rows if r["result_count"] >= 2]
    save_csv(rows, OUTPUT_ROOT / "athlete_best_event_proportion.csv")
    write_report(rows, multi)
    write_season_by_season_report(rows, multi)
    print(f"Wrote {len(rows):,} records to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
