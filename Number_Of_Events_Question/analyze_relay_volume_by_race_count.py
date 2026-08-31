"""Average relay / meet volume by race-count bin (Including_Relays_Plots).

For each gender × event_group, athletes are binned by how many races they
contested in that group-season. Within each bin, report:
  • mean number of relay races
  • mean number of distinct meets (Averaged_Out_Plots companion)

Two binning axes:
  1. total_result_count  — same definition as the point-jump / Averaged_Out plots
  2. individual_result_count — individual marks only (excludes relays)

Outputs → number_of_events_question/Including_Relays_Plots/
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
OUT_DIR = MODULE_ROOT / "Including_Relays_Plots"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYLIBS = PROJECT_ROOT / "non_relays_findings" / "Sprints_Events_Counting" / ".pylibs"
if PYLIBS.exists():
    sys.path.insert(0, str(PYLIBS))
os.environ.setdefault("MPLCONFIGDIR", str(MODULE_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared loaders / constants from the point-jump analysis
sys.path.insert(0, str(MODULE_ROOT))
from analyze_point_jump_by_competition_count import (  # noqa: E402
    EVENT_GROUPS,
    MIN_ATHLETES_PER_BIN,
    STANDARD_RELAY_IDS,
    AthleteResult,
    event_allowed_in_group,
    load_combined_dataset,
)

SEASONS = ["2024", "2025", "2026"]
# User callout focused on these; still produce all groups.
FOCUS_GROUPS = ("Sprints", "Distance", "Hurdles", "Jumps", "Throws")


def athlete_group_season_counts(
    athlete_results: list[AthleteResult],
) -> list[dict]:
    """One row per athlete × gender × season × event_group."""
    grouped: dict[tuple[str, str, str, str], list[AthleteResult]] = defaultdict(list)
    for result in athlete_results:
        for group in EVENT_GROUPS:
            if event_allowed_in_group(result.running_event_id, group, result.gender):
                key = (result.athlete_id, result.gender, result.season, group)
                grouped[key].append(result)

    rows: list[dict] = []
    for (athlete_id, gender, season, group), results in grouped.items():
        by_result: dict[str, AthleteResult] = {}
        for r in results:
            by_result[r.result_id] = r
        unique = list(by_result.values())
        n_relay = sum(1 for r in unique if r.running_event_id in STANDARD_RELAY_IDS)
        n_individual = len(unique) - n_relay
        meet_ids = {r.meet_id for r in unique if r.meet_id}
        rows.append(
            {
                "athlete_id": athlete_id,
                "gender": gender,
                "season": season,
                "event_group": group,
                "total_result_count": len(unique),
                "individual_result_count": n_individual,
                "relay_result_count": n_relay,
                "meet_count": len(meet_ids),
            }
        )
    return rows


def aggregate_bins(
    rows: list[dict],
    *,
    count_key: str,
    pooled_seasons: bool,
) -> list[dict]:
    """Mean relay / meet counts by (gender, event_group[, season], count_key bin)."""
    bins: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if pooled_seasons:
            key = (r["gender"], r["event_group"], r[count_key])
        else:
            key = (r["gender"], r["event_group"], r["season"], r[count_key])
        bins[key].append(r)

    out: list[dict] = []
    for key, members in sorted(bins.items()):
        n = len(members)
        if n < MIN_ATHLETES_PER_BIN:
            continue
        if pooled_seasons:
            gender, group, x = key
            season = "2024-2026"
        else:
            gender, group, season, x = key
        avg_relays = sum(m["relay_result_count"] for m in members) / n
        avg_indiv = sum(m["individual_result_count"] for m in members) / n
        avg_total = sum(m["total_result_count"] for m in members) / n
        avg_meets = sum(m["meet_count"] for m in members) / n
        pct_any_relay = 100.0 * sum(1 for m in members if m["relay_result_count"] > 0) / n
        out.append(
            {
                "gender": gender,
                "event_group": group,
                "season": season,
                "bin_axis": count_key,
                "x_races": x,
                "n_athletes": n,
                "avg_relay_races": round(avg_relays, 3),
                "avg_individual_races": round(avg_indiv, 3),
                "avg_total_races": round(avg_total, 3),
                "avg_meets": round(avg_meets, 3),
                "pct_athletes_with_any_relay": round(pct_any_relay, 1),
            }
        )
    return out


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_avg_relays(
    summary: list[dict],
    out_dir: Path,
    *,
    bin_axis: str,
    title_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for gender in ("Men", "Women"):
        for group in FOCUS_GROUPS:
            rows = [
                r
                for r in summary
                if r["gender"] == gender
                and r["event_group"] == group
                and r["bin_axis"] == bin_axis
            ]
            if not rows:
                continue
            rows.sort(key=lambda r: r["x_races"])
            x = [r["x_races"] for r in rows]
            y = [r["avg_relay_races"] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, marker="o", color="#1f77b4", linewidth=2)
            for r in rows:
                ax.annotate(
                    f"n={r['n_athletes']}",
                    (r["x_races"], r["avg_relay_races"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="#555555",
                )
            ax.set_xlabel(title_prefix)
            ax.set_ylabel("Average relay races")
            ax.set_title(
                f"{gender} {group} — avg relays by {title_prefix.lower()}\n"
                f"(bins with ≥{MIN_ATHLETES_PER_BIN} athletes; seasons 2024–2026 pooled)"
            )
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            fig.tight_layout()
            slug = f"avg_relays_by_{bin_axis}_{gender.lower()}_{group.lower()}.png"
            fig.savefig(out_dir / slug, dpi=150)
            plt.close(fig)


def plot_avg_meets(
    summary: list[dict],
    out_dir: Path,
    *,
    bin_axis: str,
    title_prefix: str,
) -> None:
    """Companion to Averaged_Out_Plots: avg distinct meets vs race-count bin."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for gender in ("Men", "Women"):
        for group in FOCUS_GROUPS:
            rows = [
                r
                for r in summary
                if r["gender"] == gender
                and r["event_group"] == group
                and r["bin_axis"] == bin_axis
            ]
            if not rows:
                continue
            rows.sort(key=lambda r: r["x_races"])
            x = [r["x_races"] for r in rows]
            y = [r["avg_meets"] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, marker="o", color="#d62728", linewidth=2)
            # Reference: if every race were a separate meet, y = x
            ax.plot(x, x, linestyle="--", color="#aaaaaa", linewidth=1, label="meets = races")
            for r in rows:
                ax.annotate(
                    f"n={r['n_athletes']}",
                    (r["x_races"], r["avg_meets"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="#555555",
                )
            ax.set_xlabel(title_prefix)
            ax.set_ylabel("Average distinct meets")
            ax.set_title(
                f"{gender} {group} — avg meets by {title_prefix.lower()}\n"
                f"(bins with ≥{MIN_ATHLETES_PER_BIN} athletes; seasons 2024–2026 pooled)"
            )
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            fig.tight_layout()
            slug = f"avg_meets_by_{bin_axis}_{gender.lower()}_{group.lower()}.png"
            fig.savefig(out_dir / slug, dpi=150)
            plt.close(fig)


def write_report(
    athlete_rows: list[dict],
    by_total: list[dict],
    by_indiv: list[dict],
    path: Path,
) -> None:
    n_ay = len(athlete_rows)
    with_relay = sum(1 for r in athlete_rows if r["relay_result_count"] > 0)
    lines = [
        "Average Relay Volume by Race-Count Bin",
        "======================================",
        "",
        "Question:",
        "  For athletes who contest x races in a gender × event-group season,",
        "  how many of those races are relays on average?",
        "",
        "Context:",
        "  The main Averaged_Out_Plots / point-jump panels use result_count as the",
        "  x-axis but do not break out how much of that volume is relays vs",
        "  individual events — especially salient for Sprints and Distance.",
        "",
        "Method:",
        "  • Same combined relay-inclusive outdoor dataset as the point-jump analysis",
        "    (March 1+; 2024–2026; deduplicated by result_id; relay legs expanded).",
        "  • Event-group membership matches analyze_point_jump_by_competition_count.py",
        "    (standard relay IDs allowed into each group panel).",
        "  • Relay IDs: 21, 22, 24, 26, 29, 30, 31.",
        f"  • Bins require ≥{MIN_ATHLETES_PER_BIN} athlete-seasons.",
        "  • Pooled seasons 2024–2026 for the primary tables/plots.",
        "",
        "Two x-axis definitions:",
        "  1. total_result_count — individual + relay (same as point-jump result_count).",
        "  2. individual_result_count — individual marks only; relays reported separately.",
        "",
        f"Athlete-season-event_group rows: {n_ay:,}",
        f"  with ≥1 relay: {with_relay:,} ({100.0 * with_relay / n_ay:.1f}%)",
        "",
    ]

    def emit_table(title: str, rows: list[dict], groups: tuple[str, ...]) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        for gender in ("Men", "Women"):
            for group in groups:
                sub = [
                    r
                    for r in rows
                    if r["gender"] == gender and r["event_group"] == group
                ]
                if not sub:
                    continue
                sub.sort(key=lambda r: r["x_races"])
                lines.append(f"  {gender} — {group}")
                lines.append(
                    f"    {'x':>4}  {'n':>5}  {'avg_relays':>10}  "
                    f"{'avg_indiv':>9}  {'avg_meets':>9}  {'pct_any_relay':>13}"
                )
                for r in sub:
                    lines.append(
                        f"    {r['x_races']:4d}  {r['n_athletes']:5d}  "
                        f"{r['avg_relay_races']:10.2f}  "
                        f"{r['avg_individual_races']:9.2f}  "
                        f"{r['avg_meets']:9.2f}  "
                        f"{r['pct_athletes_with_any_relay']:12.1f}%"
                    )
                lines.append("")
        lines.append("")

    emit_table(
        "A. Binned by TOTAL result count (matches point-jump x-axis)",
        by_total,
        FOCUS_GROUPS,
    )
    emit_table(
        "B. Binned by INDIVIDUAL result count (relays excluded from x)",
        by_indiv,
        FOCUS_GROUPS,
    )

    # Highlight Sprints / Distance — relays and meets at high volume
    lines += [
        "Headline — Sprints & Distance (TOTAL race-count bins; matches Averaged_Out_Plots x-axis)",
        "-" * 80,
    ]
    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance"):
            sub = [
                r
                for r in by_total
                if r["gender"] == gender and r["event_group"] == group
            ]
            if not sub:
                continue
            sub.sort(key=lambda r: r["x_races"])
            # spotlight x=6 when present (user example), else high end
            six = next((r for r in sub if r["x_races"] == 6), None)
            high = sub[-1]
            if six:
                lines.append(
                    f"  {gender} {group}: at 6 races → avg {six['avg_meets']:.2f} meets, "
                    f"avg {six['avg_relay_races']:.2f} relays (n={six['n_athletes']})"
                )
            lines.append(
                f"  {gender} {group}: at {high['x_races']} races → avg {high['avg_meets']:.2f} meets, "
                f"avg {high['avg_relay_races']:.2f} relays (n={high['n_athletes']})"
            )

    lines += [
        "",
        "Headline — Sprints & Distance (individual-count bins)",
        "-" * 55,
    ]
    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance"):
            sub = [
                r
                for r in by_indiv
                if r["gender"] == gender and r["event_group"] == group
            ]
            if not sub:
                continue
            sub.sort(key=lambda r: r["x_races"])
            low, high = sub[0], sub[-1]
            lines.append(
                f"  {gender} {group}: at {low['x_races']} individual race(s) → "
                f"avg {low['avg_relay_races']:.2f} relays / {low['avg_meets']:.2f} meets "
                f"(n={low['n_athletes']}); "
                f"at {high['x_races']} individual → avg {high['avg_relay_races']:.2f} "
                f"relays / {high['avg_meets']:.2f} meets (n={high['n_athletes']})"
            )
    lines += [
        "",
        "Artifacts:",
        "  athlete_relay_counts_by_season.csv",
        "  avg_relays_by_total_result_count.csv          (includes avg_meets)",
        "  avg_relays_by_individual_result_count.csv     (includes avg_meets)",
        "  plots/avg_relays_by_*_*.png",
        "  plots/avg_meets_by_total_result_count_*.png   (Averaged_Out_Plots companion)",
        "  plots/avg_meets_by_individual_result_count_*.png",
        "",
        "Source: number_of_events_question/analyze_relay_volume_by_race_count.py",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading combined dataset...")
    _deduped, athlete_results = load_combined_dataset(metric="wa")
    print(f"  Athlete-result rows: {len(athlete_results):,}")

    print("Counting individual vs relay marks per athlete-season-group...")
    athlete_rows = athlete_group_season_counts(athlete_results)
    save_csv(OUT_DIR / "athlete_relay_counts_by_season.csv", athlete_rows)

    by_total = aggregate_bins(
        athlete_rows, count_key="total_result_count", pooled_seasons=True
    )
    by_indiv = aggregate_bins(
        athlete_rows, count_key="individual_result_count", pooled_seasons=True
    )
    # also keep season-specific for optional inspection
    by_total_season = aggregate_bins(
        athlete_rows, count_key="total_result_count", pooled_seasons=False
    )
    by_indiv_season = aggregate_bins(
        athlete_rows, count_key="individual_result_count", pooled_seasons=False
    )

    save_csv(OUT_DIR / "avg_relays_by_total_result_count.csv", by_total)
    save_csv(OUT_DIR / "avg_relays_by_individual_result_count.csv", by_indiv)
    save_csv(OUT_DIR / "avg_relays_by_total_result_count_by_season.csv", by_total_season)
    save_csv(
        OUT_DIR / "avg_relays_by_individual_result_count_by_season.csv", by_indiv_season
    )

    print("Plotting...")
    plot_avg_relays(
        by_total,
        OUT_DIR / "plots",
        bin_axis="total_result_count",
        title_prefix="Total races in group (incl. relays)",
    )
    plot_avg_relays(
        by_indiv,
        OUT_DIR / "plots",
        bin_axis="individual_result_count",
        title_prefix="Individual races in group (excl. relays)",
    )
    plot_avg_meets(
        by_total,
        OUT_DIR / "plots",
        bin_axis="total_result_count",
        title_prefix="Total races in group (incl. relays)",
    )
    plot_avg_meets(
        by_indiv,
        OUT_DIR / "plots",
        bin_axis="individual_result_count",
        title_prefix="Individual races in group (excl. relays)",
    )

    # Dedicated CSV focused on Averaged_Out_Plots companion (total-count bins)
    meets_focus = [
        {
            "gender": r["gender"],
            "event_group": r["event_group"],
            "x_races": r["x_races"],
            "n_athletes": r["n_athletes"],
            "avg_meets": r["avg_meets"],
            "avg_relay_races": r["avg_relay_races"],
            "avg_individual_races": r["avg_individual_races"],
            "races_per_meet": round(r["x_races"] / r["avg_meets"], 3)
            if r["avg_meets"]
            else None,
        }
        for r in by_total
    ]
    save_csv(OUT_DIR / "avg_meets_by_total_result_count.csv", meets_focus)

    write_report(
        athlete_rows, by_total, by_indiv, OUT_DIR / "relay_volume_findings.txt"
    )
    print(f"Done. Outputs in {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  See {OUT_DIR / 'relay_volume_findings.txt'}")


if __name__ == "__main__":
    main()
