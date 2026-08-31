"""Outdoor-only first-event recommendations in WA band [750, 950) — all metrics.

Same cohort and outdoor first-event definition as
first_event_recommendations_band_750_950_*.txt, but reports only outdoor
season openers (no indoor sections).

Reads existing multiple_metrics detail CSVs (does not re-scrape). For WA,
uses the copied detail file.

Outputs → Indoor_Outdoor_Interplay/multiple_metrics/
  outdoor_first_event_recommendations_band_750_950_{wa,vdot,purdy,mercier}.txt
  outdoor_first_event_band_750_950_detail_{metric}.csv
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    PROJECT_ROOT
    / "indoor_analysis"
    / "Indoor_Outdoor_Interplay"
    / "multiple_metrics"
)

METRICS = ("wa", "vdot", "purdy", "mercier")
METRIC_LABELS = {
    "wa": "World Athletics Points",
    "vdot": "VDOT",
    "purdy": "Gardner–Purdy",
    "mercier": "Mercier (1999)",
}
FRAMING = {
    "wa": "sports",
    "vdot": "sports",
    "purdy": "scientific",
    "mercier": "scientific",
}
BAND_LO, BAND_HI = 750, 950
MIN_N_RECOMMEND = 5
MIN_N_OVERALL = 10
GROUPS = ("Sprints", "Distance", "Hurdles", "Jumps", "Throws")


def load_detail(metric: str) -> list[dict]:
    path = OUT_DIR / f"first_event_band_750_950_detail_{metric}.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run analyze_first_event_multi_metric.py first."
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["outdoor_first_score"] = float(r["outdoor_first_score"])
        if "outdoor_first_wa" in r and r["outdoor_first_wa"] not in ("", None):
            r["outdoor_first_wa"] = float(r["outdoor_first_wa"])
    return rows


def summarize(
    records: list[tuple[str, str, str, float]],
) -> dict[tuple[str, str], list[tuple[str, int, float, float]]]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for gender, group, event, score in records:
        buckets[(gender, group, event)].append(score)
    by_cell: dict[tuple[str, str], list[tuple[str, int, float, float]]] = defaultdict(
        list
    )
    for (gender, group, event), vals in buckets.items():
        by_cell[(gender, group)].append(
            (event, len(vals), statistics.mean(vals), statistics.median(vals))
        )
    for key in by_cell:
        by_cell[key].sort(key=lambda t: (-t[2], t[0]))
    return by_cell


def overall_top(
    records: list[tuple[str, str, str, float]], min_n: int
) -> dict[str, list[tuple[str, int, float, float]]]:
    by_ge: dict[tuple[str, str], list[float]] = defaultdict(list)
    for gender, _group, event, score in records:
        by_ge[(gender, event)].append(score)
    out: dict[str, list[tuple[str, int, float, float]]] = {"Men": [], "Women": []}
    for (gender, event), vals in by_ge.items():
        if len(vals) < min_n:
            continue
        out[gender].append(
            (event, len(vals), statistics.mean(vals), statistics.median(vals))
        )
    for g in out:
        out[g].sort(key=lambda t: (-t[2], t[0]))
    return out


def best_in_cell(
    ranked: list[tuple[str, int, float, float]], min_n: int
) -> tuple[str, float, int] | None:
    eligible = [t for t in ranked if t[1] >= min_n]
    pool = eligible if eligible else ranked
    if not pool:
        return None
    ev, n, mean, _med = pool[0]
    return ev, mean, n


def write_report(metric: str, rows: list[dict]) -> None:
    label = METRIC_LABELS[metric]
    framing = FRAMING[metric]
    unit = "WA" if metric == "wa" else label

    outdoor_recs = [
        (
            r["gender"],
            r["outdoor_first_group"],
            r["outdoor_first_event"],
            r["outdoor_first_score"],
        )
        for r in rows
    ]
    scores = [r["outdoor_first_score"] for r in rows]
    cells = summarize(outdoor_recs)
    top = overall_top(outdoor_recs, MIN_N_OVERALL)

    lines: list[str] = [
        f"Best Outdoor Season Opener — Athletes with Outdoor Opener in WA {BAND_LO}–{BAND_HI}",
        f"Metric: {label} [{framing}]",
        "=" * 72,
        "",
        "Question:",
        f"  For athletes whose first outdoor-season result has WA in [{BAND_LO}, {BAND_HI}),",
        f"  which outdoor events tended to produce the highest {label} as the",
        "  outdoor season opener — by gender and event group?",
        "",
        "(Outdoor-only companion to first_event_recommendations_band_750_950_*.txt;",
        " indoor openers omitted.)",
        "",
        "Method:",
        "  • Years 2024–2026; athlete-year must have ≥1 indoor and ≥1 outdoor",
        "    individual (non-relay) result (same dual-season cohort as the full report).",
        "  • Outdoor season opener = results on the chronologically earliest outdoor",
        f"    meet date; if multiple events that day, the highest-{label} result",
        "    that day is used.",
        f"  • Inclusion: outdoor first-event WA ∈ [{BAND_LO}, {BAND_HI}).",
        f"  • Within each gender × event-group cell, rank events by mean outdoor-opener {label}.",
        f"  • “Best” recommendation uses events with n≥{MIN_N_RECOMMEND} when available.",
        f"  • Sample: {len(rows)} athlete-years.",
        "",
        f"Mean outdoor opener {unit}: {statistics.mean(scores):.1f}",
        f"Median outdoor opener {unit}: {statistics.median(scores):.1f}",
        "",
        f"Overall outdoor openers with highest mean {unit} (min n={MIN_N_OVERALL})",
    ]
    for gender in ("Men", "Women"):
        lines.append(f"  {gender}:")
        for ev, n, mean, med in top[gender][:8]:
            lines.append(f"    {ev}: mean={mean:.1f} med={med:.1f} n={n}")
    lines.append("")

    headlines: list[str] = []
    lines += [
        f"Outdoor opener — mean {unit} by gender × event group",
        "-" * 60,
        "",
    ]
    for gender in ("Men", "Women"):
        for group in GROUPS:
            ranked = cells.get((gender, group), [])
            if not ranked:
                continue
            cell_scores = [
                s for g, grp, _e, s in outdoor_recs if g == gender and grp == group
            ]
            n_ay = len(cell_scores)
            cell_mean = statistics.mean(cell_scores)
            lines.append(
                f"{gender} — {group}  (n={n_ay} athlete-years, mean {unit}={cell_mean:.1f})"
            )
            lines.append(f"  {'Event':<22} {'n':>4}  {'mean':>8}   {'median':>7}")
            for ev, n, mean, med in ranked:
                lines.append(f"  {ev:<22} {n:4d}  {mean:8.1f}   {med:7.1f}")
            best = best_in_cell(ranked, MIN_N_RECOMMEND)
            if best:
                ev, mean, n = best
                lines.append(
                    f"  → Highest mean {unit} (n≥{MIN_N_RECOMMEND}): "
                    f"{ev} ({mean:.1f}, n={n})"
                )
                headlines.append(
                    f"  Outdoor | {gender} {group}: {ev} "
                    f"(mean {unit} {mean:.1f}, n={n})"
                )
            lines.append("")

    lines += [
        f"Headline recommendations (n≥{MIN_N_RECOMMEND} within gender × group)",
        "-" * 52,
        "Outdoor season openers:",
        *headlines,
        "",
        "Caveats:",
        f"  • Conditioning on outdoor opener already in {BAND_LO}–{BAND_HI} truncates",
        "    outdoor scores from below; rankings are within that cohort only.",
        f"  • Same-day multi-event openers: highest-{label} mark that day is used.",
        "  • Observational — does not prove causality of event choice.",
        "  • Purdy/VDOT: field openers may be absent (undefined for those metrics).",
        "",
        f"Source detail CSV: outdoor_first_event_band_750_950_detail_{metric}.csv",
        "Derived from: first_event_band_750_950_detail_{metric}.csv",
        "",
    ]
    path = OUT_DIR / f"outdoor_first_event_recommendations_band_750_950_{metric}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


def write_outdoor_detail(metric: str, rows: list[dict]) -> None:
    out_rows = [
        {
            "year": r["year"],
            "athlete_id": r["athlete_id"],
            "gender": r["gender"],
            "metric": metric,
            "outdoor_first_event": r["outdoor_first_event"],
            "outdoor_first_group": r["outdoor_first_group"],
            "outdoor_first_score": r["outdoor_first_score"],
            "outdoor_first_date": r["outdoor_first_date"],
            "outdoor_first_wa": r.get("outdoor_first_wa", ""),
        }
        for r in rows
    ]
    path = OUT_DIR / f"outdoor_first_event_band_750_950_detail_{metric}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


def update_readme() -> None:
    readme = OUT_DIR / "README.txt"
    block = """
Outdoor-only first-event recommendations (band 750–950)
-------------------------------------------------------
  outdoor_first_event_recommendations_band_750_950_{wa,vdot,purdy,mercier}.txt
  outdoor_first_event_band_750_950_detail_{metric}.csv

  Same dual-season cohort as first_event_recommendations_*; indoor sections omitted.
  Regenerate:
    python indoor_analysis/Indoor_Outdoor_Interplay/analyze_outdoor_first_event_multi_metric.py
""".strip()
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "outdoor_first_event_recommendations" not in text:
            readme.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    else:
        readme.write_text(block + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        rows = load_detail(metric)
        print(f"{metric}: {len(rows)} athlete-years")
        write_outdoor_detail(metric, rows)
        write_report(metric, rows)
    update_readme()
    print("Done.")


if __name__ == "__main__":
    main()
