"""Cross-metric comparison for paper RQ1A-style best-event agreement.

Scientific framing: purdy + mercier
Sports / coaching framing: wa + vdot

Usage:
  python -m scoring.compare_metrics
  python main.py compare-metrics
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring.columns import ALL_METRICS, SCIENTIFIC_METRICS, SPORTS_METRICS, metric_framing, points_col
from scoring.marks import is_field_event

OUT = ROOT / "scoring" / "output"
NON_RELAYS = ROOT / "non_relays_findings"
# Canonical distance + corrected steeplechase WA (supersedes non_relays Distance_Events_Counting)
NEW_STEEPLE = ROOT / "new_steeplechase_data" / "Distance_Relays_Findings"
SEASONS = ["2024", "2025", "2026"]

# (discipline, base_dir, subfolder_or_None, file_prefix, event_map)
# Distance uses new_steeplechase_data (corrected steeple WA); other disciplines use non_relays_findings.
DISCIPLINES = [
    (
        "Sprints",
        NON_RELAYS,
        "Sprints_Events_Counting",
        "Sprinters",
        {3: "100m", 4: "200m", 6: "400m"},
    ),
    (
        "Distance",
        NEW_STEEPLE,
        None,
        "Relays_Distance",
        {9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"},
    ),
    (
        "Hurdles",
        NON_RELAYS,
        "Hurdles_Events_Counting",
        "Hurdles",
        {34: "100m Hurdles", 35: "110m Hurdles", 37: "400m Hurdles"},
    ),
    (
        "Jumps",
        NON_RELAYS,
        "Jumps_Events_Counting",
        "Jumps",
        {38: "Long Jump", 39: "Triple Jump", 40: "High Jump"},
    ),
    (
        "Throws",
        NON_RELAYS,
        "Throws_Events_Counting",
        "Throws",
        {41: "Shot Put", 42: "Discus", 43: "Hammer Throw", 45: "Javelin Throw"},
    ),
]


def _float(val) -> float | None:
    try:
        if val is None or val == "":
            return None
        x = float(val)
        if x != x:  # NaN
            return None
        return x
    except (TypeError, ValueError):
        return None


def load_rows(
    base: Path,
    subfolder: str | None,
    prefix: str,
    gender: str,
) -> list[dict]:
    rows: list[dict] = []
    root = base / subfolder if subfolder else base
    for year in SEASONS:
        path = root / f"{prefix}_{gender}_Outdoor_{year}_Data.csv"
        if path.exists():
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                rows.extend(csv.DictReader(f))
    return rows


def athlete_bests(
    rows: list[dict],
    event_map: dict[int, str],
    gender: str,
    metric: str,
) -> dict[str, dict[str, float]]:
    """athlete_id -> {event_name: best score}."""
    col = points_col(gender, metric)
    best: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        try:
            eid = int(float(r["running_event_id"]))
        except (TypeError, ValueError):
            continue
        if eid not in event_map:
            continue
        # Purdy/VDOT undefined for field — skip those events for those metrics
        ename = event_map[eid]
        if metric in {"purdy", "vdot"} and is_field_event(ename):
            continue
        pts = _float(r.get(col))
        if pts is None or pts <= 0:
            continue
        aid = (r.get("athlete_id") or "").strip()
        if not aid or aid.lower() == "nan":
            continue
        prev = best[aid].get(ename)
        if prev is None or pts > prev:
            best[aid][ename] = pts
    return best


def best_event_counts(bests: dict[str, dict[str, float]], min_events: int = 2) -> Counter:
    counts: Counter = Counter()
    for events in bests.values():
        if len(events) < min_events:
            continue
        # highest score wins; ties broken alphabetically for stability
        winner = max(events.items(), key=lambda kv: (kv[1], kv[0]))[0]
        counts[winner] += 1
    return counts


def agreement(
    bests_a: dict[str, dict[str, float]],
    bests_b: dict[str, dict[str, float]],
    min_events: int = 2,
) -> tuple[int, int]:
    """Return (agree, total) for athletes with ≥min_events under both metrics."""
    agree = total = 0
    common = set(bests_a) & set(bests_b)
    for aid in common:
        ea, eb = bests_a[aid], bests_b[aid]
        shared_events = set(ea) & set(eb)
        if len(shared_events) < min_events:
            continue
        # Restrict to shared event set for fair comparison
        wa = max(((e, ea[e]) for e in shared_events), key=lambda kv: kv[1])[0]
        wb = max(((e, eb[e]) for e in shared_events), key=lambda kv: kv[1])[0]
        total += 1
        if wa == wb:
            agree += 1
    return agree, total


def run_comparison() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    count_rows: list[dict] = []
    agree_rows: list[dict] = []
    report: list[str] = [
        "CROSS-METRIC BEST-EVENT COMPARISON",
        "Scientific: Gardner–Purdy + Mercier (1999)",
        "Sports / coaching: World Athletics Points + VDOT",
        "",
        "Athletes with season bests in ≥2 events in a discipline; best event = max score.",
        "Field events omitted for Purdy/VDOT (undefined).",
        "",
        "Distance data: new_steeplechase_data/Distance_Relays_Findings",
        "  (corrected steeplechase WA; supersedes non_relays Distance_Events_Counting).",
        "Other disciplines: non_relays_findings/*_Events_Counting.",
        "",
    ]

    for gender in ("Men", "Women"):
        report.append(f"## {gender}")
        for disc, base, subfolder, prefix, emap in DISCIPLINES:
            rows = load_rows(base, subfolder, prefix, gender)
            if not rows:
                continue
            metric_bests = {}
            report.append(f"### {disc}")
            for metric in ALL_METRICS:
                # Skip field-only disciplines for running-only metrics
                if metric in {"purdy", "vdot"} and disc in {"Jumps", "Throws"}:
                    continue
                bests = athlete_bests(rows, emap, gender, metric)
                counts = best_event_counts(bests, min_events=2)
                n_ath = sum(counts.values())
                metric_bests[metric] = bests
                framing = metric_framing(metric)
                report.append(f"  [{framing}] {metric} (n={n_ath} multi-event athletes)")
                for event, n in counts.most_common():
                    pct = 100.0 * n / n_ath if n_ath else 0.0
                    report.append(f"    {event}: {n} ({pct:.1f}%)")
                    count_rows.append(
                        {
                            "gender": gender,
                            "discipline": disc,
                            "metric": metric,
                            "framing": framing,
                            "event": event,
                            "n_athletes": n,
                            "pct": round(pct, 2),
                            "n_multi_event": n_ath,
                        }
                    )
                report.append("")

            # Pairwise agreement within and across framings
            metrics_avail = list(metric_bests)
            for i, m1 in enumerate(metrics_avail):
                for m2 in metrics_avail[i + 1 :]:
                    a, t = agreement(metric_bests[m1], metric_bests[m2])
                    pct = 100.0 * a / t if t else float("nan")
                    agree_rows.append(
                        {
                            "gender": gender,
                            "discipline": disc,
                            "metric_a": m1,
                            "metric_b": m2,
                            "framing_a": metric_framing(m1),
                            "framing_b": metric_framing(m2),
                            "agree": a,
                            "total": t,
                            "pct_agree": round(pct, 2) if t else None,
                        }
                    )
                    report.append(
                        f"  Agreement {m1} vs {m2}: {a}/{t}"
                        + (f" ({pct:.1f}%)" if t else " (n/a)")
                    )
            report.append("")

    # Write artifacts
    counts_path = OUT / "best_event_counts_by_metric.csv"
    agree_path = OUT / "best_event_agreement_by_metric.csv"
    report_path = OUT / "cross_metric_comparison_report.txt"

    with open(counts_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "gender",
            "discipline",
            "metric",
            "framing",
            "event",
            "n_athletes",
            "pct",
            "n_multi_event",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(count_rows)

    with open(agree_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "gender",
            "discipline",
            "metric_a",
            "metric_b",
            "framing_a",
            "framing_b",
            "agree",
            "total",
            "pct_agree",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(agree_rows)

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    # Framing summary
    summary_lines = [
        "FRAMING SUMMARY",
        f"Scientific metrics: {', '.join(SCIENTIFIC_METRICS)}",
        f"Sports metrics: {', '.join(SPORTS_METRICS)}",
        "",
        "Mean pairwise best-event agreement (all disciplines/genders with n>0):",
    ]
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in agree_rows:
        if r["total"] and r["pct_agree"] is not None:
            by_pair[(r["metric_a"], r["metric_b"])].append(r["pct_agree"])
    for pair, vals in sorted(by_pair.items()):
        summary_lines.append(f"  {pair[0]} vs {pair[1]}: mean {sum(vals)/len(vals):.1f}% over {len(vals)} slices")

    summary_path = OUT / "framing_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote {counts_path.relative_to(ROOT)}")
    print(f"Wrote {agree_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    return {
        "counts": count_rows,
        "agreement": agree_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare best-event results across scoring metrics.")
    parser.parse_args(argv)
    run_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
