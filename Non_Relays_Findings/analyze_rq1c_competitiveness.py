"""RQ1C: Which event should athletes pursue to be most competitive?"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT / "RQ1C_Competitiveness"
OUTPUT.mkdir(exist_ok=True)

SEASONS = ["2024", "2025", "2026"]

THRESHOLDS = {
    ("Men", "800m"): 829.3,
    ("Men", "1500m"): 827.0,
    ("Men", "200m"): 820.3,
    ("Men", "100m"): 818.0,
    ("Men", "400m"): 791.7,
    ("Men", "Long Jump"): 791.7,
    ("Men", "5000m"): 786.0,
    ("Men", "High Jump"): 732.0,
    ("Men", "400m Hurdles"): 728.3,
    ("Men", "Triple Jump"): 725.7,
    ("Men", "Shot Put"): 671.7,
    ("Men", "Discus"): 659.3,
    ("Men", "110m Hurdles"): 629.0,
    ("Men", "3000m Steeplechase"): 456.0,
    ("Women", "100m"): 814.3,
    ("Women", "5000m"): 808.5,
    ("Women", "200m"): 799.3,
    ("Women", "1500m"): 791.7,
    ("Women", "400m"): 788.3,
    ("Women", "Long Jump"): 760.3,
    ("Women", "800m"): 759.7,
    ("Women", "Triple Jump"): 741.0,
    ("Women", "High Jump"): 681.0,
    ("Women", "400m Hurdles"): 660.7,
    ("Women", "100m Hurdles"): 607.7,
    ("Women", "Shot Put"): 579.7,
    ("Women", "3000m Steeplechase"): 510.7,
    ("Women", "Discus"): 504.3,
}

DISCIPLINES = [
    ("Sprints", "Sprints_Events_Counting", "Sprinters", {
        3: "100m", 4: "200m", 6: "400m"
    }, ["100m", "200m", "400m"]),
    ("Distance", "Distance_Events_Counting", "Distance", {
        9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"
    }, ["800m", "1500m", "3000m Steeplechase", "5000m"]),
    ("Hurdles", "Hurdles_Events_Counting", "Hurdles", {
        35: "110m Hurdles", 37: "400m Hurdles", 34: "100m Hurdles", 36: "400m Hurdles"
    }, None),
    ("Jumps", "Jumps_Events_Counting", "Jumps", {
        38: "Long Jump", 39: "Triple Jump", 40: "High Jump"
    }, ["Long Jump", "Triple Jump", "High Jump"]),
    ("Throws", "Throws_Events_Counting", "Throws", {
        41: "Shot Put", 42: "Discus"
    }, ["Shot Put", "Discus"]),
]


def pcol(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


def load_combined(folder: str, prefix: str, gender: str, event_map: dict) -> list[dict]:
    rows = []
    for year in SEASONS:
        path = ROOT / folder / f"{prefix}_{gender}_Outdoor_{year}_Data.csv"
        if path.exists():
            rows.extend(csv.DictReader(open(path)))
    return rows


def athlete_event_bests(rows: list[dict], event_map: dict, gender: str) -> dict[str, dict[str, float]]:
    pc = pcol(gender)
    best = defaultdict(dict)
    for r in rows:
        eid = int(r["running_event_id"])
        if eid not in event_map:
            continue
        ev = event_map[eid]
        aid = r["athlete_id"]
        pts = float(r[pc])
        if ev not in best[aid] or pts > best[aid][ev]:
            best[aid][ev] = pts
    return dict(best)


def rq1a_event(pts: dict, order: list[str]) -> str | None:
    if not pts:
        return None
    mx = max(pts.values())
    for e in order:
        if e in pts and pts[e] == mx:
            return e
    return None


def rq1c_event(pts: dict, gender: str, order: list[str]) -> str | None:
    margins = {}
    for ev, p in pts.items():
        thr = THRESHOLDS.get((gender, ev))
        if thr is not None:
            margins[ev] = p - thr
    if not margins:
        return None
    return max(margins, key=lambda e: (margins[e], -order.index(e) if e in order else 0))


def compare_pairing(ab: dict, gender: str, ev_a: str, ev_b: str) -> tuple[int, int, int]:
    count_a = count_b = ties = 0
    for pts in ab.values():
        if ev_a not in pts or ev_b not in pts:
            continue
        ma = pts[ev_a] - THRESHOLDS[(gender, ev_a)]
        mb = pts[ev_b] - THRESHOLDS[(gender, ev_b)]
        if ma > mb:
            count_a += 1
        elif mb > ma:
            count_b += 1
        else:
            ties += 1
    return count_a, count_b, ties


def events_by_clear_count(clear_by_event: dict, total_by_event: dict) -> list[str]:
    """Order events by clear-rate numerator (count clearing bar), descending."""
    return sorted(total_by_event, key=lambda e: (-clear_by_event.get(e, 0), e))


def main():
    lines = [
        "RQ1C — Which Event Should Athletes Pursue to Be Most Competitive?",
        "Method: 3-year average nationals 8th-place WA threshold (RQ1B); 2024–2026 combined athlete personal bests.",
        "RQ1C event = event with largest margin above (or smallest margin below) the 8th-place nationals threshold.",
        "RQ1A event = event with highest absolute WA personal best.",
        "Clear-rate listings ordered by count clearing 8th-place bar (numerator), not percentage.",
        "",
    ]
    population_rates: dict[str, list[tuple]] = {"Men": [], "Women": []}

    for disc, folder, prefix, event_map, event_order in DISCIPLINES:
        lines.append("=" * 70)
        lines.append(f"## {disc}")
        lines.append("=" * 70)
        for gender in ("Men", "Women"):
            em = {k: v for k, v in event_map.items()
                  if not (gender == "Women" and v == "110m Hurdles")
                  and not (gender == "Men" and v == "100m Hurdles")}
            eo = [e for e in (event_order or []) if e in em.values()] if event_order else sorted(em.values())
            rows = load_combined(folder, prefix, gender, em)
            ab = athlete_event_bests(rows, em, gender)

            rq1a_counts = defaultdict(int)
            rq1c_counts = defaultdict(int)
            clear_by_event = defaultdict(int)
            total_by_event = defaultdict(int)
            agree = disagree = 0
            multi = 0

            for pts in ab.values():
                for ev in pts:
                    total_by_event[ev] += 1
                    thr = THRESHOLDS.get((gender, ev))
                    if thr and pts[ev] >= thr:
                        clear_by_event[ev] += 1
                a = rq1a_event(pts, eo)
                c = rq1c_event(pts, gender, eo)
                if a:
                    rq1a_counts[a] += 1
                if c:
                    rq1c_counts[c] += 1
                if len(pts) >= 2:
                    multi += 1
                    if a == c:
                        agree += 1
                    else:
                        disagree += 1

            lines.append(f"\n### {gender}")
            lines.append(f"Athletes in discipline: {len(ab)}")
            lines.append(f"Multi-event athletes (2+): {multi}")
            if multi:
                lines.append(f"RQ1A = RQ1C agreement: {agree} ({100*agree/multi:.1f}%) | Mismatch: {disagree}")
            lines.append("")
            lines.append("Most competitive event (RQ1C) — athletes who should pursue this event:")
            for ev in sorted(rq1c_counts, key=lambda e: -rq1c_counts[e]):
                nc = clear_by_event.get(ev, 0)
                nt = total_by_event.get(ev, 0)
                pct = 100 * nc / nt if nt else 0
                thr = THRESHOLDS.get((gender, ev), 0)
                lines.append(
                    f"  {ev}: {rq1c_counts[ev]} athletes | {nc}/{nt} clear 8th-place bar ({pct:.1f}%) | threshold={thr:.1f}"
                )
            lines.append("")
            lines.append("Best event (RQ1A) for comparison:")
            for ev in sorted(rq1a_counts, key=lambda e: -rq1a_counts[e]):
                lines.append(f"  {ev}: {rq1a_counts[ev]}")
            lines.append("")
            lines.append("Nationals clear-rate (ordered by count clearing 8th-place bar):")
            for ev in events_by_clear_count(clear_by_event, total_by_event):
                nc = clear_by_event[ev]
                nt = total_by_event[ev]
                pct = 100 * nc / nt if nt else 0
                lines.append(f"  {ev}: {nc}/{nt} ({pct:.1f}%)")
                population_rates[gender].append((ev, nc, nt, pct, disc))

        lines.append("")

    lines.append("=" * 70)
    lines.append("## Distance — Pairwise Competitiveness (margin above 8th-place threshold)")
    lines.append("=" * 70)
    for gender in ("Men", "Women"):
        em = {9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"}
        ab = athlete_event_bests(load_combined("Distance_Events_Counting", "Distance", gender, em), em, gender)
        lines.append(f"\n### {gender}")
        for ev_a, ev_b in [
            ("800m", "1500m"), ("1500m", "5000m"), ("800m", "5000m"),
            ("1500m", "3000m Steeplechase"),
        ]:
            ca, cb, ties = compare_pairing(ab, gender, ev_a, ev_b)
            n = ca + cb + ties
            if n:
                tie_str = f" | ties: {ties}" if ties else ""
                lines.append(
                    f"{ev_a} vs. {ev_b} ({n} in both): {ev_a} more competitive: {ca} | {ev_b} more competitive: {cb}{tie_str}"
                )

    lines.append("")
    lines.append("=" * 70)
    lines.append("## Population-Level: Nationals Clear Counts (ordered by athletes clearing 8th-place bar)")
    lines.append("=" * 70)
    for gender in ("Men", "Women"):
        lines.append(f"\n### {gender}")
        rates = population_rates[gender]
        rates.sort(key=lambda x: (-x[1], x[0]))  # sort by numerator nc descending
        for ev, nc, nt, pct, disc in rates:
            lines.append(f"  {ev:<24} {nc:>3} clear ({nc}/{nt}, {pct:.1f}%) [{disc}]")

    text = "\n".join(lines).rstrip() + "\n"
    out_path = OUTPUT / "rq1c_competitiveness_analysis.txt"
    out_path.write_text(text)
    print(text)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
