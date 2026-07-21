"""RQ1C: Competitiveness analysis for relay-inclusive outdoor data."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from relay_rq1_data import (
    DISCIPLINE_INDIVIDUAL,
    ROOT,
    STANDARD_RELAY_MAP,
    STANDARD_RELAY_IDS,
    discipline_event_map,
    discipline_event_order,
    load_athlete_event_bests_for_discipline,
)

OUTPUT = ROOT / "RQ1C_Competitiveness"
THRESHOLD_PATH = ROOT / "RQ1B_Nationals" / "rq1c_thresholds.json"

DISCIPLINES = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]


def load_thresholds() -> dict[tuple[str, str], float]:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {THRESHOLD_PATH}. Run analyze_rq1b_nationals_relays.py first."
        )
    raw = json.loads(THRESHOLD_PATH.read_text())
    return {(g, e): float(v) for key, v in raw.items() for g, e in [key.split("|", 1)]}


def rq1a_event(pts: dict, order: list[str]) -> str | None:
    if not pts:
        return None
    mx = max(pts.values())
    for e in order:
        if e in pts and pts[e] == mx:
            return e
    return None


def rq1c_event(pts: dict, gender: str, order: list[str], thresholds: dict) -> str | None:
    margins = {}
    for ev, p in pts.items():
        thr = thresholds.get((gender, ev))
        if thr is not None:
            margins[ev] = p - thr
    if not margins:
        return None
    return max(margins, key=lambda e: (margins[e], -order.index(e) if e in order else 0))


def compare_pairing(
    ab: dict,
    gender: str,
    ev_a: str,
    ev_b: str,
    thresholds: dict,
) -> tuple[int, int, int]:
    count_a = count_b = ties = 0
    for pts in ab.values():
        if ev_a not in pts or ev_b not in pts:
            continue
        ta = thresholds.get((gender, ev_a))
        tb = thresholds.get((gender, ev_b))
        if ta is None or tb is None:
            continue
        ma = pts[ev_a] - ta
        mb = pts[ev_b] - tb
        if ma > mb:
            count_a += 1
        elif mb > ma:
            count_b += 1
        else:
            ties += 1
    return count_a, count_b, ties


def individual_events_in_order(order: list[str]) -> list[str]:
    relay_names = set(STANDARD_RELAY_MAP.values())
    return [e for e in order if e not in relay_names]


def relay_events_in_data(ab: dict, order: list[str]) -> list[str]:
    relay_names = set(STANDARD_RELAY_MAP.values())
    seen = set()
    for pts in ab.values():
        for ev in pts:
            if ev in relay_names:
                seen.add(ev)
    return [e for e in order if e in seen]


def events_by_clear_count(clear_by_event: dict, total_by_event: dict) -> list[str]:
    return sorted(total_by_event, key=lambda e: (-clear_by_event.get(e, 0), e))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    thresholds = load_thresholds()
    lines = [
        "RQ1C — Which Event Should Athletes Pursue to Be Most Competitive? [Relay-Inclusive]",
        "Method: 3-year average nationals 8th-place WA threshold from relays_findings RQ1B.",
        "RQ1C event = largest margin above (or smallest margin below) the 8th-place nationals threshold.",
        "RQ1A event = highest absolute WA personal best.",
        "Relay events: team WA credited to each leg athlete; clear-rate listings ordered by count clearing bar.",
        "",
    ]
    population_rates: dict[str, list[tuple]] = {"Men": [], "Women": []}

    for disc in DISCIPLINES:
        lines.append("=" * 70)
        lines.append(f"## {disc}")
        lines.append("=" * 70)

        for gender in ("Men", "Women"):
            order = discipline_event_order(disc, gender)
            individual = individual_events_in_order(order)
            ab = load_athlete_event_bests_for_discipline(disc, gender)

            rq1a_counts = defaultdict(int)
            rq1c_counts = defaultdict(int)
            clear_by_event = defaultdict(int)
            total_by_event = defaultdict(int)
            agree = disagree = 0
            multi = 0

            for pts in ab.values():
                for ev in pts:
                    total_by_event[ev] += 1
                    thr = thresholds.get((gender, ev))
                    if thr is not None and pts[ev] >= thr:
                        clear_by_event[ev] += 1
                a = rq1a_event(pts, order)
                c = rq1c_event(pts, gender, order, thresholds)
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
                lines.append(
                    f"RQ1A = RQ1C agreement: {agree} ({100 * agree / multi:.1f}%) | Mismatch: {disagree}"
                )
            lines.append("")
            lines.append("Most competitive event (RQ1C):")
            for ev in sorted(rq1c_counts, key=lambda e: -rq1c_counts[e]):
                nc = clear_by_event.get(ev, 0)
                nt = total_by_event.get(ev, 0)
                pct = 100 * nc / nt if nt else 0
                thr = thresholds.get((gender, ev), 0)
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

            relays = relay_events_in_data(ab, order)
            if relays and individual:
                lines.append("")
                lines.append("Individual×relay competitiveness (margin above 8th-place threshold):")
                for ind in individual:
                    for relay in relays:
                        ca, cb, ties = compare_pairing(ab, gender, ind, relay, thresholds)
                        n = ca + cb + ties
                        if n == 0:
                            continue
                        tie_str = f" | ties: {ties}" if ties else ""
                        lines.append(
                            f"  {ind} vs. {relay} ({n} in both): {ind} more competitive: {ca} | "
                            f"{relay} more competitive: {cb}{tie_str}"
                        )

                lines.append("")
                lines.append("Individual head-to-head competitiveness:")
                for ev_a, ev_b in combinations(individual, 2):
                    ca, cb, ties = compare_pairing(ab, gender, ev_a, ev_b, thresholds)
                    n = ca + cb + ties
                    if n == 0:
                        continue
                    tie_str = f" | ties: {ties}" if ties else ""
                    lines.append(
                        f"  {ev_a} vs. {ev_b} ({n} in both): {ev_a} more competitive: {ca} | "
                        f"{ev_b} more competitive: {cb}{tie_str}"
                    )

        lines.append("")

    lines.append("=" * 70)
    lines.append("## Population-Level: Nationals Clear Counts (ordered by athletes clearing 8th-place bar)")
    lines.append("=" * 70)
    for gender in ("Men", "Women"):
        lines.append(f"\n### {gender}")
        rates = population_rates[gender]
        rates.sort(key=lambda x: (-x[1], x[0]))
        for ev, nc, nt, pct, disc in rates:
            lines.append(f"  {ev:<24} {nc:>3} clear ({nc}/{nt}, {pct:.1f}%) [{disc}]")

    excluded = set()
    for gender in ("Men", "Women"):
        for eid, ev in STANDARD_RELAY_MAP.items():
            if (gender, ev) not in thresholds:
                excluded.add(f"  {gender} {ev} — no valid nationals 8th-place WA score in dataset")

    if excluded:
        lines.append("")
        lines.append("=" * 70)
        lines.append("## Relay Events Excluded from Threshold-Based Comparisons")
        lines.append("=" * 70)
        lines.extend(sorted(excluded))

    text = "\n".join(lines).rstrip() + "\n"
    out_path = OUTPUT / "rq1c_competitiveness_analysis.txt"
    out_path.write_text(text)

    csv_path = OUTPUT / "rq1c_thresholds_used.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gender", "event_name", "threshold_8th_place_wa"])
        w.writeheader()
        for (gender, event), thr in sorted(thresholds.items()):
            w.writerow({"gender": gender, "event_name": event, "threshold_8th_place_wa": thr})

    print(text)
    print(f"Saved to {out_path}")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
