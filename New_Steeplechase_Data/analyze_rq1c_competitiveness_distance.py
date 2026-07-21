"""RQ1C: Distance-only competitiveness using New_Steeplechase_Data RQ1B thresholds."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

RELAYS_FINDINGS = Path(__file__).resolve().parents[1] / "Relays_Findings"
sys.path.insert(0, str(RELAYS_FINDINGS))

from analyze_rq1c_competitiveness_relays import (  # noqa: E402
    compare_pairing,
    events_by_clear_count,
    individual_events_in_order,
    relay_events_in_data,
    rq1a_event,
    rq1c_event,
)
from relay_best_event_analysis import (  # noqa: E402
    STANDARD_RELAY_MAP,
    merge_event_maps,
)
from relay_rq1_data import (  # noqa: E402
    SEASONS,
    STANDARD_RELAY_IDS,
    is_in_season,
    leg_athlete_ids,
    parse_athlete_id,
    points_col,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Distance_Relays_Findings"
OUTPUT = ROOT / "RQ1C_Competitiveness"
THRESHOLD_PATH = ROOT / "RQ1B_Nationals" / "rq1c_thresholds.json"

DISTANCE_INDIVIDUAL = {
    9: "800m",
    11: "1500m",
    17: "5000m",
    20: "3000m Steeplechase",
}
ORDER = ["800m", "1500m", "3000m Steeplechase", "5000m"] + [
    STANDARD_RELAY_MAP[eid] for eid in sorted(STANDARD_RELAY_IDS)
]
EVENT_MAP = merge_event_maps(DISTANCE_INDIVIDUAL, STANDARD_RELAY_MAP)


def load_thresholds() -> dict[tuple[str, str], float]:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {THRESHOLD_PATH}. Run analyze_rq1b_nationals_distance.py first."
        )
    raw = json.loads(THRESHOLD_PATH.read_text())
    return {(g, e): float(v) for key, v in raw.items() for g, e in [key.split("|", 1)]}


def load_athlete_event_bests(gender: str) -> dict[str, dict[str, float]]:
    pcol = points_col(gender)
    athlete_event_best: dict[str, dict[str, float]] = defaultdict(dict)
    for year in SEASONS:
        path = DATA_DIR / f"Relays_Distance_{gender}_Outdoor_{year}_Data.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not is_in_season(row["start_date"]):
                    continue
                event_id = int(row["running_event_id"])
                if event_id not in EVENT_MAP:
                    continue
                event_name = EVENT_MAP[event_id]
                points = float(row[pcol])
                if points <= 0:
                    continue
                if event_id in STANDARD_RELAY_IDS:
                    athlete_ids = leg_athlete_ids(row)
                else:
                    aid = parse_athlete_id(row.get("athlete_id", ""))
                    athlete_ids = [aid] if aid else []
                for aid in athlete_ids:
                    cur = athlete_event_best[aid].get(event_name)
                    if cur is None or points > cur:
                        athlete_event_best[aid][event_name] = points
    return dict(athlete_event_best)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    lines = [
        "RQ1C — Which Event Should Athletes Pursue to Be Most Competitive?",
        "[Distance only — New_Steeplechase_Data / corrected steeplechase WA]",
        "Method: 3-year average nationals 8th-place WA threshold from local RQ1B.",
        "RQ1C event = largest margin above (or smallest margin below) the 8th-place nationals threshold.",
        "RQ1A event = highest absolute WA personal best.",
        "Relay events: team WA credited to each leg athlete.",
        "",
        "=" * 70,
        "## Distance",
        "=" * 70,
    ]
    population_rates: dict[str, list[tuple]] = {"Men": [], "Women": []}

    for gender in ("Men", "Women"):
        individual = individual_events_in_order(ORDER)
        ab = load_athlete_event_bests(gender)

        rq1a_counts: dict[str, int] = defaultdict(int)
        rq1c_counts: dict[str, int] = defaultdict(int)
        clear_by_event: dict[str, int] = defaultdict(int)
        total_by_event: dict[str, int] = defaultdict(int)
        agree = disagree = multi = 0

        for pts in ab.values():
            for ev in pts:
                total_by_event[ev] += 1
                thr = thresholds.get((gender, ev))
                if thr is not None and pts[ev] >= thr:
                    clear_by_event[ev] += 1
            a = rq1a_event(pts, ORDER)
            c = rq1c_event(pts, gender, ORDER, thresholds)
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
                f"RQ1A = RQ1C agreement: {agree} ({100 * agree / multi:.1f}%) | "
                f"Mismatch: {disagree}"
            )
        lines.append("")
        lines.append("Most competitive event (RQ1C):")
        for ev in sorted(rq1c_counts, key=lambda e: -rq1c_counts[e]):
            nc = clear_by_event.get(ev, 0)
            nt = total_by_event.get(ev, 0)
            pct = 100 * nc / nt if nt else 0
            thr = thresholds.get((gender, ev), 0)
            lines.append(
                f"  {ev}: {rq1c_counts[ev]} athletes | {nc}/{nt} clear 8th-place bar "
                f"({pct:.1f}%) | threshold={thr:.1f}"
            )
        lines.append("")
        lines.append("Best event (RQ1A) for comparison:")
        for ev in sorted(rq1a_counts, key=lambda e: -rq1a_counts[e]):
            lines.append(f"  {ev}: {rq1a_counts[ev]}")
        lines.append("")
        lines.append(
            "Nationals clear-rate (ordered by count clearing 8th-place bar):"
        )
        for ev in events_by_clear_count(clear_by_event, total_by_event):
            nc = clear_by_event[ev]
            nt = total_by_event[ev]
            pct = 100 * nc / nt if nt else 0
            lines.append(f"  {ev}: {nc}/{nt} ({pct:.1f}%)")
            population_rates[gender].append((ev, nc, nt, pct))

        relays = relay_events_in_data(ab, ORDER)
        if relays and individual:
            lines.append("")
            lines.append(
                "Individual×relay competitiveness (margin above 8th-place threshold):"
            )
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
    lines.append(
        "## Population-Level: Nationals Clear Counts "
        "(ordered by athletes clearing 8th-place bar)"
    )
    lines.append("=" * 70)
    for gender in ("Men", "Women"):
        lines.append(f"\n### {gender}")
        rates = population_rates[gender]
        rates.sort(key=lambda x: (-x[1], x[0]))
        for ev, nc, nt, pct in rates:
            lines.append(f"  {ev:<24} {nc:>3} clear ({nc}/{nt}, {pct:.1f}%)")

    text = "\n".join(lines).rstrip() + "\n"
    out_path = OUTPUT / "rq1c_competitiveness_analysis.txt"
    out_path.write_text(text)

    csv_path = OUTPUT / "rq1c_thresholds_used.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["gender", "event_name", "threshold_8th_place_wa"]
        )
        w.writeheader()
        for (gender, event), thr in sorted(thresholds.items()):
            w.writerow(
                {
                    "gender": gender,
                    "event_name": event,
                    "threshold_8th_place_wa": thr,
                }
            )

    print(text)
    print(f"Saved to {out_path}")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
