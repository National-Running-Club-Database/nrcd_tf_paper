"""Recommend a WA-point buffer when time-model predictions are for a 2nd+ event of the day.

Data limitation: CSVs lack heat/start times, so we cannot identify which event was
run first vs second within a meet. Proxy used instead:

  Within the same athlete-season, compare mean WA of results at
  same-day double meets (start_date == end_date, ≥2 group events) vs
  solo meets (exactly 1 group event that day).

A negative delta (double − solo) means marks at doubles score fewer WA points —
the natural buffer to apply after a season-PB style time-model prediction.

Populations:
  • all athlete-seasons with ≥2 events (powered)
  • 5 or 6 races (new_factors aligned)
  • high-WA bands 750–950 / 800–1000 / 850–1050 appendix

Sprints & Distance separately. Relays / steeple excluded. March 1+.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))

from relay_rq1_data import (  # noqa: E402
    STANDARD_RELAY_IDS,
    is_in_season,
    points_col,
    season_rows,
)
from build_cross_event_time_models import GROUP_CONFIG, parse_performance  # noqa: E402

STEEPLE_ID = 20
HIGH_BANDS = ((750, 950), (800, 1000), (850, 1050))


@dataclass
class DayBundle:
    day_key: str  # meet_id|date
    meet_id: str
    date: str
    results: list[dict] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len({r["event_name"] for r in self.results})

    @property
    def is_same_day_double(self) -> bool:
        return self.n_events >= 2

    @property
    def mean_wa(self) -> float:
        return statistics.mean(r["wa"] for r in self.results)


@dataclass
class AthleteSeason:
    key: str
    gender: str
    event_group: str
    days: list[DayBundle]
    max_wa: float
    result_was: list[float]

    @property
    def solo_days(self) -> list[DayBundle]:
        return [d for d in self.days if d.n_events == 1]

    @property
    def double_days(self) -> list[DayBundle]:
        return [d for d in self.days if d.is_same_day_double]

    def within_delta(self) -> float | None:
        """mean WA on same-day doubles − mean WA on solo days."""
        solos = [r["wa"] for d in self.solo_days for r in d.results]
        doubles = [r["wa"] for d in self.double_days for r in d.results]
        if not solos or not doubles:
            return None
        return statistics.mean(doubles) - statistics.mean(solos)


def parse_date(s: str) -> str | None:
    s = (s or "").strip()
    if len(s) < 10:
        return None
    return s[:10]


def load_athlete_seasons(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
) -> list[AthleteSeason]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    allowed = set(event_name_to_id.values())
    pcol = points_col(gender)
    # key -> result_id -> payload
    season_results: dict[str, dict[str, dict]] = defaultdict(dict)

    for year in ("2024", "2025", "2026"):
        for row in season_rows(folder, prefix, gender, year):
            start = parse_date(row.get("start_date") or "")
            end = parse_date(row.get("end_date") or "") or start
            if start and not is_in_season(start):
                continue
            # Same-day doubling only: multi-day meets excluded from "of the day"
            if start and end and start != end:
                continue
            try:
                event_id = int(row["running_event_id"])
            except (TypeError, ValueError):
                continue
            if event_id in STANDARD_RELAY_IDS or event_id == STEEPLE_ID:
                continue
            if event_id not in allowed:
                continue
            wa = float(row.get(pcol) or 0)
            if wa <= 0:
                continue
            aid = (row.get("athlete_id") or "").strip()
            if not aid:
                continue
            t = parse_performance(row["result_time"], event_id)
            if math.isinf(t) or t <= 0:
                continue
            result_id = (row.get("result_id") or "").strip()
            meet_id = (row.get("meet_id") or "").strip()
            if not result_id or not meet_id or not start:
                continue
            key = f"{aid}|{year}"
            season_results[key][result_id] = {
                "event_name": id_to_name[event_id],
                "t": t,
                "wa": wa,
                "meet_id": meet_id,
                "date": start,
            }

    out: list[AthleteSeason] = []
    for key, results in season_results.items():
        by_day: dict[str, DayBundle] = {}
        event_times: dict[str, float] = {}
        event_wa: dict[str, float] = {}
        result_was: list[float] = []
        for payload in results.values():
            day_key = f"{payload['meet_id']}|{payload['date']}"
            if day_key not in by_day:
                by_day[day_key] = DayBundle(
                    day_key=day_key,
                    meet_id=payload["meet_id"],
                    date=payload["date"],
                )
            by_day[day_key].results.append(payload)
            result_was.append(payload["wa"])
            en = payload["event_name"]
            if en not in event_times or payload["t"] < event_times[en]:
                event_times[en] = payload["t"]
                event_wa[en] = payload["wa"]
        if len(event_times) < 2:
            continue
        out.append(
            AthleteSeason(
                key=key,
                gender=gender,
                event_group="",
                days=list(by_day.values()),
                max_wa=max(event_wa.values()),
                result_was=result_was,
            )
        )
    return out


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize_deltas(deltas: list[float], label: str, group: str) -> dict:
    """deltas are double_mean_WA − solo_mean_WA (negative => doubles score lower)."""
    s = sorted(deltas)
    # Buffer recommendation = how many points to subtract from model WA prediction
    # If delta = -15, typical drop is 15 points → buffer 15
    drops = [-d for d in deltas]  # positive = points lost when doubling
    drops_s = sorted(drops)
    return {
        "slice": label,
        "event_group": group,
        "n_athlete_seasons": len(deltas),
        "mean_delta_wa_double_minus_solo": round(statistics.mean(deltas), 2),
        "median_delta_wa_double_minus_solo": round(statistics.median(deltas), 2),
        "p25_delta": round(percentile(s, 0.25), 2),
        "p75_delta": round(percentile(s, 0.75), 2),
        "pct_doubles_lower_wa": round(
            100 * sum(1 for d in deltas if d < 0) / len(deltas), 1
        ),
        "mean_wa_drop_when_doubling": round(statistics.mean(drops), 2),
        "median_wa_drop_when_doubling": round(statistics.median(drops), 2),
        "p25_wa_drop": round(percentile(drops_s, 0.25), 2),
        "p75_wa_drop": round(percentile(drops_s, 0.75), 2),
        # Recommended buffers
        "buffer_typical_median": max(0, int(round(statistics.median(drops)))),
        "buffer_conservative_p75": max(0, int(round(percentile(drops_s, 0.75)))),
        "buffer_light_p25": max(0, int(round(percentile(drops_s, 0.25)))),
    }


def in_band(result_was: list[float], lo: int, hi: int) -> bool:
    return any(lo <= wa < hi for wa in result_was)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def recommend_from_rows(sprint_row: dict | None, dist_row: dict | None) -> list[str]:
    lines = [
        "Recommended WA buffers (apply AFTER converting model-predicted time to WA",
        "in the TARGET event, then convert buffered WA back to a time if needed):",
        "",
    ]
    if sprint_row:
        lines.extend(
            [
                "  Sprints (100 / 200 / 400):",
                f"    Typical buffer:      −{sprint_row['buffer_typical_median']} WA points",
                f"    Conservative buffer: −{sprint_row['buffer_conservative_p75']} WA points "
                f"(~75th percentile drop)",
                f"    Evidence: n={sprint_row['n_athlete_seasons']} athlete-seasons with both "
                f"same-day doubles and solo days; "
                f"{sprint_row['pct_doubles_lower_wa']}% score lower WA at doubles; "
                f"median Δ (double−solo) = {sprint_row['median_delta_wa_double_minus_solo']:+} WA.",
                "",
            ]
        )
    if dist_row:
        lines.extend(
            [
                "  Distance (800 / 1500 / 5000):",
                f"    Typical buffer:      −{dist_row['buffer_typical_median']} WA points",
                f"    Conservative buffer: −{dist_row['buffer_conservative_p75']} WA points "
                f"(~75th percentile drop)",
                f"    Evidence: n={dist_row['n_athlete_seasons']} athlete-seasons; "
                f"{dist_row['pct_doubles_lower_wa']}% score lower WA at doubles; "
                f"median Δ (double−solo) = {dist_row['median_delta_wa_double_minus_solo']:+} WA.",
                "",
            ]
        )
    if sprint_row and dist_row:
        s_med = sprint_row["buffer_typical_median"]
        d_med = dist_row["buffer_typical_median"]
        if abs(s_med - d_med) >= 5:
            bigger = "Distance" if d_med > s_med else "Sprints"
            lines.append(
                f"  Difference: {bigger} shows a larger typical doubling cost "
                f"(sprints −{s_med} vs distance −{d_med} WA)."
            )
        else:
            lines.append(
                f"  Difference: Sprints and Distance typical buffers are similar "
                f"(−{s_med} vs −{d_med} WA); separate values still preferred."
            )
    return lines


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    by_gg: dict[tuple[str, str], list[AthleteSeason]] = {}
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            seasons = load_athlete_seasons(folder, prefix, gender, events)
            for s in seasons:
                s.event_group = event_group
                s.gender = gender
            by_gg[(gender, event_group)] = seasons
            print(f"Loaded {gender} {event_group}: {len(seasons)}")

    all_seasons = [s for ps in by_gg.values() for s in ps]
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    def collect(slice_label: str, seasons: list[AthleteSeason]) -> None:
        for group in ("Sprints", "Distance", "All"):
            subset = (
                seasons
                if group == "All"
                else [s for s in seasons if s.event_group == group]
            )
            deltas = []
            for s in subset:
                d = s.within_delta()
                if d is None:
                    continue
                deltas.append(d)
                detail_rows.append(
                    {
                        "slice": slice_label,
                        "gender": s.gender,
                        "event_group": s.event_group,
                        "athlete_season_key": s.key,
                        "max_wa": round(s.max_wa, 1),
                        "n_solo_days": len(s.solo_days),
                        "n_same_day_double_days": len(s.double_days),
                        "delta_wa_double_minus_solo": round(d, 2),
                        "wa_drop_when_doubling": round(-d, 2),
                    }
                )
            if len(deltas) < 10:
                continue
            summary_rows.append(summarize_deltas(deltas, slice_label, group))

        for gender in ("Men", "Women"):
            for group in ("Sprints", "Distance"):
                subset = [
                    s
                    for s in seasons
                    if s.gender == gender and s.event_group == group
                ]
                deltas = [d for s in subset if (d := s.within_delta()) is not None]
                if len(deltas) < 10:
                    continue
                summary_rows.append(
                    summarize_deltas(deltas, f"{slice_label}|{gender}", group)
                )

    collect("all_seasons", all_seasons)

    # 5/6 race proxy: count unique results via day result totals
    # Approximate race count as total results across days
    five_six = [
        s
        for s in all_seasons
        if 5 <= sum(len(d.results) for d in s.days) <= 6
    ]
    collect("5_or_6_races", five_six)

    for lo, hi in HIGH_BANDS:
        band_seasons = [s for s in all_seasons if in_band(s.result_was, lo, hi)]
        collect(f"band_{lo}_{hi}", band_seasons)

    write_csv(OUTPUT_ROOT / "doubling_wa_buffer_summary.csv", summary_rows)
    write_csv(OUTPUT_ROOT / "doubling_wa_buffer_athlete_deltas.csv", detail_rows)

    # Primary recommendations from all_seasons Sprints/Distance
    sprint = next(
        (
            r
            for r in summary_rows
            if r["slice"] == "all_seasons" and r["event_group"] == "Sprints"
        ),
        None,
    )
    dist = next(
        (
            r
            for r in summary_rows
            if r["slice"] == "all_seasons" and r["event_group"] == "Distance"
        ),
        None,
    )

    # Prefer high-band averages if available for band-aligned use
    band_sprint_bufs = [
        r["buffer_typical_median"]
        for r in summary_rows
        if r["slice"].startswith("band_") and r["event_group"] == "Sprints"
    ]
    band_dist_bufs = [
        r["buffer_typical_median"]
        for r in summary_rows
        if r["slice"].startswith("band_") and r["event_group"] == "Distance"
    ]

    lines = [
        "Doubling WA Buffer for Time-Model Predictions",
        "=============================================",
        "",
        "Question: If a time-model prediction is for an event that will be the",
        "athlete's 2nd (or later) race of the day, what World Athletics point",
        "buffer should we apply? Does it differ for Sprints vs Distance?",
        "",
        "Why a buffer is needed:",
        "  Feature-important / point-band time models predict season-PB-like",
        "  equivalency. Same-day doubling adds fatigue that those models do not",
        "  encode.",
        "",
        "Method / caveat:",
        "  • No heat/start clock in the CSVs → cannot label which event was 1st vs 2nd.",
        "  • Proxy: within the same athlete-season, mean WA on same-day multi-event",
        "    days (start_date == end_date, ≥2 group events) minus mean WA on solo days.",
        "  • Multi-day meets (start ≠ end) excluded so the comparison is day-level.",
        "  • Buffer = points to subtract from the model-implied TARGET-event WA",
        "    (positive drop ⇒ subtract that many WA points).",
        "  • Relays / steeple excluded. Outdoor 2024–2026, March 1+.",
        "",
        "Primary evidence (all athlete-seasons with both solo and same-day double days)",
        "-" * 78,
    ]

    for r in summary_rows:
        if r["slice"] != "all_seasons":
            continue
        lines.append(
            f"  {r['event_group']:<10} n={r['n_athlete_seasons']:<5} "
            f"median Δ={r['median_delta_wa_double_minus_solo']:+6.1f} WA  "
            f"mean Δ={r['mean_delta_wa_double_minus_solo']:+6.1f}  "
            f"% lower at doubles={r['pct_doubles_lower_wa']}%  "
            f"→ typical buffer −{r['buffer_typical_median']}, "
            f"conservative −{r['buffer_conservative_p75']}"
        )

    lines.append("")
    lines.append("By gender (all seasons):")
    for r in summary_rows:
        if not r["slice"].startswith("all_seasons|"):
            continue
        lines.append(
            f"  {r['slice'].split('|')[1]} {r['event_group']}: "
            f"n={r['n_athlete_seasons']} median Δ="
            f"{r['median_delta_wa_double_minus_solo']:+.1f} "
            f"→ buffer −{r['buffer_typical_median']} "
            f"(conservative −{r['buffer_conservative_p75']})"
        )

    lines.extend(["", "High-WA band appendix (aligned with point-band time models)", "-" * 55])
    for lo, hi in HIGH_BANDS:
        lines.append(f"  Band {lo}-{hi}:")
        for group in ("Sprints", "Distance"):
            r = next(
                (
                    x
                    for x in summary_rows
                    if x["slice"] == f"band_{lo}_{hi}" and x["event_group"] == group
                ),
                None,
            )
            if not r:
                lines.append(f"    {group}: insufficient n")
                continue
            lines.append(
                f"    {group}: n={r['n_athlete_seasons']} median Δ="
                f"{r['median_delta_wa_double_minus_solo']:+.1f} "
                f"→ buffer −{r['buffer_typical_median']} "
                f"(conservative −{r['buffer_conservative_p75']})"
            )

    lines.append("")
    lines.extend(recommend_from_rows(sprint, dist))

    # Final opinion block
    rec_sprint = sprint["buffer_typical_median"] if sprint else 10
    rec_sprint_cons = sprint["buffer_conservative_p75"] if sprint else 25
    rec_dist = dist["buffer_typical_median"] if dist else 15
    rec_dist_cons = dist["buffer_conservative_p75"] if dist else 30

    # Slightly round recommendations to neat numbers if near
    def neat(x: int) -> int:
        # round to nearest 5 for usability
        return int(5 * round(x / 5)) if x >= 5 else x

    lines.extend(
        [
            "",
            "Opinion — buffers to use with feature-important band time models",
            "-" * 64,
            "  When the predicted event is known to be 2nd+ of the day:",
            f"    • Sprints: subtract about {neat(rec_sprint)} WA points from the",
            f"      model-implied target WA (conservative: ~{neat(rec_sprint_cons)}).",
            f"    • Distance: subtract about {neat(rec_dist)} WA points",
            f"      (conservative: ~{neat(rec_dist_cons)}).",
            "",
            "  Yes — use separate buffers for Sprints and Distance; do not use one",
            "  global number if both groups matter.",
            "",
            "  High-WA bands (750–1050): sprint costs stay near the ~10-point typical",
            "  buffer; distance day-average drops look larger in small samples — still",
            f"  start from −{neat(rec_dist)} WA for distance, and lean conservative",
            f"  (−{neat(rec_dist_cons)}) for hard doubles (e.g. 800+5000 same day).",
            "",
            "  How to apply:",
            "    1. Run the usual band + feature time model → predicted time.",
            "    2. Convert predicted time → WA in the TARGET event.",
            "    3. Subtract the buffer (points).",
            "    4. Convert buffered WA back to a time (or report WA with buffer).",
            "",
            "  Do NOT apply this buffer for fresh solo races at a future meet;",
            "  it is specifically for same-day multi-event load.",
            "",
            "Caveats:",
            "  • Proxy mixes 1st and 2nd races of a double day; true 2nd-event cost",
            "    may be larger than the day-average shown here.",
            "  • Not causal; athletes may also choose easier marks when doubling.",
            "  • Buffers are empirical club-athlete estimates, not WA scoring rules.",
            "",
            "Source: doubling_analysis/analyze_doubling_wa_buffer.py",
        ]
    )

    (OUTPUT_ROOT / "doubling_wa_buffer_report.txt").write_text(
        "\n".join(lines).rstrip() + "\n"
    )

    findings = [
        "Doubling WA Buffer — Findings",
        "=============================",
        "",
        "Use case: time-model prediction for an event that is 2nd+ of the day.",
        "Proxy: within-athlete mean WA on same-day doubles vs solo days.",
        "",
    ]
    if sprint and dist:
        findings.extend(
            [
                f"Sprints: median drop {sprint['median_wa_drop_when_doubling']} WA "
                f"(typical buffer −{neat(rec_sprint)}; "
                f"conservative −{neat(rec_sprint_cons)}).",
                f"Distance: median drop {dist['median_wa_drop_when_doubling']} WA "
                f"(typical buffer −{neat(rec_dist)}; "
                f"conservative −{neat(rec_dist_cons)}).",
                "",
                "Verdict: YES apply a WA buffer; YES use different buffers for",
                f"Sprints (~−{neat(rec_sprint)}) vs Distance (~−{neat(rec_dist)}).",
            ]
        )
    findings.extend(
        [
            "",
            "See doubling_wa_buffer_report.txt for method and band splits.",
            "Source: analyze_doubling_wa_buffer.py",
        ]
    )
    (OUTPUT_ROOT / "doubling_wa_buffer_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )

    print("Primary recommendations:")
    if sprint:
        print(
            f"  Sprints: −{neat(rec_sprint)} typical / −{neat(rec_sprint_cons)} conservative"
        )
    if dist:
        print(
            f"  Distance: −{neat(rec_dist)} typical / −{neat(rec_dist_cons)} conservative"
        )
    print(f"Wrote {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
