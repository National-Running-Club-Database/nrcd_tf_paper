"""Cross-event time models stratified by World Athletics point bands.

Inclusion: an athlete-season is in a band if at least one individual (non-relay,
non-steeple) season result has WA points in that band. Modeling uses season PBs
in Sprints / Distance as in other time-model work.

Band widths analyzed separately: 100, 150, 200 WA points.
Bands slide in BAND_STEP increments (overlapping windows), not only tiled edges.

Excludes relay event IDs and 3000m Steeplechase. Sprints and Distance only.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[1]
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
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    format_params,
)

MIN_PAIR_N = 20
BAND_WIDTHS = (100, 150, 200)
BAND_STEP = 50  # overlapping windows: start every 50 points
# Steeple id from DISCIPLINE_INDIVIDUAL — never included via GROUP_CONFIG, but keep explicit
STEEPLE_ID = 20


def band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def bands_for_width(width: int, max_wa: float, step: int = BAND_STEP) -> list[tuple[int, int]]:
    """Sliding [lo, lo+width) windows starting every `step` points while lo < max_wa."""
    bands: list[tuple[int, int]] = []
    lo = 0
    while lo < max_wa:
        bands.append((lo, lo + width))
        lo += step
    return bands


def load_athlete_season_records(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """athlete_season_key -> {event_times, result_was, max_wa, n_results}."""
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    allowed = set(event_name_to_id.values())
    pcol = points_col(gender)

    # key -> result_id -> (event_name, time, wa)
    season_results: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)

    for year in ("2024", "2025", "2026"):
        for row in season_rows(folder, prefix, gender, year):
            date_str = (row.get("start_date") or "").strip()
            if date_str and not is_in_season(date_str):
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
            if not result_id:
                continue
            key = f"{aid}|{year}"
            event_name = id_to_name[event_id]
            season_results[key][result_id] = (event_name, t, wa)

    out: dict[str, dict[str, Any]] = {}
    for key, results in season_results.items():
        event_times: dict[str, float] = {}
        event_wa: dict[str, float] = {}
        result_was: list[float] = []
        for event_name, t, wa in results.values():
            result_was.append(wa)
            prev = event_times.get(event_name)
            if prev is None or t < prev:
                event_times[event_name] = t
                event_wa[event_name] = wa
        if len(event_times) < 2:
            continue
        out[key] = {
            "event_times": event_times,
            "event_wa": event_wa,
            "result_was": result_was,
            "max_wa": max(result_was),
            "n_results": len(results),
        }
    return out


def in_band(result_was: list[float], lo: int, hi: int) -> bool:
    return any(lo <= wa < hi for wa in result_was)


def run_search_for_bests(
    bests: dict[str, dict[str, float]],
    gender: str,
    event_group: str,
) -> list[dict]:
    order = EVENT_ORDER[event_group]
    best_rows: list[dict] = []
    for from_ev, to_ev in permutations(order, 2):
        pairs = [
            (b[from_ev], b[to_ev])
            for b in bests.values()
            if from_ev in b and to_ev in b
        ]
        if len(pairs) < MIN_PAIR_N:
            continue
        results = evaluate_pair(pairs)
        baseline_cv = next(r for r in results if r.name == "linear_ols").cv_median_abs
        extra = []
        if event_group == "Sprints" and from_ev == "100m" and to_ev == "400m":
            c = evaluate_chain_models(bests, "100m", "200m", "400m")
            if c:
                extra.append(c)
            m = evaluate_multivariate(bests, "100m", "200m", "400m")
            if m:
                extra.append(m)
        if event_group == "Distance" and from_ev == "800m" and to_ev == "5000m":
            c = evaluate_chain_models(bests, "800m", "1500m", "5000m")
            if c:
                extra.append(c)
            m = evaluate_multivariate(bests, "800m", "1500m", "5000m")
            if m:
                extra.append(m)
        all_r = results + extra
        winner = min(all_r, key=lambda r: r.cv_median_abs)
        best_rows.append(
            {
                "gender": gender,
                "event_group": event_group,
                "from_event": from_ev,
                "to_event": to_ev,
                "n_athlete_seasons": len(pairs),
                "best_model": winner.name,
                "linear_cv_median_abs": round(baseline_cv, 4),
                "best_cv_median_abs": round(winner.cv_median_abs, 4),
                "improvement_seconds": round(baseline_cv - winner.cv_median_abs, 4),
                "params_json": json.dumps(winner.params),
                "params_summary": format_params(winner.name, winner.params),
            }
        )
    return best_rows


def analyze_width(
    width: int,
    profiles_by_gg: dict[tuple[str, str], dict[str, dict]],
    global_max_wa: float,
) -> dict[str, Any]:
    bands = bands_for_width(width, global_max_wa)
    band_rows: list[dict] = []
    summary_rows: list[dict] = []
    cohort_rows: list[dict] = []

    # all baseline (no band filter)
    all_best: list[dict] = []
    for (gender, event_group), profiles in profiles_by_gg.items():
        bests = {k: p["event_times"] for k, p in profiles.items()}
        rows = run_search_for_bests(bests, gender, event_group)
        for r in rows:
            r2 = {**r, "band_width": width, "band": "all", "band_lo": None, "band_hi": None}
            all_best.append(r2)
            band_rows.append(r2)

    if all_best:
        summary_rows.append(
            {
                "band_width": width,
                "band": "all",
                "band_lo": None,
                "band_hi": None,
                "n_athlete_seasons": sum(len(p) for p in profiles_by_gg.values()),
                "pairs_evaluated": len(all_best),
                "mean_best_cv": round(
                    statistics.mean(r["best_cv_median_abs"] for r in all_best), 4
                ),
                "mean_linear_cv": round(
                    statistics.mean(r["linear_cv_median_abs"] for r in all_best), 4
                ),
                "median_best_cv": round(
                    statistics.median(r["best_cv_median_abs"] for r in all_best), 4
                ),
            }
        )

    for lo, hi in bands:
        label = band_label(lo, hi)
        band_best: list[dict] = []
        n_as = 0
        for (gender, event_group), profiles in profiles_by_gg.items():
            included = {
                k: p["event_times"]
                for k, p in profiles.items()
                if in_band(p["result_was"], lo, hi)
            }
            n_as += len(included)
            cohort_rows.append(
                {
                    "band_width": width,
                    "band": label,
                    "band_lo": lo,
                    "band_hi": hi,
                    "gender": gender,
                    "event_group": event_group,
                    "n_athlete_seasons": len(included),
                }
            )
            rows = run_search_for_bests(included, gender, event_group)
            for r in rows:
                r2 = {
                    **r,
                    "band_width": width,
                    "band": label,
                    "band_lo": lo,
                    "band_hi": hi,
                }
                band_best.append(r2)
                band_rows.append(r2)

        if band_best:
            summary_rows.append(
                {
                    "band_width": width,
                    "band": label,
                    "band_lo": lo,
                    "band_hi": hi,
                    "n_athlete_seasons": n_as,
                    "pairs_evaluated": len(band_best),
                    "mean_best_cv": round(
                        statistics.mean(r["best_cv_median_abs"] for r in band_best), 4
                    ),
                    "mean_linear_cv": round(
                        statistics.mean(r["linear_cv_median_abs"] for r in band_best), 4
                    ),
                    "median_best_cv": round(
                        statistics.median(r["best_cv_median_abs"] for r in band_best), 4
                    ),
                }
            )
        else:
            summary_rows.append(
                {
                    "band_width": width,
                    "band": label,
                    "band_lo": lo,
                    "band_hi": hi,
                    "n_athlete_seasons": n_as,
                    "pairs_evaluated": 0,
                    "mean_best_cv": None,
                    "mean_linear_cv": None,
                    "median_best_cv": None,
                }
            )

    return {
        "width": width,
        "band_rows": band_rows,
        "summary_rows": summary_rows,
        "cohort_rows": cohort_rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_width_report(result: dict, path: Path) -> None:
    width = result["width"]
    summaries = [s for s in result["summary_rows"] if s["band"] != "all"]
    all_sum = next((s for s in result["summary_rows"] if s["band"] == "all"), None)
    ranked = sorted(
        [s for s in summaries if s["mean_best_cv"] is not None and s["pairs_evaluated"] >= 4],
        key=lambda s: s["mean_best_cv"],
    )

    lines = [
        f"World Athletics Point-Band Time Models — Band Width {width}",
        "=" * (52 + len(str(width))),
        "",
        "Question: Which WA point band yields the most accurate cross-event",
        f"time models when band width = {width} points?",
        "",
        "Method:",
        "  • Data: outdoor CSVs 2024–2026, results on/after March 1.",
        "  • Events: Sprints (100/200/400) and Distance (800/1500/5000) only.",
        "  • Excludes relay results and 3000m Steeplechase.",
        "  • Inclusion: athlete-season enters a band if ≥1 individual result has",
        f"    WA points in [lo, lo+{width}).",
        f"  • Bands slide every {BAND_STEP} points (overlapping windows).",
        "  • Modeling: season PBs for athletes included in the band; same CV model",
        f"    search as other time models (seed={CV_SEED}, folds={CV_FOLDS}).",
        f"  • Minimum n={MIN_PAIR_N} athlete-season pairs per event pair.",
        "  • Primary metric: mean CV median |error| (seconds) across evaluated pairs.",
        "",
    ]
    if all_sum and all_sum["mean_best_cv"] is not None:
        lines.append(
            f"Unbanded baseline (all athlete-seasons): mean best CV = {all_sum['mean_best_cv']:.3f}s "
            f"({all_sum['pairs_evaluated']} pairs)."
        )
        lines.append("")

    lines.extend([
        f"Band leaderboard (width={width}; require ≥4 pairs)",
        "-" * 50,
        f"{'Band':<12} {'Ath-seas':>9} {'Pairs':>6} {'Mean best CV':>14} {'Mean linear':>12}",
    ])
    for s in ranked:
        lines.append(
            f"{s['band']:<12} {s['n_athlete_seasons']:>9} {s['pairs_evaluated']:>6} "
            f"{s['mean_best_cv']:>13.3f}s {s['mean_linear_cv']:>11.3f}s"
        )
    # bands with <4 pairs
    thin = [
        s for s in summaries if s["mean_best_cv"] is not None and s["pairs_evaluated"] < 4
    ]
    if thin:
        lines.append("")
        lines.append("Bands with <4 evaluated pairs (unstable averages):")
        for s in sorted(thin, key=lambda x: x["band_lo"] or 0):
            lines.append(
                f"  {s['band']}: n_as={s['n_athlete_seasons']}, pairs={s['pairs_evaluated']}, "
                f"mean CV={s['mean_best_cv']:.3f}s"
            )

    if ranked:
        best = ranked[0]
        lines.extend([
            "",
            "Most accurate band",
            "------------------",
            f"  {best['band']} WA points  →  mean best CV {best['mean_best_cv']:.3f}s "
            f"({best['pairs_evaluated']} pairs, {best['n_athlete_seasons']} athlete-seasons).",
        ])
        if all_sum and all_sum["mean_best_cv"] is not None:
            d = best["mean_best_cv"] - all_sum["mean_best_cv"]
            lines.append(f"  vs unbanded all: Δ {d:+.3f}s")

        # pair detail for best band
        best_pairs = [
            r
            for r in result["band_rows"]
            if r["band"] == best["band"]
        ]
        lines.extend(["", f"Pair results — band {best['band']}", "-" * 40])
        for r in sorted(
            best_pairs,
            key=lambda x: (x["event_group"], x["gender"], x["from_event"], x["to_event"]),
        ):
            lines.append(
                f"  {r['gender']} {r['event_group']} {r['from_event']}->{r['to_event']} "
                f"n={r['n_athlete_seasons']}  {r['best_model']}  "
                f"CV={r['best_cv_median_abs']:.3f}s"
            )

    # Top 3
    if len(ranked) >= 2:
        lines.extend(["", "Top 3 bands by mean best CV", "---------------------------"])
        for i, s in enumerate(ranked[:3], 1):
            lines.append(
                f"  {i}. {s['band']}: {s['mean_best_cv']:.3f}s "
                f"({s['pairs_evaluated']} pairs)"
            )

    lines.extend([
        "",
        "Caveats:",
        "  • Athletes can appear in multiple bands (any result membership).",
        "  • Low/high extreme bands have small n and few pairs.",
        "  • Mean CV across unequal pair sets favors bands that retain easy pairs;",
        "    prefer bands with ≥4 pairs and stable coverage.",
        "",
        f"Source: Point_Bands_Time_Models/analyze_point_bands_time_models.py (width={width})",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_combined_findings(results: list[dict], path: Path) -> None:
    lines = [
        "Point-Band Time Models — Findings (Widths 100 / 150 / 200)",
        "=========================================================",
        "",
        "Question: Which WA point band produces the most accurate time models?",
        "",
        f"Inclusion: ≥1 individual result (non-relay, non-steeple) in the band.",
        f"Bands: width W with starts every {BAND_STEP} points (overlapping).",
        "Groups: Sprints and Distance only.",
        "",
    ]
    winners = []
    for result in results:
        width = result["width"]
        ranked = sorted(
            [
                s
                for s in result["summary_rows"]
                if s["band"] != "all"
                and s["mean_best_cv"] is not None
                and s["pairs_evaluated"] >= 4
            ],
            key=lambda s: s["mean_best_cv"],
        )
        all_sum = next((s for s in result["summary_rows"] if s["band"] == "all"), None)
        lines.append(f"Band width {width}:")
        if ranked:
            best = ranked[0]
            winners.append((width, best))
            lines.append(
                f"  Best band: {best['band']}  mean CV {best['mean_best_cv']:.3f}s "
                f"({best['pairs_evaluated']} pairs, {best['n_athlete_seasons']} athlete-seasons)"
            )
            for s in ranked[:5]:
                lines.append(
                    f"    {s['band']}: {s['mean_best_cv']:.3f}s ({s['pairs_evaluated']} pairs)"
                )
            if all_sum and all_sum["mean_best_cv"] is not None:
                lines.append(
                    f"  Unbanded all: {all_sum['mean_best_cv']:.3f}s "
                    f"(Δ best−all = {best['mean_best_cv']-all_sum['mean_best_cv']:+.3f}s)"
                )
        else:
            lines.append("  No band with ≥4 pairs.")
        lines.append("")

    if winners:
        lines.append("Across widths, most accurate bands:")
        for width, best in winners:
            lines.append(
                f"  width {width}: {best['band']} ({best['mean_best_cv']:.3f}s)"
            )
        # overall winner by mean CV among width winners (not strictly comparable)
        overall = min(winners, key=lambda w: w[1]["mean_best_cv"])
        lines.append("")
        lines.append(
            f"Lowest mean CV among width-winners: width {overall[0]} band "
            f"{overall[1]['band']} at {overall[1]['mean_best_cv']:.3f}s "
            "(compare within width; pair sets differ)."
        )

    lines.extend([
        "",
        "See point_band_report_width_*.txt for full leaderboards.",
        "Source: analyze_point_bands_time_models.py",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    profiles_by_gg: dict[tuple[str, str], dict[str, dict]] = {}
    global_max = 0.0
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            profiles = load_athlete_season_records(folder, prefix, gender, events)
            profiles_by_gg[(gender, event_group)] = profiles
            for p in profiles.values():
                global_max = max(global_max, p["max_wa"])
            print(f"Loaded {gender} {event_group}: {len(profiles)} athlete-seasons")

    print(f"Global max WA among included results: {global_max:.1f}")

    all_band_rows: list[dict] = []
    all_summaries: list[dict] = []
    all_cohorts: list[dict] = []
    results: list[dict] = []

    for width in BAND_WIDTHS:
        print(f"Analyzing band width {width}...")
        result = analyze_width(width, profiles_by_gg, global_max)
        results.append(result)
        all_band_rows.extend(result["band_rows"])
        all_summaries.extend(result["summary_rows"])
        all_cohorts.extend(result["cohort_rows"])
        write_width_report(
            result, OUTPUT_ROOT / f"point_band_report_width_{width}.txt"
        )
        ranked = sorted(
            [
                s
                for s in result["summary_rows"]
                if s["band"] != "all"
                and s["mean_best_cv"] is not None
                and s["pairs_evaluated"] >= 4
            ],
            key=lambda s: s["mean_best_cv"],
        )
        if ranked:
            print(
                f"  best band {ranked[0]['band']}: mean CV {ranked[0]['mean_best_cv']:.3f}s "
                f"({ranked[0]['pairs_evaluated']} pairs)"
            )

    write_csv(OUTPUT_ROOT / "best_time_models_by_point_band.csv", all_band_rows)
    write_csv(OUTPUT_ROOT / "summary_by_point_band.csv", all_summaries)
    write_csv(OUTPUT_ROOT / "cohort_sizes_by_point_band.csv", all_cohorts)
    write_combined_findings(results, OUTPUT_ROOT / "point_band_findings.txt")

    # Combined report
    lines = [
        "Point-Band Time Models — Combined Report",
        "========================================",
        "",
        "Three waves: band widths 100, 150, and 200 World Athletics points.",
        f"Band starts every {BAND_STEP} points (overlapping windows).",
        "Athlete-season included in a band if ≥1 individual result falls in that band.",
        "Relays and 3000m Steeplechase excluded. Sprints & Distance only.",
        "",
    ]
    for result in results:
        width = result["width"]
        lines.append(f"=== Width {width} ===")
        ranked = sorted(
            [
                s
                for s in result["summary_rows"]
                if s["band"] != "all"
                and s["mean_best_cv"] is not None
                and s["pairs_evaluated"] >= 4
            ],
            key=lambda s: s["mean_best_cv"],
        )
        if ranked:
            best = ranked[0]
            lines.append(
                f"Most accurate: {best['band']} (mean CV {best['mean_best_cv']:.3f}s, "
                f"{best['pairs_evaluated']} pairs)"
            )
            for s in ranked[:8]:
                lines.append(
                    f"  {s['band']:<12} {s['mean_best_cv']:>8.3f}s  "
                    f"pairs={s['pairs_evaluated']:>2}  n_as={s['n_athlete_seasons']}"
                )
        lines.append("")
    lines.append("See point_band_report_width_*.txt and point_band_findings.txt.")
    lines.append("Source: analyze_point_bands_time_models.py")
    (OUTPUT_ROOT / "point_band_combined_report.txt").write_text("\n".join(lines).rstrip() + "\n")

    print(f"Wrote point-band analysis to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
