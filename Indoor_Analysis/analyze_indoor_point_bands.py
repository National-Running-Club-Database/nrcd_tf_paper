"""Indoor WA point-band time models — width 200 only.

Finds which 200-point World Athletics band yields the most accurate cross-event
time models using indoor track data only (2024–2026 seasons in indoor_analysis).

Method mirrors outdoor Point_Bands_Time_Models/analyze_point_bands_time_models.py:
  • Inclusion: athlete-season in band if ≥1 individual result WA falls in [lo, hi)
  • Bands slide every 50 points
  • Season PBs; CV median |error| in seconds
  • Sprints: 60m / 200m / 400m; Distance: 800m / Mile / 3000m
  • Excludes relays
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDOOR_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = INDOOR_ROOT / "Point_Bands_Time_Models"
TIME_MODELS_ROOT = PROJECT_ROOT / "time_models"
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))

from relay_rq1_data import parse_performance  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    format_params,
)

SEASONS = ("2024", "2025", "2026")
STANDARD_RELAY_IDS = {21, 22, 23, 24, 25, 26, 29, 30, 31}
MIN_PAIR_N = 20
BAND_WIDTH = 200
BAND_STEP = 50

SPRINT_EVENTS = {"60m": 2, "200m": 4, "400m": 6}
DISTANCE_EVENTS = {"800m": 9, "Mile": 13, "3000m": 14}
EVENT_ORDER = {
    "Sprints": ["60m", "200m", "400m"],
    "Distance": ["800m", "Mile", "3000m"],
}
GROUP_FOLDERS = {
    "Sprints": ("Indoor_Sprints", "Sprints", SPRINT_EVENTS),
    "Distance": ("Indoor_Distance", "Distance", DISTANCE_EVENTS),
}


def points_col(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


def band_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def bands_for_width(width: int, max_wa: float, step: int = BAND_STEP) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    lo = 0
    while lo < max_wa:
        bands.append((lo, lo + width))
        lo += step
    return bands


def csv_path(folder: str, group: str, gender: str, year: str) -> Path:
    return INDOOR_ROOT / folder / f"Indoor_Relays_{group}_{gender}_{year}_Data.csv"


def load_athlete_season_records(
    folder: str,
    group: str,
    gender: str,
    event_name_to_id: dict[str, int],
) -> dict[str, dict[str, Any]]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    allowed = set(event_name_to_id.values())
    pcol = points_col(gender)
    season_results: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)

    for year in SEASONS:
        path = csv_path(folder, group, gender, year)
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    event_id = int(float(row["running_event_id"]))
                except (TypeError, ValueError):
                    continue
                if event_id in STANDARD_RELAY_IDS or event_id not in allowed:
                    continue
                wa = float(row.get(pcol) or 0)
                if wa <= 0:
                    continue
                aid = (row.get("athlete_id") or "").strip()
                if not aid or aid.lower() == "nan":
                    continue
                try:
                    aid = str(int(float(aid)))
                except ValueError:
                    continue
                t = parse_performance(row.get("result_time", ""), event_id)
                if math.isinf(t) or t <= 0:
                    continue
                result_id = (row.get("result_id") or "").strip()
                if not result_id:
                    continue
                key = f"{aid}|{year}"
                season_results[key][result_id] = (id_to_name[event_id], t, wa)

    out: dict[str, dict[str, Any]] = {}
    for key, results in season_results.items():
        event_times: dict[str, float] = {}
        result_was: list[float] = []
        for event_name, t, wa in results.values():
            result_was.append(wa)
            prev = event_times.get(event_name)
            if prev is None or t < prev:
                event_times[event_name] = t
        if len(event_times) < 2:
            continue
        out[key] = {
            "event_times": event_times,
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
        if event_group == "Sprints" and from_ev == "60m" and to_ev == "400m":
            c = evaluate_chain_models(bests, "60m", "200m", "400m")
            if c:
                extra.append(c)
            m = evaluate_multivariate(bests, "60m", "200m", "400m")
            if m:
                extra.append(m)
        if event_group == "Distance" and from_ev == "800m" and to_ev == "3000m":
            c = evaluate_chain_models(bests, "800m", "Mile", "3000m")
            if c:
                extra.append(c)
            m = evaluate_multivariate(bests, "800m", "Mile", "3000m")
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


def write_report(result: dict, path: Path) -> None:
    width = result["width"]
    summaries = [s for s in result["summary_rows"] if s["band"] != "all"]
    all_sum = next((s for s in result["summary_rows"] if s["band"] == "all"), None)
    ranked = sorted(
        [s for s in summaries if s["mean_best_cv"] is not None and s["pairs_evaluated"] >= 4],
        key=lambda s: s["mean_best_cv"],
    )

    lines = [
        f"Indoor World Athletics Point-Band Time Models — Band Width {width}",
        "=" * (58 + len(str(width))),
        "",
        "Question: Which WA point band yields the most accurate cross-event",
        f"time models on indoor data when band width = {width} points?",
        "",
        "Method:",
        "  • Data: indoor_analysis CSVs 2024–2026 (all dates in files).",
        "  • Events: Sprints (60/200/400) and Distance (800/Mile/3000) only.",
        "  • Excludes relay results.",
        "  • Inclusion: athlete-season enters a band if ≥1 individual result has",
        f"    WA points in [lo, lo+{width}).",
        f"  • Bands slide every {BAND_STEP} points (overlapping windows).",
        "  • Modeling: season PBs; same CV model search as outdoor time models",
        f"    (seed={CV_SEED}, folds={CV_FOLDS}).",
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

    lines.extend(
        [
            f"Band leaderboard (width={width}; require ≥4 pairs)",
            "-" * 50,
            f"{'Band':<12} {'Ath-seas':>9} {'Pairs':>6} {'Mean best CV':>14} {'Mean linear':>12}",
        ]
    )
    for s in ranked:
        lines.append(
            f"{s['band']:<12} {s['n_athlete_seasons']:>9} {s['pairs_evaluated']:>6} "
            f"{s['mean_best_cv']:>13.3f}s {s['mean_linear_cv']:>11.3f}s"
        )

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
        lines.extend(
            [
                "",
                "Most accurate band",
                "------------------",
                f"  {best['band']} WA points  →  mean best CV {best['mean_best_cv']:.3f}s "
                f"({best['pairs_evaluated']} pairs, {best['n_athlete_seasons']} athlete-seasons).",
            ]
        )
        if all_sum and all_sum["mean_best_cv"] is not None:
            d = best["mean_best_cv"] - all_sum["mean_best_cv"]
            lines.append(f"  vs unbanded all: Δ {d:+.3f}s")

        best_pairs = [r for r in result["band_rows"] if r["band"] == best["band"]]
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

        # Also report top bands with fuller pair coverage (≥12 of max 24)
        fuller = [s for s in ranked if s["pairs_evaluated"] >= 12]
        if fuller and fuller[0]["band"] != best["band"]:
            lines.extend(
                [
                    "",
                    "Best band among those with ≥12 pairs (broader coverage)",
                    "-" * 55,
                ]
            )
            fb = fuller[0]
            lines.append(
                f"  {fb['band']}: mean CV {fb['mean_best_cv']:.3f}s "
                f"({fb['pairs_evaluated']} pairs, {fb['n_athlete_seasons']} athlete-seasons)"
            )

    if len(ranked) >= 2:
        lines.extend(["", "Top 5 bands by mean best CV", "---------------------------"])
        for i, s in enumerate(ranked[:5], 1):
            lines.append(
                f"  {i}. {s['band']}: {s['mean_best_cv']:.3f}s "
                f"({s['pairs_evaluated']} pairs, {s['n_athlete_seasons']} athlete-seasons)"
            )

    lines.extend(
        [
            "",
            "Caveats:",
            "  • Athletes can appear in multiple bands (any-result membership).",
            "  • Low/high extreme bands have small n and few pairs.",
            "  • Mean CV across unequal pair sets favors bands that retain easy pairs;",
            "    prefer bands with ≥4 pairs and stable coverage.",
            "  • Indoor distance models use Mile/3000m (not outdoor 1500m/5000m).",
            "",
            "Source: indoor_analysis/analyze_indoor_point_bands.py",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_findings(result: dict, path: Path) -> None:
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
    fuller = [s for s in ranked if s["pairs_evaluated"] >= 12]

    lines = [
        "Indoor Point-Band Time Models — Findings (Width 200)",
        "===================================================",
        "",
        "Question: Which WA point bands of width 200 are best for indoor time models?",
        "",
        "Events: Sprints 60/200/400; Distance 800/Mile/3000. Relays excluded.",
        f"Bands: width {BAND_WIDTH}, starts every {BAND_STEP} points.",
        "",
    ]
    if ranked:
        best = ranked[0]
        lines.append(
            f"Most accurate (≥4 pairs): {best['band']}  "
            f"mean CV {best['mean_best_cv']:.3f}s "
            f"({best['pairs_evaluated']} pairs, {best['n_athlete_seasons']} athlete-seasons)"
        )
    if fuller:
        fb = fuller[0]
        lines.append(
            f"Best with ≥12 pairs:      {fb['band']}  "
            f"mean CV {fb['mean_best_cv']:.3f}s "
            f"({fb['pairs_evaluated']} pairs, {fb['n_athlete_seasons']} athlete-seasons)"
        )
    if all_sum and all_sum["mean_best_cv"] is not None:
        lines.append(
            f"Unbanded all:             mean CV {all_sum['mean_best_cv']:.3f}s "
            f"({all_sum['pairs_evaluated']} pairs)"
        )
    lines.append("")
    lines.append("Top 5 (require ≥4 pairs):")
    for i, s in enumerate(ranked[:5], 1):
        lines.append(
            f"  {i}. {s['band']}: {s['mean_best_cv']:.3f}s "
            f"({s['pairs_evaluated']} pairs)"
        )
    lines.extend(
        [
            "",
            "Recommendation: use the top width-200 band(s) above when building indoor",
            "equivalency / future-meet time models. Prefer the ≥12-pair band if you need",
            "stable coverage across most sprint and distance ordered pairs.",
            "",
            "See point_band_report_width_200.txt for the full leaderboard.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    profiles_by_gg: dict[tuple[str, str], dict[str, dict]] = {}
    global_max = 0.0
    for event_group, (folder, group, events) in GROUP_FOLDERS.items():
        for gender in ("Men", "Women"):
            profiles = load_athlete_season_records(folder, group, gender, events)
            profiles_by_gg[(gender, event_group)] = profiles
            for p in profiles.values():
                global_max = max(global_max, p["max_wa"])
            print(f"Loaded {gender} {event_group}: {len(profiles)} athlete-seasons")

    print(f"Global max WA among included results: {global_max:.1f}")
    print(f"Analyzing band width {BAND_WIDTH}...")
    result = analyze_width(BAND_WIDTH, profiles_by_gg, global_max)

    write_csv(OUTPUT_ROOT / "point_band_pair_results_width_200.csv", result["band_rows"])
    write_csv(OUTPUT_ROOT / "point_band_summary_width_200.csv", result["summary_rows"])
    write_csv(OUTPUT_ROOT / "point_band_cohort_width_200.csv", result["cohort_rows"])
    write_report(result, OUTPUT_ROOT / "point_band_report_width_200.txt")
    write_findings(result, OUTPUT_ROOT / "point_band_findings_width_200.txt")

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
            f"Best band: {ranked[0]['band']} mean CV {ranked[0]['mean_best_cv']:.3f}s "
            f"({ranked[0]['pairs_evaluated']} pairs)"
        )
    print(f"Wrote outputs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
