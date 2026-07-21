"""Rerun cross-event time model search filtered by season race volume.

Counts races per athlete-season within each event group (Sprints / Distance),
using the same discipline CSV sources as build_cross_event_time_models.py.
Compares model accuracy for athletes with exactly 4, 5, or 6 season results
vs. all athlete-seasons.

Outputs to time_models/model_search/race_count/
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
RELAYS_ROOT = PROJECT_ROOT / "relays_findings"
OUTPUT_ROOT = Path(__file__).resolve().parent / "race_count"
BASELINE_ALL_CSV = Path(__file__).resolve().parent / "best_time_models.csv"

sys.path.insert(0, str(RELAYS_ROOT))
sys.path.insert(0, str(TIME_MODELS_ROOT))
from relay_rq1_data import points_col, season_rows  # noqa: E402
from build_cross_event_time_models import (  # noqa: E402
    GROUP_CONFIG,
    linreg,
    parse_performance,
)
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    format_params,
)

RACE_COUNT_FILTERS = (4, 5, 6)
MIN_PAIR_N = 20


def load_athlete_season_pbs(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
    race_count: int | None = None,
) -> dict[str, dict[str, float]]:
    """athlete_season_key -> {event_name: best_time_seconds}.

    If race_count is set, only athlete-seasons with exactly that many valid
    results in this discipline folder are included.
    """
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    pcol = points_col(gender)

    # athlete-season -> result_id -> row info
    season_results: dict[str, dict[str, tuple[int, str, float]]] = defaultdict(dict)
    season_bests: dict[str, dict[str, float]] = defaultdict(dict)

    for year in ("2024", "2025", "2026"):
        for row in season_rows(folder, prefix, gender, year):
            event_id = int(row["running_event_id"])
            if event_id not in id_to_name:
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
            season_results[key][result_id] = (event_id, id_to_name[event_id], t)

    for key, results in season_results.items():
        if race_count is not None and len(results) != race_count:
            continue
        bests: dict[str, float] = {}
        for _, event_name, t in results.values():
            current = bests.get(event_name)
            if current is None or t < current:
                bests[event_name] = t
        if bests:
            season_bests[key] = bests

    return dict(season_bests)


def run_model_search(
    race_count: int | None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    comparison_rows: list[dict] = []
    best_rows: list[dict] = []
    wins_by_model: dict[str, int] = {}
    pair_summaries: list[dict] = []

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            bests = load_athlete_season_pbs(folder, prefix, gender, events, race_count)

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
                    chain = evaluate_chain_models(bests, "100m", "200m", "400m")
                    if chain:
                        extra.append(chain)
                    multi = evaluate_multivariate(bests, "100m", "200m", "400m")
                    if multi:
                        extra.append(multi)
                if event_group == "Distance" and from_ev == "800m" and to_ev == "5000m":
                    chain = evaluate_chain_models(bests, "800m", "1500m", "5000m")
                    if chain:
                        extra.append(chain)
                    multi = evaluate_multivariate(bests, "800m", "1500m", "5000m")
                    if multi:
                        extra.append(multi)

                all_results = results + extra
                winner = min(all_results, key=lambda r: r.cv_median_abs)
                wins_by_model[winner.name] = wins_by_model.get(winner.name, 0) + 1
                improve = baseline_cv - winner.cv_median_abs

                pair_summaries.append(
                    {
                        "race_count_filter": race_count if race_count is not None else "all",
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n_athlete_seasons": len(pairs),
                        "linear_cv_median_abs": round(baseline_cv, 4),
                        "best_model": winner.name,
                        "best_cv_median_abs": round(winner.cv_median_abs, 4),
                        "improvement_seconds": round(improve, 4),
                    }
                )

                for r in all_results:
                    comparison_rows.append(
                        {
                            "race_count_filter": race_count if race_count is not None else "all",
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "n_athlete_seasons": len(pairs),
                            "model": r.name,
                            "cv_median_abs_error": round(r.cv_median_abs, 4),
                            "cv_rmse": round(r.cv_rmse, 4),
                            "delta_vs_linear_cv_med": round(baseline_cv - r.cv_median_abs, 4),
                            "is_winner": r.name == winner.name,
                            "params_summary": format_params(r.name, r.params),
                        }
                    )

                best_rows.append(
                    {
                        "race_count_filter": race_count if race_count is not None else "all",
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n_athlete_seasons": len(pairs),
                        "best_model": winner.name,
                        "linear_cv_median_abs": round(baseline_cv, 4),
                        "best_cv_median_abs": round(winner.cv_median_abs, 4),
                        "improvement_seconds": round(improve, 4),
                        "params_json": json.dumps(winner.params),
                        "params_summary": format_params(winner.name, winner.params),
                        "notes": winner.notes,
                    }
                )

    summary = {
        "race_count_filter": race_count if race_count is not None else "all",
        "pairs_evaluated": len(best_rows),
        "wins_by_model": wins_by_model,
        "mean_best_cv": round(
            statistics.mean(r["best_cv_median_abs"] for r in best_rows), 4
        )
        if best_rows
        else None,
        "mean_linear_cv": round(
            statistics.mean(r["linear_cv_median_abs"] for r in best_rows), 4
        )
        if best_rows
        else None,
        "pairs_improved_over_linear": sum(
            1 for r in best_rows if r["improvement_seconds"] > 0.01
        ),
    }
    return comparison_rows, best_rows, summary


def load_pooled_baseline() -> dict[tuple[str, str, str, str], dict[str, float]]:
    """Original pooled-athlete baseline from best_time_models.csv."""
    out: dict[tuple[str, str, str, str], dict[str, float]] = {}
    if not BASELINE_ALL_CSV.exists():
        return out
    for row in csv.DictReader(open(BASELINE_ALL_CSV)):
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        out[key] = {
            "best_cv_median_abs": float(row["best_cv_median_abs"]),
            "linear_cv_median_abs": float(row["linear_cv_median_abs"]),
            "best_model": row["best_model"],
            "n_athletes": int(row["n_athletes"]),
        }
    return out


def write_comparison_report(
    summaries: list[dict],
    pair_summaries: dict[str | int, list[dict]],
    path: Path,
) -> None:
    pooled_baseline = load_pooled_baseline()
    lines = [
        "Cross-Event Time Models by Season Race Volume",
        "==============================================",
        "",
        "Question: Do time-prediction models become more accurate when restricted",
        "to athletes who competed in more races during a season?",
        "",
        "Method:",
        "  • Unit: athlete-season (not pooled across years).",
        "  • Race count: unique valid results in the discipline CSV for that event group",
        "    (same sources as build_cross_event_time_models.py).",
        f"  • Filters: all athlete-seasons, and exactly {', '.join(str(x) for x in RACE_COUNT_FILTERS)} races.",
        f"  • Model search: same 5-fold CV candidates as compare_time_model_candidates.py",
        f"    (seed={CV_SEED}, folds={CV_FOLDS}).",
        "  • Primary metric: CV median absolute error (seconds). Lower is better.",
        "",
        "Headline comparison (mean CV median |error| across evaluated pairs)",
        "----------------------------------------------------------------",
        f"{'Filter':<12} {'Pairs':>6} {'Mean linear':>12} {'Mean best':>12} {'Δ (linear-best)':>16}",
    ]

    for s in summaries:
        label = str(s["race_count_filter"])
        delta = (s["mean_linear_cv"] or 0) - (s["mean_best_cv"] or 0)
        lines.append(
            f"{label:<12} {s['pairs_evaluated']:>6} "
            f"{s['mean_linear_cv']:>11.3f}s {s['mean_best_cv']:>11.3f}s {delta:>15.3f}s"
        )

    lines.extend(
        [
            "",
            "Note: 'all' uses athlete-season units. The original pooled-athlete baseline",
            "(best mark per athlete across 2024–2026) is shown per pair below for reference.",
            "",
        ]
    )

    # Pair-level deltas: race count vs all athlete-seasons
    all_pairs = {
        (r["gender"], r["event_group"], r["from_event"], r["to_event"]): r
        for r in pair_summaries.get("all", [])
    }

    for rc in RACE_COUNT_FILTERS:
        lines.extend(
            [
                "",
                f"=== Exactly {rc} races vs all athlete-seasons ===",
                f"{'Gender':<7} {'Group':<9} {'Pair':<12} {'n(all)':>7} {'n({rc})':>7} "
                f"{'CV all':>8} {'CV {rc}':>8} {'Δ':>8} {'Better?':>8}".format(rc=rc),
            ]
        )
        rc_rows = pair_summaries.get(rc, [])
        deltas = []
        for row in sorted(rc_rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"])):
            key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
            base = all_pairs.get(key)
            if not base:
                continue
            delta = row["best_cv_median_abs"] - base["best_cv_median_abs"]
            deltas.append(delta)
            pair = f"{row['from_event']}->{row['to_event']}"
            better = "yes" if delta < -0.01 else ("no" if delta > 0.01 else "~same")
            lines.append(
                f"{row['gender']:<7} {row['event_group']:<9} {pair:<12} "
                f"{base['n_athlete_seasons']:>7} {row['n_athlete_seasons']:>7} "
                f"{base['best_cv_median_abs']:>7.3f}s {row['best_cv_median_abs']:>7.3f}s "
                f"{delta:>+7.3f}s {better:>8}"
            )
        if deltas:
            improved = sum(1 for d in deltas if d < -0.01)
            worsened = sum(1 for d in deltas if d > 0.01)
            lines.extend(
                [
                    "",
                    f"  Pairs more accurate at {rc} races: {improved}/{len(deltas)}",
                    f"  Pairs less accurate at {rc} races: {worsened}/{len(deltas)}",
                    f"  Mean Δ CV med|err| ({rc} minus all): {statistics.mean(deltas):+.3f}s",
                    f"  Median Δ: {statistics.median(deltas):+.3f}s",
                ]
            )

    lines.extend(["", "Detailed pair results by filter", "-----------------------------"])
    for label, rows in sorted(pair_summaries.items(), key=lambda x: str(x[0])):
        lines.extend(["", f"Filter: {label}", "-" * 40])
        for row in sorted(rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"])):
            pair = f"{row['from_event']} -> {row['to_event']}"
            pooled = pooled_baseline.get(
                (row["gender"], row["event_group"], row["from_event"], row["to_event"])
            )
            pooled_note = ""
            if pooled:
                pooled_note = (
                    f" | pooled-athlete baseline: {pooled['best_cv_median_abs']:.3f}s "
                    f"(n={pooled['n_athletes']})"
                )
            lines.append(
                f"  {row['gender']} {row['event_group']} {pair}  "
                f"n={row['n_athlete_seasons']}  "
                f"linear={row['linear_cv_median_abs']:.3f}s  "
                f"best={row['best_model']} {row['best_cv_median_abs']:.3f}s"
                f"{pooled_note}"
            )

    # Insights section
    lines.extend(["", "Interpretation", "--------------"])
    all_summary = next(s for s in summaries if s["race_count_filter"] == "all")
    rc_summaries = [s for s in summaries if s["race_count_filter"] in RACE_COUNT_FILTERS]

    if rc_summaries:
        best_rc = min(rc_summaries, key=lambda s: s["mean_best_cv"] or float("inf"))
        worst_rc = max(rc_summaries, key=lambda s: s["mean_best_cv"] or 0)
        lines.append(
            f"Among {', '.join(str(x) for x in RACE_COUNT_FILTERS)}-race filters, "
            f"{best_rc['race_count_filter']} races had the lowest mean CV error "
            f"({best_rc['mean_best_cv']:.3f}s vs all athlete-seasons {all_summary['mean_best_cv']:.3f}s)."
        )
        for s in rc_summaries:
            delta_vs_all = (s["mean_best_cv"] or 0) - (all_summary["mean_best_cv"] or 0)
            direction = "more accurate" if delta_vs_all < -0.01 else (
                "less accurate" if delta_vs_all > 0.01 else "about the same"
            )
            lines.append(
                f"  • {s['race_count_filter']} races: mean best CV = {s['mean_best_cv']:.3f}s "
                f"({direction} than all-season average by {abs(delta_vs_all):.3f}s)"
            )

    lines.extend(
        [
            "",
            "Caveats:",
            "  • Smaller sample sizes at fixed race counts reduce model stability.",
            "  • Athletes with more races may be more committed / coached — selection effect.",
            "  • Race count here is total discipline results, not per-event race count.",
            "",
            "Source: time_models/model_search/compare_time_models_by_race_count.py",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_comparison: list[dict] = []
    all_best: list[dict] = []
    summaries: list[dict] = []
    pair_summaries: dict[str | int, list[dict]] = {}

    filters: list[int | None] = [None, *list(RACE_COUNT_FILTERS)]
    for rc in filters:
        label = "all" if rc is None else rc
        print(f"Running model search for race_count={label}...")
        comparison, best, summary = run_model_search(rc)
        all_comparison.extend(comparison)
        all_best.extend(best)
        summaries.append(summary)
        pair_summaries[label] = [
            {
                "race_count_filter": label,
                "gender": r["gender"],
                "event_group": r["event_group"],
                "from_event": r["from_event"],
                "to_event": r["to_event"],
                "n_athlete_seasons": r["n_athlete_seasons"],
                "linear_cv_median_abs": r["linear_cv_median_abs"],
                "best_model": r["best_model"],
                "best_cv_median_abs": r["best_cv_median_abs"],
                "improvement_seconds": r["improvement_seconds"],
            }
            for r in best
        ]

    if all_comparison:
        with open(OUTPUT_ROOT / "model_comparison_by_race_count.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_comparison[0].keys()))
            w.writeheader()
            w.writerows(all_comparison)

    if all_best:
        with open(OUTPUT_ROOT / "best_time_models_by_race_count.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_best[0].keys()))
            w.writeheader()
            w.writerows(all_best)

    with open(OUTPUT_ROOT / "summary_by_race_count.csv", "w", newline="") as f:
        rows = [
            {
                "race_count_filter": s["race_count_filter"],
                "pairs_evaluated": s["pairs_evaluated"],
                "mean_linear_cv_median_abs": s["mean_linear_cv"],
                "mean_best_cv_median_abs": s["mean_best_cv"],
                "pairs_improved_over_linear": s["pairs_improved_over_linear"],
            }
            for s in summaries
        ]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    write_comparison_report(summaries, pair_summaries, OUTPUT_ROOT / "race_count_analysis_report.txt")
    print(f"Wrote race-count model search results to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
