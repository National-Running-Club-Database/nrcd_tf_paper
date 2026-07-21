"""Time-model accuracy for athlete-seasons with 6–10 races.

Compares:
  • all athlete-seasons (baseline)
  • exactly 6, 7, 8, 9, 10 races
  • combined 6–10 races

Outputs to time_models/6-10 races_time_models/
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
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

from build_cross_event_time_models import GROUP_CONFIG, linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    format_params,
    fit_linear,
    pred_linear,
    fit_log_linear,
    pred_log_linear,
    fit_median_ratio,
    pred_median_ratio,
    fit_ratio_linear,
    pred_ratio_linear,
    fit_quadratic,
    pred_quadratic,
    fit_robust_trimmed,
    pred_binned_ratio,
    metrics,
)
from compare_time_models_by_race_count import load_athlete_season_pbs  # noqa: E402

MIN_PAIR_N = 20
EXACT_COUNTS = (6, 7, 8, 9, 10)
COMBINED_COUNTS = frozenset({6, 7, 8, 9, 10})
COMBINED_LABEL = "6_to_10"

PRED = {
    "linear_ols": (fit_linear, pred_linear),
    "log_linear": (fit_log_linear, pred_log_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "binned_ratio": (None, pred_binned_ratio),
}


def load_pbs_multi(
    folder: str,
    prefix: str,
    gender: str,
    events: dict[str, int],
    race_filter: int | frozenset[int] | None,
) -> dict[str, dict[str, float]]:
    if race_filter is None or isinstance(race_filter, int):
        return load_athlete_season_pbs(folder, prefix, gender, events, race_filter)
    merged: dict[str, dict[str, float]] = {}
    for rc in sorted(race_filter):
        for key, bests in load_athlete_season_pbs(folder, prefix, gender, events, rc).items():
            if key not in merged:
                merged[key] = bests
    return merged


def filter_label(race_filter: int | frozenset[int] | None) -> str:
    if race_filter is None:
        return "all"
    if isinstance(race_filter, frozenset):
        return COMBINED_LABEL
    return str(race_filter)


def run_model_search(race_filter: int | frozenset[int] | None) -> tuple[list[dict], list[dict], dict]:
    comparison_rows: list[dict] = []
    best_rows: list[dict] = []
    wins_by_model: dict[str, int] = {}
    label = filter_label(race_filter)

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            bests = load_pbs_multi(folder, prefix, gender, events, race_filter)
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

                all_results = results + extra
                winner = min(all_results, key=lambda r: r.cv_median_abs)
                wins_by_model[winner.name] = wins_by_model.get(winner.name, 0) + 1
                improve = baseline_cv - winner.cv_median_abs

                for r in all_results:
                    comparison_rows.append(
                        {
                            "race_count_filter": label,
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
                        "race_count_filter": label,
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
        "race_count_filter": label,
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


def cohort_sizes() -> list[dict]:
    rows = []
    for eg, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            for rc in range(1, 15):
                n = len(load_athlete_season_pbs(folder, prefix, gender, events, rc))
                if n:
                    rows.append(
                        {
                            "gender": gender,
                            "event_group": eg,
                            "race_count": rc,
                            "n_athlete_seasons": n,
                        }
                    )
            n_610 = sum(
                len(load_athlete_season_pbs(folder, prefix, gender, events, rc))
                for rc in EXACT_COUNTS
            )
            rows.append(
                {
                    "gender": gender,
                    "event_group": eg,
                    "race_count": COMBINED_LABEL,
                    "n_athlete_seasons": n_610,
                }
            )
    return rows


def human_formula(model: str, params: dict, from_ev: str, to_ev: str) -> str:
    if model in ("linear_ols", "robust_trimmed_linear"):
        a, b = params["intercept"], params["slope"]
        if abs(a) < 1e-6:
            return f"{to_ev} = {b:.3f} × {from_ev}"
        if a >= 0:
            return f"{to_ev} = {a:.3f} + {b:.3f} × {from_ev}"
        return f"{to_ev} = {b:.3f} × {from_ev} − {abs(a):.3f}"
    if model == "log_linear":
        return f"{to_ev} = exp({params['intercept']:.3f} + {params['slope']:.3f}×log({from_ev}))"
    if model == "median_ratio":
        return f"{to_ev} = {params['ratio']:.3f} × {from_ev}"
    if model == "ratio_linear":
        a, b = params["intercept"], params["slope"]
        sign = "+" if b >= 0 else "−"
        return f"{to_ev} = {from_ev} × ({a:.3f} {sign} {abs(b):.6f}×{from_ev})"
    if model == "quadratic":
        a, b, c = params["intercept"], params["slope"], params["quad"]
        if b < 0 and c < 0:
            return f"{to_ev} = {a:.3f} − {abs(b):.3f}×{from_ev} − {abs(c):.6f}×{from_ev}²"
        if b < 0:
            return f"{to_ev} = {a:.3f} − {abs(b):.3f}×{from_ev} + {c:.6f}×{from_ev}²"
        if c < 0:
            return f"{to_ev} = {a:.3f} + {b:.3f}×{from_ev} − {abs(c):.6f}×{from_ev}²"
        return f"{to_ev} = {a:.3f} + {b:.3f}×{from_ev} + {c:.6f}×{from_ev}²"
    if model == "binned_ratio":
        return f"{to_ev} = {from_ev} × ratio_bin({from_ev})"
    if model == "multivariate_ols":
        return (
            f"{to_ev} = {params['intercept']:.2f} + {params['coef_a']:.3f}×{params['event_a']} "
            f"+ {params['coef_b']:.3f}×{params['event_b']}"
        )
    return format_params(model, params)


def write_formula_report(best_rows: list[dict], path: Path) -> None:
    rows = [r for r in best_rows if r["race_count_filter"] == COMBINED_LABEL]
    if not rows:
        path.write_text("No pairs met n≥20 for the 6–10 race combined cohort.\n")
        return

    mean_cv = statistics.mean(r["best_cv_median_abs"] for r in rows)
    lines = [
        "Cross-Event Time Models — 6 to 10 Races Per Season (Sprints & Distance)",
        "=" * 72,
        "",
        "Method:",
        "  • Data: relay-inclusive outdoor CSVs, 2024–2026, results on or after March 1.",
        "  • Population: athlete-seasons with 6, 7, 8, 9, or 10 valid results in the",
        "    event group's discipline CSV.",
        "  • Athlete-season PB: fastest valid mark per event within that season (WA > 0).",
        "  • Model selection: 5-fold CV across candidate families; lowest CV median |error|.",
        f"  • Minimum n = {MIN_PAIR_N} athlete-season pairs to report a model.",
        "",
        "Use:",
        "  • Apply when the athlete has 6–10 season races in the event group.",
        f"  • Mean CV median |error| across reported pairs: {mean_cv:.2f}s.",
        "  • Predict time, then convert to WA in the TARGET event table.",
        "",
        f"Coverage: {len(rows)} event pairs met n ≥ {MIN_PAIR_N}.",
        "",
    ]

    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            section_rows = [
                r for r in rows if r["gender"] == gender and r["event_group"] == group
            ]
            section = f"{group} — {gender}"
            lines.extend(["", section, "-" * len(section), ""])
            if not section_rows:
                lines.append(f"No pairs met n ≥ {MIN_PAIR_N}.")
                lines.append("")
                continue
            folder, prefix, events = GROUP_CONFIG[group]
            bests = load_pbs_multi(folder, prefix, gender, events, COMBINED_COUNTS)
            for row in sorted(section_rows, key=lambda r: (r["from_event"], r["to_event"])):
                from_ev, to_ev = row["from_event"], row["to_event"]
                pairs = [
                    (b[from_ev], b[to_ev])
                    for b in bests.values()
                    if from_ev in b and to_ev in b
                ]
                _, _, r, _, _ = linreg(pairs)
                params = json.loads(row["params_json"])
                formula = human_formula(row["best_model"], params, from_ev, to_ev)
                r_med, r_p25, r_p75 = ratio_quartiles(pairs)
                lines.append(
                    f"{from_ev} -> {to_ev}  (n={len(pairs)} athlete-seasons, r={r:.3f})"
                )
                lines.append(f"  Model: {row['best_model']}")
                lines.append(f"  Formula: {formula}")
                lines.append(f"  CV median |error|: {row['best_cv_median_abs']:.3f}s")
                if row["best_model"] == "binned_ratio":
                    edges = params.get("edges", [])
                    ratios = params.get("ratios", [])
                    lines.append(
                        f"  Binned lookup for {from_ev} -> {to_ev} (nearest {from_ev} edge):"
                    )
                    for edge, ratio in zip(edges, ratios):
                        lines.append(f"    nearest {from_ev} edge {edge:.2f}s -> ratio {ratio:.3f}")
                lines.append(
                    f"  Ratio {to_ev}/{from_ev}: median {r_med:.3f} "
                    f"(IQR {r_p25:.3f}–{r_p75:.3f})"
                )
                lines.append(
                    f"  Median times: {from_ev} {statistics.median(p[0] for p in pairs):.2f}s, "
                    f"{to_ev} {statistics.median(p[1] for p in pairs):.2f}s"
                )
                lines.append("")

    lines.extend(["Pair summary", "------------", ""])
    lines.append(f"  {'From':<8} {'To':<8} {'n':>5} {'CV':>10}  Formula")
    for row in sorted(
        rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"], r["to_event"])
    ):
        params = json.loads(row["params_json"])
        formula = human_formula(row["best_model"], params, row["from_event"], row["to_event"])
        lines.append(
            f"  {row['from_event']:<8} {row['to_event']:<8} {row['n_athlete_seasons']:>5} "
            f"{row['best_cv_median_abs']:>9.3f}s  {formula}"
        )
    lines.append("")
    lines.append("Source: time_models/6-10 races_time_models/analyze_6_to_10_races_time_models.py")
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_analysis_report(
    summaries: list[dict],
    best_by_filter: dict[str, list[dict]],
    sizes: list[dict],
    path: Path,
) -> None:
    lines = [
        "Time Models for Athletes with 6–10 Season Races",
        "===============================================",
        "",
        "Question: If we include athletes with 6–10 races in a season (instead of",
        "stopping at exactly 6), do cross-event time models become more accurate?",
        "",
        "Method:",
        "  • Unit: athlete-season.",
        "  • Race count: unique valid results in the discipline CSV (Sprints / Distance).",
        "  • Filters: all; exactly 6, 7, 8, 9, 10; combined 6–10.",
        f"  • Same 5-fold CV model search (seed={CV_SEED}, folds={CV_FOLDS}).",
        f"  • Minimum n={MIN_PAIR_N} pairs to evaluate a model.",
        "  • Metric: CV median absolute error in seconds (lower is better).",
        "",
        "Cohort sizes (athlete-seasons)",
        "-----------------------------",
    ]
    for gender in ("Men", "Women"):
        for eg in ("Sprints", "Distance"):
            parts = []
            for rc in EXACT_COUNTS:
                n = next(
                    (
                        r["n_athlete_seasons"]
                        for r in sizes
                        if r["gender"] == gender
                        and r["event_group"] == eg
                        and r["race_count"] == rc
                    ),
                    0,
                )
                parts.append(f"{rc}={n}")
            n610 = next(
                (
                    r["n_athlete_seasons"]
                    for r in sizes
                    if r["gender"] == gender
                    and r["event_group"] == eg
                    and r["race_count"] == COMBINED_LABEL
                ),
                0,
            )
            lines.append(f"  {gender} {eg}: {', '.join(parts)}  | 6–10={n610}")

    lines.extend([
        "",
        "Headline: mean best-model CV median |error|",
        "-------------------------------------------",
        f"{'Filter':<12} {'Pairs':>6} {'Mean linear':>12} {'Mean best':>12}",
    ])
    order = ["all", "6", "7", "8", "9", "10", COMBINED_LABEL]
    by_label = {str(s["race_count_filter"]): s for s in summaries}
    for lab in order:
        s = by_label.get(lab)
        if not s or s["mean_best_cv"] is None:
            lines.append(f"{lab:<12} {0:>6} {'—':>12} {'—':>12}  (no pairs with n≥{MIN_PAIR_N})")
            continue
        lines.append(
            f"{lab:<12} {s['pairs_evaluated']:>6} "
            f"{s['mean_linear_cv']:>11.3f}s {s['mean_best_cv']:>11.3f}s"
        )

    all_rows = {
        (r["gender"], r["event_group"], r["from_event"], r["to_event"]): r
        for r in best_by_filter.get("all", [])
    }
    six_rows = {
        (r["gender"], r["event_group"], r["from_event"], r["to_event"]): r
        for r in best_by_filter.get("6", [])
    }
    comb_rows = {
        (r["gender"], r["event_group"], r["from_event"], r["to_event"]): r
        for r in best_by_filter.get(COMBINED_LABEL, [])
    }

    # Matched comparison: pairs present in both 6 and 6_to_10
    lines.extend([
        "",
        "Matched pairs: exactly 6 vs combined 6–10",
        "----------------------------------------",
        f"{'Pair':<34} {'n6':>4} {'n6-10':>6} {'CV6':>8} {'CV6-10':>8} {'Δ':>8} Better?",
    ])
    matched_deltas = []
    for key in sorted(set(six_rows) & set(comb_rows)):
        a, b = six_rows[key], comb_rows[key]
        delta = b["best_cv_median_abs"] - a["best_cv_median_abs"]
        matched_deltas.append(delta)
        pair = f"{a['gender']} {a['event_group']} {a['from_event']}->{a['to_event']}"
        better = "6-10" if delta < -0.01 else ("6" if delta > 0.01 else "~same")
        lines.append(
            f"{pair:<34} {a['n_athlete_seasons']:>4} {b['n_athlete_seasons']:>6} "
            f"{a['best_cv_median_abs']:>7.3f}s {b['best_cv_median_abs']:>7.3f}s "
            f"{delta:>+7.3f}s {better}"
        )
    if matched_deltas:
        lines.append("")
        lines.append(
            f"  6–10 more accurate: {sum(1 for d in matched_deltas if d < -0.01)}/{len(matched_deltas)}"
        )
        lines.append(
            f"  Exact-6 more accurate: {sum(1 for d in matched_deltas if d > 0.01)}/{len(matched_deltas)}"
        )
        lines.append(f"  Mean Δ (6–10 minus 6): {statistics.mean(matched_deltas):+.3f}s")
        lines.append(f"  Median Δ: {statistics.median(matched_deltas):+.3f}s")

    # 6–10 vs all
    lines.extend([
        "",
        "Matched pairs: combined 6–10 vs all athlete-seasons",
        "---------------------------------------------------",
        f"{'Pair':<34} {'n_all':>5} {'n6-10':>6} {'CV all':>8} {'CV6-10':>8} {'Δ':>8} Better?",
    ])
    all_deltas = []
    for key in sorted(set(all_rows) & set(comb_rows)):
        a, b = all_rows[key], comb_rows[key]
        delta = b["best_cv_median_abs"] - a["best_cv_median_abs"]
        all_deltas.append(delta)
        pair = f"{a['gender']} {a['event_group']} {a['from_event']}->{a['to_event']}"
        better = "6-10" if delta < -0.01 else ("all" if delta > 0.01 else "~same")
        lines.append(
            f"{pair:<34} {a['n_athlete_seasons']:>5} {b['n_athlete_seasons']:>6} "
            f"{a['best_cv_median_abs']:>7.3f}s {b['best_cv_median_abs']:>7.3f}s "
            f"{delta:>+7.3f}s {better}"
        )
    if all_deltas:
        lines.append("")
        lines.append(
            f"  6–10 more accurate than all: {sum(1 for d in all_deltas if d < -0.01)}/{len(all_deltas)}"
        )
        lines.append(
            f"  All more accurate than 6–10: {sum(1 for d in all_deltas if d > 0.01)}/{len(all_deltas)}"
        )
        lines.append(f"  Mean Δ (6–10 minus all): {statistics.mean(all_deltas):+.3f}s")
        lines.append(f"  Median Δ: {statistics.median(all_deltas):+.3f}s")

    # Exact count detail
    for rc in EXACT_COUNTS:
        rows = best_by_filter.get(str(rc), [])
        lines.extend(["", f"Exactly {rc} races — pair results", "-" * 34])
        if not rows:
            lines.append(f"  No pairs with n≥{MIN_PAIR_N} (sample too small).")
            continue
        for row in sorted(rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"])):
            lines.append(
                f"  {row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']} "
                f"n={row['n_athlete_seasons']}  best={row['best_model']} "
                f"CV={row['best_cv_median_abs']:.3f}s"
            )

    lines.extend(["", "Interpretation", "--------------"])
    s_all = by_label.get("all")
    s6 = by_label.get("6")
    s610 = by_label.get(COMBINED_LABEL)
    if s610 and s610["mean_best_cv"] is not None and s_all and s_all["mean_best_cv"] is not None:
        d_all = s610["mean_best_cv"] - s_all["mean_best_cv"]
        lines.append(
            f"Combined 6–10 mean best CV = {s610['mean_best_cv']:.3f}s "
            f"({s610['pairs_evaluated']} pairs) vs all = {s_all['mean_best_cv']:.3f}s "
            f"(Δ {d_all:+.3f}s)."
        )
    if s610 and s6 and s610["mean_best_cv"] is not None and s6["mean_best_cv"] is not None:
        d6 = s610["mean_best_cv"] - s6["mean_best_cv"]
        lines.append(
            f"Vs exact-6 mean best CV {s6['mean_best_cv']:.3f}s ({s6['pairs_evaluated']} pairs): "
            f"Δ {d6:+.3f}s on headline averages (pair sets differ — see matched table)."
        )
    if matched_deltas:
        if statistics.mean(matched_deltas) < -0.05:
            lines.append(
                "Verdict: YES — on matched pairs, pooling 6–10 races improves accuracy "
                "vs exact-6."
            )
        elif statistics.mean(matched_deltas) > 0.05:
            lines.append(
                "Verdict: NO — on matched pairs, exact-6 is more accurate than pooling 6–10; "
                "adding 7–10 race seasons does not help (and can dilute relationships)."
            )
        else:
            lines.append(
                "Verdict: MIXED / SMALL — combined 6–10 is similar to exact-6 on matched pairs; "
                "main benefit is more pairs / coverage rather than large accuracy gains."
            )
    lines.append(
        "Note: exact 7–10 alone rarely meet n≥20; most of the 6–10 signal comes from "
        "athletes with 6–8 races."
    )
    lines.extend([
        "",
        "Caveats:",
        "  • Small n at 9–10 races; unstable if analyzed alone.",
        "  • Athlete-seasons with more races may differ systematically (selection).",
        "  • Mean CV across unequal pair sets is not as informative as matched-pair Δ.",
        "",
        "Output files:",
        "  race_count_6_to_10_report.txt",
        "  race_count_6_to_10_findings.txt",
        "  summary_by_race_count.csv",
        "  best_time_models_by_race_count.csv",
        "  model_comparison_by_race_count.csv",
        "  cohort_sizes.csv",
        "  time_models_6_to_10_races_formulas.txt",
        "",
        "Source: time_models/6-10 races_time_models/analyze_6_to_10_races_time_models.py",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_findings(
    summaries: list[dict],
    best_by_filter: dict[str, list[dict]],
    path: Path,
) -> None:
    by_label = {str(s["race_count_filter"]): s for s in summaries}
    six = {
        (r["gender"], r["event_group"], r["from_event"], r["to_event"]): r
        for r in best_by_filter.get("6", [])
    }
    comb = {
        (r["gender"], r["event_group"], r["from_event"], r["to_event"]): r
        for r in best_by_filter.get(COMBINED_LABEL, [])
    }
    deltas = []
    for key in set(six) & set(comb):
        deltas.append(comb[key]["best_cv_median_abs"] - six[key]["best_cv_median_abs"])

    lines = [
        "6–10 Race Time Models — Findings",
        "================================",
        "",
        "Question: Do time models become more accurate using athletes with 6–10",
        "season races (vs all-season or exact-6 only)?",
        "",
    ]
    for lab in ("all", "6", "7", "8", "9", "10", COMBINED_LABEL):
        s = by_label.get(lab)
        if s and s["mean_best_cv"] is not None:
            lines.append(
                f"  {lab}: {s['pairs_evaluated']} pairs, mean best CV {s['mean_best_cv']:.3f}s"
            )
        else:
            lines.append(f"  {lab}: no pairs with n≥{MIN_PAIR_N}")
    lines.append("")
    if deltas:
        lines.append(
            f"Matched 6 vs 6–10: {sum(1 for d in deltas if d < -0.01)}/{len(deltas)} "
            f"favor 6–10; mean Δ {statistics.mean(deltas):+.3f}s"
        )
    s610 = by_label.get(COMBINED_LABEL)
    s_all = by_label.get("all")
    if s610 and s_all and s610["mean_best_cv"] and s_all["mean_best_cv"]:
        lines.append(
            f"6–10 vs all headline mean: {s610['mean_best_cv']:.3f}s vs "
            f"{s_all['mean_best_cv']:.3f}s (Δ {s610['mean_best_cv']-s_all['mean_best_cv']:+.3f}s)"
        )
    lines.extend([
        "",
        "See race_count_6_to_10_report.txt for full tables.",
        "Source: analyze_6_to_10_races_time_models.py",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        (OUTPUT_ROOT / name).write_text("")
        return
    with open(OUTPUT_ROOT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    filters: list[int | frozenset[int] | None] = [None, *list(EXACT_COUNTS), COMBINED_COUNTS]
    all_comparison: list[dict] = []
    all_best: list[dict] = []
    summaries: list[dict] = []
    best_by_filter: dict[str, list[dict]] = {}

    for rf in filters:
        label = filter_label(rf)
        print(f"Running model search for race_count={label}...")
        comparison, best, summary = run_model_search(rf)
        all_comparison.extend(comparison)
        all_best.extend(best)
        summaries.append(summary)
        best_by_filter[label] = best

    sizes = cohort_sizes()
    write_csv("cohort_sizes.csv", sizes)
    write_csv("model_comparison_by_race_count.csv", all_comparison)
    write_csv("best_time_models_by_race_count.csv", all_best)
    write_csv(
        "summary_by_race_count.csv",
        [
            {
                "race_count_filter": s["race_count_filter"],
                "pairs_evaluated": s["pairs_evaluated"],
                "mean_linear_cv_median_abs": s["mean_linear_cv"],
                "mean_best_cv_median_abs": s["mean_best_cv"],
                "pairs_improved_over_linear": s["pairs_improved_over_linear"],
            }
            for s in summaries
        ],
    )

    write_analysis_report(
        summaries,
        best_by_filter,
        sizes,
        OUTPUT_ROOT / "race_count_6_to_10_report.txt",
    )
    write_findings(summaries, best_by_filter, OUTPUT_ROOT / "race_count_6_to_10_findings.txt")
    write_formula_report(all_best, OUTPUT_ROOT / "time_models_6_to_10_races_formulas.txt")

    print(f"Wrote results to {OUTPUT_ROOT}")
    for s in summaries:
        print(
            f"  {s['race_count_filter']}: pairs={s['pairs_evaluated']} "
            f"mean_best_cv={s['mean_best_cv']}"
        )


if __name__ == "__main__":
    main()
