"""Test whether best_is_from + best_event + bal/spec improve 6-race time models.

Population: athlete-seasons with exactly 6 races.

Note: best_is_from is implied by best_event (best_event == from_event ↔ best_is_from).
The informative three-factor request therefore reduces to bal/spec × best_event,
with best_is_from as a coarser nested label.

Strategies compared via 5-fold CV routing:
  pooled
  bal_spec
  best_is_from
  best_event
  bal_x_best_is_from
  bal_x_best_event          ← primary "know all three" model
  hierarchical_all_three    ← try finest cell first, then fall back
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
NEW_FACTORS_ROOT = Path(__file__).resolve().parents[1]
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
SPEC_ROOT = TIME_MODELS_ROOT / "specialized_time_models"

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(NEW_FACTORS_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    evaluate_pair,
    fit_linear,
    fit_log_linear,
    fit_median_ratio,
    fit_quadratic,
    fit_ratio_linear,
    fit_robust_trimmed,
    format_params,
    metrics,
    pred_binned_ratio,
    pred_linear,
    pred_log_linear,
    pred_median_ratio,
    pred_quadratic,
    pred_ratio_linear,
)
from analyze_specialization_time_models import specialization_label  # noqa: E402
from analyze_new_factors import load_rich_profiles, RichProfile  # noqa: E402

MIN_COHORT_N = 12
MIN_REPORT_N = 20
RACE_COUNTS = [6]

FIT_PRED: dict[str, tuple[Callable | None, Callable | None]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}

STRATEGIES = [
    "pooled",
    "bal_spec",
    "best_is_from",
    "best_event",
    "bal_x_best_is_from",
    "bal_x_best_event",
    "hierarchical_all_three",
]


def fit_best(pairs: list[tuple[float, float]]):
    if len(pairs) < MIN_COHORT_N:
        return None
    winner = min(evaluate_pair(pairs), key=lambda r: r.cv_median_abs)
    fit_fn, pred_fn = FIT_PRED.get(winner.name, (fit_linear, pred_linear))
    if fit_fn is None or pred_fn is None:
        return None
    return winner.name, fit_fn(pairs), pred_fn


def predict_with(fit, x: float, fallback) -> float:
    if fit is None:
        return fallback(x)
    _, params, pred_fn = fit
    return pred_fn(x, params)


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
        if c < 0:
            return f"{to_ev} = {a:.3f} + {b:.3f}×{from_ev} − {abs(c):.6f}×{from_ev}²"
        return f"{to_ev} = {a:.3f} + {b:.3f}×{from_ev} + {c:.6f}×{from_ev}²"
    if model == "binned_ratio":
        return f"{to_ev} = {from_ev} × ratio_bin({from_ev})"
    return format_params(model, params)


def cv_strategies(
    records: list[tuple[float, float, str, str, str]],
) -> dict[str, float] | None:
    """records: (x, y, bal_spec, best_is_from, best_event)"""
    if len(records) < MIN_REPORT_N:
        return None
    n = len(records)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)

    buckets: dict[str, tuple[list[float], list[float]]] = {s: ([], []) for s in STRATEGIES}

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else n
        test_idx = set(idx[start:end])
        train = [records[i] for i in range(n) if i not in test_idx]
        test = [records[i] for i in test_idx]
        if len(train) < 10:
            continue

        pooled = fit_best([(a, b) for a, b, _, _, _ in train])
        if pooled is None:
            continue
        _, pp, ppred = pooled

        def pooled_fn(x: float) -> float:
            return ppred(x, pp)

        def fits_for(key_fn) -> dict:
            out = {}
            keys = {key_fn(r) for r in train}
            for key in keys:
                subset = [(a, b) for a, b, *rest in train if key_fn((a, b, *rest)) == key]
                fit = fit_best(subset)
                if fit:
                    out[key] = fit
            return out

        bs_fits = fits_for(lambda r: r[2])
        bif_fits = fits_for(lambda r: r[3])
        be_fits = fits_for(lambda r: r[4])
        bx_bif_fits = fits_for(lambda r: (r[2], r[3]))
        bx_be_fits = fits_for(lambda r: (r[2], r[4]))

        for x, y, bs, bif, be in test:
            buckets["pooled"][0].append(y)
            buckets["pooled"][1].append(pooled_fn(x))

            buckets["bal_spec"][0].append(y)
            buckets["bal_spec"][1].append(predict_with(bs_fits.get(bs), x, pooled_fn))

            buckets["best_is_from"][0].append(y)
            buckets["best_is_from"][1].append(predict_with(bif_fits.get(bif), x, pooled_fn))

            buckets["best_event"][0].append(y)
            buckets["best_event"][1].append(predict_with(be_fits.get(be), x, pooled_fn))

            buckets["bal_x_best_is_from"][0].append(y)
            if (bs, bif) in bx_bif_fits:
                buckets["bal_x_best_is_from"][1].append(
                    predict_with(bx_bif_fits[(bs, bif)], x, pooled_fn)
                )
            elif bif in bif_fits:
                buckets["bal_x_best_is_from"][1].append(
                    predict_with(bif_fits[bif], x, pooled_fn)
                )
            elif bs in bs_fits:
                buckets["bal_x_best_is_from"][1].append(
                    predict_with(bs_fits[bs], x, pooled_fn)
                )
            else:
                buckets["bal_x_best_is_from"][1].append(pooled_fn(x))

            buckets["bal_x_best_event"][0].append(y)
            if (bs, be) in bx_be_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(bx_be_fits[(bs, be)], x, pooled_fn)
                )
            elif be in be_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(be_fits[be], x, pooled_fn)
                )
            elif bif in bif_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(bif_fits[bif], x, pooled_fn)
                )
            elif bs in bs_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(bs_fits[bs], x, pooled_fn)
                )
            else:
                buckets["bal_x_best_event"][1].append(pooled_fn(x))

            # hierarchical: bal×best_event → bal×best_is_from → best_event →
            # best_is_from → bal_spec → pooled
            buckets["hierarchical_all_three"][0].append(y)
            if (bs, be) in bx_be_fits:
                pred = predict_with(bx_be_fits[(bs, be)], x, pooled_fn)
            elif (bs, bif) in bx_bif_fits:
                pred = predict_with(bx_bif_fits[(bs, bif)], x, pooled_fn)
            elif be in be_fits:
                pred = predict_with(be_fits[be], x, pooled_fn)
            elif bif in bif_fits:
                pred = predict_with(bif_fits[bif], x, pooled_fn)
            elif bs in bs_fits:
                pred = predict_with(bs_fits[bs], x, pooled_fn)
            else:
                pred = pooled_fn(x)
            buckets["hierarchical_all_three"][1].append(pred)

    out = {}
    for name, (actual, pred) in buckets.items():
        if not actual:
            return None
        out[name], _ = metrics(actual, pred)
    return out


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    strategy_rows: list[dict] = []
    cell_rows: list[dict] = []
    formula_rows: list[dict] = []
    profile_rows: list[dict] = []

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            profiles = load_rich_profiles(folder, prefix, gender, events, RACE_COUNTS)
            for p in profiles.values():
                p.event_group = event_group
                profile_rows.append(
                    {
                        "athlete_season_key": p.key,
                        "gender": gender,
                        "event_group": event_group,
                        "race_count": p.race_count,
                        "bal_spec": specialization_label(p.wa_spread),
                        "wa_spread": round(p.wa_spread, 1),
                        "best_event": p.best_event,
                        "events_competed": p.events_competed,
                        "wa_by_event": json.dumps(
                            {k: round(v, 1) for k, v in p.event_wa.items()}
                        ),
                    }
                )

            for from_ev, to_ev in permutations(order, 2):
                pair_profs = [
                    p
                    for p in profiles.values()
                    if from_ev in p.event_times and to_ev in p.event_times
                ]
                if len(pair_profs) < MIN_REPORT_N:
                    continue

                records = []
                cell_counts: dict[str, int] = defaultdict(int)
                for p in pair_profs:
                    bs = specialization_label(p.wa_spread)
                    bif = "best_is_from" if p.best_event == from_ev else "best_not_from"
                    be = f"best_{p.best_event}"
                    records.append(
                        (p.event_times[from_ev], p.event_times[to_ev], bs, bif, be)
                    )
                    cell_counts[f"{bs}|{bif}|{be}"] += 1
                    cell_rows.append(
                        {
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "bal_spec": bs,
                            "best_is_from": bif,
                            "best_event": be,
                            "cell": f"{bs}|{bif}|{be}",
                            "n_in_cell_placeholder": 1,
                        }
                    )

                _, _, r, _, _ = linreg([(a, b) for a, b, _, _, _ in records])
                cvs = cv_strategies(records)
                if not cvs:
                    continue

                best_name = min(cvs.items(), key=lambda kv: kv[1])[0]
                row = {
                    "gender": gender,
                    "event_group": event_group,
                    "from_event": from_ev,
                    "to_event": to_ev,
                    "n": len(records),
                    "r": round(r, 4),
                    "cells_n12": sum(1 for c in cell_counts.values() if c >= MIN_COHORT_N),
                    "n_cells": len(cell_counts),
                    "best_strategy": best_name,
                }
                for s in STRATEGIES:
                    row[f"cv_{s}"] = round(cvs[s], 4)
                row["delta_bal_x_best_event_vs_pooled"] = round(
                    cvs["pooled"] - cvs["bal_x_best_event"], 4
                )
                row["delta_bal_x_best_event_vs_bal_spec"] = round(
                    cvs["bal_spec"] - cvs["bal_x_best_event"], 4
                )
                row["delta_hierarchical_vs_pooled"] = round(
                    cvs["pooled"] - cvs["hierarchical_all_three"], 4
                )
                row["delta_hierarchical_vs_bal_spec"] = round(
                    cvs["bal_spec"] - cvs["hierarchical_all_three"], 4
                )
                row["delta_hierarchical_vs_best_event"] = round(
                    cvs["best_event"] - cvs["hierarchical_all_three"], 4
                )
                strategy_rows.append(row)

                # Fit full-sample formulas for cells with n>=12 (bal × best_event)
                groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
                for p in pair_profs:
                    bs = specialization_label(p.wa_spread)
                    be = f"best_{p.best_event}"
                    groups[(bs, be)].append(
                        (p.event_times[from_ev], p.event_times[to_ev])
                    )
                for (bs, be), pairs in sorted(groups.items()):
                    if len(pairs) < MIN_COHORT_N:
                        continue
                    fit = fit_best(pairs)
                    if not fit:
                        continue
                    model, params, _ = fit
                    # CV for this cell alone
                    cell_cv = min(evaluate_pair(pairs), key=lambda r: r.cv_median_abs)
                    formula_rows.append(
                        {
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "bal_spec": bs,
                            "best_event": be,
                            "best_is_from": (
                                "best_is_from"
                                if be == f"best_{from_ev}"
                                else "best_not_from"
                            ),
                            "n": len(pairs),
                            "best_model": model,
                            "cv_median_abs": round(cell_cv.cv_median_abs, 4),
                            "formula": human_formula(model, params, from_ev, to_ev),
                            "params_json": json.dumps(params),
                        }
                    )

                for cell, n in sorted(cell_counts.items()):
                    # rewrite cell counts more cleanly later via aggregation
                    pass

    # Aggregate cell counts properly
    cell_agg: dict[tuple, int] = defaultdict(int)
    for row in cell_rows:
        key = (
            row["gender"],
            row["event_group"],
            row["from_event"],
            row["to_event"],
            row["cell"],
            row["bal_spec"],
            row["best_is_from"],
            row["best_event"],
        )
        cell_agg[key] += 1
    cell_summary = []
    for key, n in sorted(cell_agg.items()):
        g, eg, fr, to, cell, bs, bif, be = key
        cell_summary.append(
            {
                "gender": g,
                "event_group": eg,
                "from_event": fr,
                "to_event": to,
                "bal_spec": bs,
                "best_is_from": bif,
                "best_event": be,
                "cell": cell,
                "n": n,
            }
        )

    # Summary stats
    n_pairs = len(strategy_rows)
    best_counts = defaultdict(int)
    for r in strategy_rows:
        best_counts[r["best_strategy"]] += 1

    def win_rate(col: str, base: str = "cv_pooled") -> tuple[int, float]:
        wins = sum(1 for r in strategy_rows if r[base] - r[col] > 0.01)
        mean_d = statistics.mean(r[base] - r[col] for r in strategy_rows) if strategy_rows else 0
        return wins, mean_d

    report = [
        "best_is_from × best_event × Balanced/Specialized — 6-Race Time Models",
        "======================================================================",
        "",
        "Question: If we know an athlete's best_is_from, best_event, and",
        "balanced vs specialized status, can we build more accurate time models",
        "for athletes with exactly 6 season races?",
        "",
        "Important nesting note:",
        "  best_is_from is fully determined by best_event for a given pair",
        "  (best_event == from_event ↔ best_is_from). Knowing all three is",
        "  equivalent to knowing bal/spec + best_event.",
        "",
        f"Population: exactly 6 races. CV {CV_FOLDS}-fold seed={CV_SEED}.",
        f"Min cohort n={MIN_COHORT_N}; pair n={MIN_REPORT_N}.",
        "",
        "Strategies:",
        "  pooled                 — one formula",
        "  bal_spec               — balanced vs specialized",
        "  best_is_from           — source is / isn't best-WA event",
        "  best_event             — which event has best WA",
        "  bal_x_best_is_from     — bal/spec × best_is_from",
        "  bal_x_best_event       — bal/spec × best_event  (= know all three)",
        "  hierarchical_all_three — finest available cell, then fall back",
        "",
        f"Pairs evaluated: {n_pairs}",
        f"Best-strategy counts: {dict(best_counts)}",
        "",
        "Strategy vs pooled (wins / mean Δ)",
        "----------------------------------",
    ]
    for s in STRATEGIES:
        if s == "pooled":
            continue
        wins, mean_d = win_rate(f"cv_{s}", "cv_pooled")
        report.append(f"  {s:<24} {wins}/{n_pairs}  mean Δ={mean_d:+.3f}s")

    report.extend([
        "",
        "Strategy vs bal_spec alone (wins / mean Δ)",
        "------------------------------------------",
    ])
    for s in (
        "best_is_from",
        "best_event",
        "bal_x_best_is_from",
        "bal_x_best_event",
        "hierarchical_all_three",
    ):
        wins, mean_d = win_rate(f"cv_{s}", "cv_bal_spec")
        report.append(f"  {s:<24} {wins}/{n_pairs}  mean Δ={mean_d:+.3f}s")

    report.extend([
        "",
        "Pair-level CV median |error|",
        "----------------------------",
        f"{'Pair':<34} {'n':>3} {'Pool':>7} {'Bal':>7} {'BIF':>7} {'BE':>7} "
        f"{'B×BIF':>7} {'B×BE':>7} {'Hier':>7} Best",
    ])
    for r in sorted(
        strategy_rows,
        key=lambda x: (x["event_group"], x["gender"], x["from_event"], x["to_event"]),
    ):
        pair = f"{r['gender']} {r['event_group']} {r['from_event']}->{r['to_event']}"
        report.append(
            f"{pair:<34} {r['n']:>3} "
            f"{r['cv_pooled']:>6.3f}s {r['cv_bal_spec']:>6.3f}s "
            f"{r['cv_best_is_from']:>6.3f}s {r['cv_best_event']:>6.3f}s "
            f"{r['cv_bal_x_best_is_from']:>6.3f}s {r['cv_bal_x_best_event']:>6.3f}s "
            f"{r['cv_hierarchical_all_three']:>6.3f}s {r['best_strategy']}"
        )

    # Cells with enough n for dedicated formulas
    report.extend([
        "",
        f"Dedicated bal×best_event formulas (n≥{MIN_COHORT_N})",
        "-" * 52,
    ])
    if formula_rows:
        for fr in sorted(
            formula_rows,
            key=lambda x: (x["event_group"], x["gender"], x["from_event"], x["to_event"], x["bal_spec"]),
        ):
            report.append(
                f"  [{fr['bal_spec']} ∩ {fr['best_event']} / {fr['best_is_from']}] "
                f"{fr['gender']} {fr['event_group']} {fr['from_event']}->{fr['to_event']} "
                f"n={fr['n']}: {fr['formula']}  CV={fr['cv_median_abs']:.3f}s"
            )
    else:
        report.append("  No bal×best_event cells reached n≥12.")

    # How many 3-way cells are powered?
    n12 = sum(1 for c in cell_summary if c["n"] >= MIN_COHORT_N)
    report.extend([
        "",
        "Cell power",
        "----------",
        f"  Distinct 3-way cells across reported pairs: {len(cell_summary)}",
        f"  Cells with n≥{MIN_COHORT_N}: {n12}",
        "  Most joint cells are small — hierarchical fallback is essential.",
        "",
    ])

    # Verdict
    hier_wins_p, hier_mean_p = win_rate("cv_hierarchical_all_three", "cv_pooled")
    hier_wins_b, hier_mean_b = win_rate("cv_hierarchical_all_three", "cv_bal_spec")
    bxbe_wins_p, bxbe_mean_p = win_rate("cv_bal_x_best_event", "cv_pooled")
    bxbe_wins_b, bxbe_mean_b = win_rate("cv_bal_x_best_event", "cv_bal_spec")
    be_wins_b, be_mean_b = win_rate("cv_best_event", "cv_bal_spec")

    report.extend(["Interpretation", "--------------"])
    report.append(
        f"bal×best_event (= knowing all three) beats pooled on {bxbe_wins_p}/{n_pairs} "
        f"(mean Δ {bxbe_mean_p:+.3f}s) and beats bal/spec alone on {bxbe_wins_b}/{n_pairs} "
        f"(mean Δ {bxbe_mean_b:+.3f}s)."
    )
    report.append(
        f"Hierarchical routing beats pooled on {hier_wins_p}/{n_pairs} "
        f"(mean Δ {hier_mean_p:+.3f}s) and bal/spec on {hier_wins_b}/{n_pairs} "
        f"(mean Δ {hier_mean_b:+.3f}s)."
    )
    report.append(
        f"best_event alone vs bal/spec: {be_wins_b}/{n_pairs} "
        f"(mean Δ {be_mean_b:+.3f}s)."
    )

    if bxbe_wins_b / max(n_pairs, 1) >= 0.5 and bxbe_mean_b > 0.05:
        verdict = (
            "YES — knowing bal/spec together with best_event (which implies best_is_from) "
            "improves 6-race time models for most pairs versus bal/spec alone."
        )
    elif hier_wins_b / max(n_pairs, 1) >= 0.5 and hier_mean_b > 0.05:
        verdict = (
            "YES, with hierarchical routing — the combined factors help for most pairs, "
            "but many 3-way cells are too small for dedicated formulas, so fallbacks matter."
        )
    elif bxbe_wins_p / max(n_pairs, 1) >= 0.5:
        verdict = (
            "PARTIAL — combined routing beats pooled models often, but does not "
            "consistently beat bal/spec alone across pairs."
        )
    else:
        verdict = (
            "LIMITED — knowing these three factors does not reliably improve accuracy "
            "beyond simpler routing in the 6-race sample."
        )
    report.append(f"Verdict: {verdict}")

    report.extend([
        "",
        "Practical recommendation",
        "------------------------",
        "  1. For a known pair (from→to), compute best_event and bal/spec.",
        "  2. Prefer bal×best_event formula when that cell has n≥12; else fall back",
        "     to best_event, then best_is_from, then bal/spec, then pooled 6-race.",
        "  3. best_is_from is redundant once best_event is known for the pair.",
        "",
        "Output files:",
        "  combined_best_event_bal_spec_report.txt",
        "  combined_best_event_bal_spec_findings.txt",
        "  strategy_comparison.csv",
        "  cell_counts.csv",
        "  bal_x_best_event_formulas.csv",
        "  athlete_season_labels.csv",
        "",
        "Source: new_factors/best_event_bal_spec_6races/analyze_combined_factors_6races.py",
    ])

    findings = [
        "best_is_from + best_event + bal/spec — Findings (6 Races)",
        "=========================================================",
        "",
        "Question: Can knowing these three factors create more accurate 6-race",
        "time models?",
        "",
        "Nesting: best_is_from ⊂ best_event for a given prediction pair.",
        "Knowing all three = bal/spec + best_event.",
        "",
        f"Pairs: {n_pairs}. Best strategy counts: {dict(best_counts)}",
        f"bal×best_event vs pooled: {bxbe_wins_p}/{n_pairs} (mean Δ {bxbe_mean_p:+.3f}s)",
        f"bal×best_event vs bal_spec: {bxbe_wins_b}/{n_pairs} (mean Δ {bxbe_mean_b:+.3f}s)",
        f"hierarchical vs pooled: {hier_wins_p}/{n_pairs} (mean Δ {hier_mean_p:+.3f}s)",
        f"hierarchical vs bal_spec: {hier_wins_b}/{n_pairs} (mean Δ {hier_mean_b:+.3f}s)",
        f"Dedicated bal×best_event formulas fitted: {len(formula_rows)}",
        "",
        f"Verdict: {verdict}",
        "",
        "See combined_best_event_bal_spec_report.txt for pair tables and formulas.",
        "Source: analyze_combined_factors_6races.py",
    ]

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            (OUTPUT_ROOT / name).write_text("")
            return
        with open(OUTPUT_ROOT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv("strategy_comparison.csv", strategy_rows)
    write_csv("cell_counts.csv", cell_summary)
    write_csv("bal_x_best_event_formulas.csv", formula_rows)
    write_csv("athlete_season_labels.csv", profile_rows)
    (OUTPUT_ROOT / "combined_best_event_bal_spec_report.txt").write_text(
        "\n".join(report).rstrip() + "\n"
    )
    (OUTPUT_ROOT / "combined_best_event_bal_spec_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )

    # Formula reference text
    if formula_rows:
        lines = [
            "Cross-Event Time Models — bal/spec × best_event — 6 Races",
            "=========================================================",
            "",
            "Use when athlete has exactly 6 races and both bal/spec and best_event are known.",
            "best_is_from is implied by whether best_event equals the source event.",
            "",
        ]
        current = None
        for fr in sorted(
            formula_rows,
            key=lambda x: (x["event_group"], x["gender"], x["bal_spec"], x["from_event"], x["to_event"]),
        ):
            section = f"{fr['event_group']} — {fr['gender']} — {fr['bal_spec']} ∩ {fr['best_event']}"
            if section != current:
                lines.extend(["", section, "-" * len(section), ""])
                current = section
            lines.append(
                f"{fr['from_event']} -> {fr['to_event']}  (n={fr['n']}, {fr['best_is_from']})"
            )
            lines.append(f"  Model: {fr['best_model']}")
            lines.append(f"  Formula: {fr['formula']}")
            lines.append(f"  CV median |error|: {fr['cv_median_abs']:.3f}s")
            lines.append("")
        lines.append("Source: analyze_combined_factors_6races.py")
        (OUTPUT_ROOT / "time_models_bal_x_best_event_6races_formulas.txt").write_text(
            "\n".join(lines).rstrip() + "\n"
        )

    print(f"Wrote combined-factor 6-race analysis to {OUTPUT_ROOT}")
    print(f"  pairs={n_pairs} formulas={len(formula_rows)}")
    print(f"  best counts={dict(best_counts)}")
    print(f"  bal×best_event vs bal_spec: {bxbe_wins_b}/{n_pairs} meanΔ={bxbe_mean_b:+.3f}s")
    print(f"  hierarchical vs bal_spec: {hier_wins_b}/{n_pairs} meanΔ={hier_mean_b:+.3f}s")


if __name__ == "__main__":
    main()
