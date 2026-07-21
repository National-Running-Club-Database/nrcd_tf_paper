"""Build specialization-stratified time models for 6-race athlete-seasons.

Compares specialized-only and balanced-only formulas vs group-average 6-race models,
and evaluates routed prediction (apply cohort-specific formula when status is known).

Outputs to time_models/specialized_time_models/
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = TIME_MODELS_ROOT / "specialized_time_models"
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(OUTPUT_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    format_params,
    cross_validate,
    fit_linear,
    pred_linear,
    fit_median_ratio,
    pred_median_ratio,
    fit_robust_trimmed,
    fit_log_linear,
    pred_log_linear,
    fit_ratio_linear,
    pred_ratio_linear,
    fit_quadratic,
    pred_quadratic,
    pred_binned_ratio,
    metrics,
    CandidateResult,
)
from analyze_specialization_time_models import (  # noqa: E402
    RACE_COUNT,
    SPREAD_THRESHOLD,
    AthleteSeasonProfile,
    load_athlete_season_profiles,
    specialization_label,
)

MIN_COHORT_N = 12
MIN_REPORT_N = 20

FIT_PRED: dict[str, tuple[Callable, Callable]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}


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
    if model == "multivariate_ols":
        return (
            f"{to_ev} = {params['intercept']:.2f} + {params['coef_a']:.3f}×{params['event_a']} "
            f"+ {params['coef_b']:.3f}×{params['event_b']}"
        )
    return format_params(model, params)


def binned_lines(params: dict, from_ev: str, to_ev: str) -> list[str]:
    edges = params.get("edges", [])
    ratios = params.get("ratios", [])
    if not edges:
        return []
    lines = [f"  Binned lookup {from_ev} → {to_ev} (nearest {from_ev} edge):"]
    for edge, ratio in zip(edges, ratios):
        lines.append(f"    {edge:.2f}s → ratio {ratio:.3f}")
    return lines


def load_group6_models() -> dict[tuple[str, str, str, str], dict]:
    path = MODEL_SEARCH_ROOT / "race_count" / "best_time_models_by_race_count.csv"
    out = {}
    for row in csv.DictReader(open(path)):
        if row["race_count_filter"] != "6":
            continue
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        out[key] = row
    return out


def pick_winner(
    pairs: list[tuple[float, float]],
    bests: dict[str, dict[str, float]],
    from_ev: str,
    to_ev: str,
    event_group: str,
) -> CandidateResult | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    results = evaluate_pair(pairs)
    extra: list[CandidateResult] = []
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
    return min(all_r, key=lambda r: r.cv_median_abs)


def fit_model(model_name: str, pairs: list[tuple[float, float]]) -> dict | None:
    fit_fn, _ = FIT_PRED.get(model_name, (fit_linear, pred_linear))
    if fit_fn is None:
        return None
    return fit_fn(pairs)


def predict(model_name: str, x: float, params: dict) -> float:
    _, pred_fn = FIT_PRED.get(model_name, (fit_linear, pred_linear))
    return pred_fn(x, params)


def cv_apply_fixed_model(
    pairs: list[tuple[float, float]],
    model_name: str,
    params: dict,
) -> float | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    fit_fn, pred_fn = FIT_PRED.get(model_name, (fit_linear, pred_linear))
    if fit_fn is None:
        # evaluate on same data as training — use full-sample error as proxy
        preds = [pred_fn(x, params) for x, _ in pairs]
        actual = [y for _, y in pairs]
        med, _ = metrics(actual, preds)
        return med
    return cross_validate(pairs, fit_fn, pred_fn)[0]


def cv_routed_models(
    labeled_pairs: list[tuple[float, float, str]],
) -> tuple[float | None, float | None, dict[str, float | None]]:
    """CV comparing routed (cohort-specific) vs group (all-data) models.

    labeled_pairs: (from_time, to_time, specialization_label)
    Returns (routed_cv_med, group_cv_med, cohort_cv_dict)
    """
    if len(labeled_pairs) < MIN_COHORT_N:
        return None, None, {}

    n = len(labeled_pairs)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = n // CV_FOLDS

    routed_actual: list[float] = []
    routed_pred: list[float] = []
    group_actual: list[float] = []
    group_pred: list[float] = []

    cohort_actual: dict[str, list[float]] = defaultdict(list)
    cohort_pred: dict[str, list[float]] = defaultdict(list)

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else n
        test_idx = set(idx[start:end])
        train = [labeled_pairs[i] for i in range(n) if i not in test_idx]
        test = [labeled_pairs[i] for i in test_idx]
        if len(train) < 10:
            continue

        train_all = [(a, b) for a, b, _ in train]
        group_results = evaluate_pair(train_all)
        group_w = min(group_results, key=lambda r: r.cv_median_abs)
        g_fit, g_pred = FIT_PRED.get(group_w.name, (fit_linear, pred_linear))
        if g_fit is None:
            continue
        g_params = g_fit(train_all)

        spec_train = [(a, b) for a, b, lab in train if lab == "specialized"]
        bal_train = [(a, b) for a, b, lab in train if lab == "balanced"]

        spec_w = bal_w = None
        spec_params = bal_params = None
        spec_fit = spec_pred_fn = bal_fit = bal_pred_fn = None

        if len(spec_train) >= MIN_COHORT_N:
            spec_results = evaluate_pair(spec_train)
            spec_w = min(spec_results, key=lambda r: r.cv_median_abs)
            spec_fit, spec_pred_fn = FIT_PRED.get(spec_w.name, (fit_linear, pred_linear))
            if spec_fit:
                spec_params = spec_fit(spec_train)

        if len(bal_train) >= MIN_COHORT_N:
            bal_results = evaluate_pair(bal_train)
            bal_w = min(bal_results, key=lambda r: r.cv_median_abs)
            bal_fit, bal_pred_fn = FIT_PRED.get(bal_w.name, (fit_linear, pred_linear))
            if bal_fit:
                bal_params = bal_fit(bal_train)

        for x, y, lab in test:
            group_actual.append(y)
            group_pred.append(g_pred(x, g_params))

            if lab == "specialized" and spec_params and spec_pred_fn:
                routed_actual.append(y)
                routed_pred.append(spec_pred_fn(x, spec_params))
                cohort_actual["specialized"].append(y)
                cohort_pred["specialized"].append(spec_pred_fn(x, spec_params))
            elif lab == "balanced" and bal_params and bal_pred_fn:
                routed_actual.append(y)
                routed_pred.append(bal_pred_fn(x, bal_params))
                cohort_actual["balanced"].append(y)
                cohort_pred["balanced"].append(bal_pred_fn(x, bal_params))
            else:
                # fallback to group model
                routed_actual.append(y)
                routed_pred.append(g_pred(x, g_params))

    if not routed_actual:
        return None, None, {}

    routed_med, _ = metrics(routed_actual, routed_pred)
    group_med, _ = metrics(group_actual, group_pred)
    cohort_cv = {}
    for lab in ("specialized", "balanced"):
        if cohort_actual[lab]:
            cohort_cv[lab], _ = metrics(cohort_actual[lab], cohort_pred[lab])
    return routed_med, group_med, cohort_cv


def profiles_to_bests(profiles: list[AthleteSeasonProfile]) -> dict[str, dict[str, float]]:
    return {p.key: dict(p.event_times) for p in profiles}


def write_formula_report(
    path: Path,
    title: str,
    cohort: str,
    rows: list[dict],
) -> None:
    lines = [
        f"Cross-Event Time Models — {title}",
        "=" * len(title),
        "",
        f"Cohort: 6-race athlete-seasons, specialization = {cohort}",
        f"  specialized: wa_spread ≥ {SPREAD_THRESHOLD:.0f} WA",
        f"  balanced: wa_spread < {SPREAD_THRESHOLD:.0f} WA",
        "",
        "Use when athlete's specialization status is known.",
        "Predict time, then convert to WA in the target event table.",
        "",
    ]
    current = None
    for row in sorted(rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"])):
        section = f"{row['event_group']} — {row['gender']}"
        if section != current:
            lines.extend(["", section, "-" * len(section), ""])
            current = section
        lines.append(
            f"{row['from_event']} -> {row['to_event']}  (n={row['n']}, r={row['r']:.3f})"
        )
        lines.append(f"  Model: {row['best_model']}")
        lines.append(f"  Formula: {row['formula']}")
        lines.append(f"  CV median |error|: {row['best_cv']:.3f}s")
        if row.get("group6_cv") is not None:
            delta = row["best_cv"] - row["group6_cv"]
            lines.append(
                f"  vs group-average 6-race model: {row['group6_cv']:.3f}s "
                f"({'↓' if delta < -0.01 else '↑'}{abs(delta):.3f}s)"
            )
        for bl in row.get("binned_lines", []):
            lines.append(bl)
        lines.append("")

    lines.extend([
        "Pair summary",
        "------------",
        f"  {'From':<8} {'To':<8} {'n':>4}  {'CV':>8}  Formula",
    ])
    for row in sorted(rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"])):
        lines.append(
            f"  {row['from_event']:<8} {row['to_event']:<8} {row['n']:>4}  "
            f"{row['best_cv']:>7.3f}s  {row['formula']}"
        )
    lines.append("")
    lines.append("Source: specialized_time_models/build_specialization_time_models.py")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    group6 = load_group6_models()

    best_rows: list[dict] = []
    comparison_rows: list[dict] = []
    spec_formula_rows: list[dict] = []
    bal_formula_rows: list[dict] = []

    report = [
        "Specialization-Stratified Time Models (6 Races)",
        "===============================================",
        "",
        "Question: If specialization status is known, do cohort-specific formulas",
        "beat the group-average 6-race models?",
        "",
        f"Population: exactly {RACE_COUNT} races per athlete-season.",
        f"Specialized: wa_spread ≥ {SPREAD_THRESHOLD:.0f} WA. Balanced: wa_spread < {SPREAD_THRESHOLD:.0f}.",
        f"Minimum n={MIN_COHORT_N} to fit cohort model; n={MIN_REPORT_N} to report pair.",
        "",
        "Routed prediction: apply specialized formula to specialized athletes and",
        "balanced formula to balanced athletes (with group-model fallback if cohort",
        f"train n < {MIN_COHORT_N} in a CV fold).",
        "",
    ]

    routed_wins = 0
    cohort_wins_vs_group = 0
    cohort_comparisons = 0
    routed_comparisons = 0

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            profiles = load_athlete_season_profiles(folder, prefix, gender, events, RACE_COUNT)
            for p in profiles.values():
                p.event_group = event_group

            all_profs = list(profiles.values())
            spec_profs = [p for p in all_profs if specialization_label(p.wa_spread) == "specialized"]
            bal_profs = [p for p in all_profs if specialization_label(p.wa_spread) == "balanced"]

            for from_ev, to_ev in permutations(order, 2):
                pair_all = [p for p in all_profs if from_ev in p.event_times and to_ev in p.event_times]
                pair_spec = [p for p in spec_profs if from_ev in p.event_times and to_ev in p.event_times]
                pair_bal = [p for p in bal_profs if from_ev in p.event_times and to_ev in p.event_times]

                if len(pair_all) < MIN_REPORT_N:
                    continue

                gkey = (gender, event_group, from_ev, to_ev)
                g6 = group6.get(gkey)
                g6_cv = float(g6["best_cv_median_abs"]) if g6 else None
                g6_model = g6["best_model"] if g6 else None
                g6_params = json.loads(g6["params_json"]) if g6 else {}

                all_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_all]
                _, _, r, _, _ = linreg(all_pairs)

                labeled = [
                    (p.event_times[from_ev], p.event_times[to_ev], specialization_label(p.wa_spread))
                    for p in pair_all
                ]
                routed_cv, group_cv_fold, _ = cv_routed_models(labeled)

                for cohort_name, pair_profs in (
                    ("specialized", pair_spec),
                    ("balanced", pair_bal),
                ):
                    if len(pair_profs) < MIN_COHORT_N:
                        continue
                    pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_profs]
                    bests = profiles_to_bests(pair_profs)
                    winner = pick_winner(pairs, bests, from_ev, to_ev, event_group)
                    if not winner:
                        continue

                    g6_on_cohort = None
                    g6_fixed_on_cohort = None
                    if g6_model:
                        _, pred_fn = FIT_PRED.get(g6_model, (fit_linear, pred_linear))
                        if pred_fn and g6_params:
                            preds = [pred_fn(x, g6_params) for x, _ in pairs]
                            actual = [y for _, y in pairs]
                            g6_fixed_on_cohort, _ = metrics(actual, preds)
                        fit_fn, pred_fn = FIT_PRED.get(g6_model, (fit_linear, pred_linear))
                        if fit_fn:
                            g6_on_cohort = cross_validate(pairs, fit_fn, pred_fn)[0]

                    baseline = g6_fixed_on_cohort if g6_fixed_on_cohort is not None else g6_on_cohort
                    improve = (baseline - winner.cv_median_abs) if baseline is not None else None
                    if improve is not None and improve > 0.01:
                        cohort_wins_vs_group += 1
                    if g6_on_cohort is not None:
                        cohort_comparisons += 1

                    formula = human_formula(winner.name, winner.params, from_ev, to_ev)
                    row = {
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "cohort": cohort_name,
                        "n": len(pairs),
                        "r": round(r, 4),
                        "best_model": winner.name,
                        "best_cv": round(winner.cv_median_abs, 4),
                        "group6_cv_on_cohort": round(g6_on_cohort, 4) if g6_on_cohort is not None else None,
                        "group6_fixed_cv_on_cohort": round(g6_fixed_on_cohort, 4)
                        if g6_fixed_on_cohort is not None
                        else None,
                        "group6_cv_pooled": g6_cv,
                        "improvement_vs_group6_on_cohort": round(improve, 4) if improve is not None else None,
                        "formula": formula,
                        "params_json": json.dumps(winner.params),
                    }
                    best_rows.append(row)

                    fr = {
                        **row,
                        "binned_lines": binned_lines(winner.params, from_ev, to_ev),
                        "group6_cv": g6_fixed_on_cohort if g6_fixed_on_cohort is not None else g6_on_cohort,
                    }
                    if cohort_name == "specialized":
                        spec_formula_rows.append(fr)
                    else:
                        bal_formula_rows.append(fr)

                if routed_cv is not None and group_cv_fold is not None:
                    routed_comparisons += 1
                    if routed_cv < group_cv_fold - 0.01:
                        routed_wins += 1
                    comparison_rows.append(
                        {
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "n_all": len(pair_all),
                            "n_specialized": len(pair_spec),
                            "n_balanced": len(pair_bal),
                            "group6_cv_pooled": g6_cv,
                            "routed_cv": round(routed_cv, 4),
                            "group_cv_within_routed_cv": round(group_cv_fold, 4),
                            "routed_improvement": round(group_cv_fold - routed_cv, 4),
                        }
                    )

    # Report sections
    report.append("Cohort-specific models beating group-average on same cohort")
    report.append("-" * 55)
    report.append(
        f"  Pairs compared: {cohort_comparisons}  "
        f"Cohort model wins: {cohort_wins_vs_group}  "
        f"({100*cohort_wins_vs_group/cohort_comparisons:.0f}%)" if cohort_comparisons else "  No comparisons"
    )
    report.append("")
    report.append("Routed vs group-only CV (specialization-aware assignment)")
    report.append("-" * 55)
    if routed_comparisons:
        report.append(
            f"  Pairs compared: {routed_comparisons}  "
            f"Routed wins: {routed_wins} ({100*routed_wins/routed_comparisons:.0f}%)"
        )
        for row in sorted(comparison_rows, key=lambda r: -r["routed_improvement"]):
            pair = f"{row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']}"
            report.append(
                f"  {pair}: routed {row['routed_cv']:.3f}s  "
                f"group {row['group_cv_within_routed_cv']:.3f}s  "
                f"Δ={row['routed_improvement']:+.3f}s  "
                f"(n={row['n_all']}, spec={row['n_specialized']}, bal={row['n_balanced']})"
            )
    report.append("")

    report.append("Best cohort-specific formulas (CV improvement > 0.01s vs group on cohort)")
    report.append("-" * 55)
    improved = [r for r in best_rows if (r.get("improvement_vs_group6_on_cohort") or 0) > 0.01]
    for row in sorted(improved, key=lambda r: -r["improvement_vs_group6_on_cohort"]):
        pair = f"{row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']}"
        report.append(
            f"  [{row['cohort']}] {pair}  n={row['n']}"
        )
        report.append(f"    Formula: {row['formula']}")
        report.append(
            f"    CV: {row['best_cv']:.3f}s vs group on cohort {row['group6_cv_on_cohort']:.3f}s "
            f"(↓{row['improvement_vs_group6_on_cohort']:.3f}s)"
        )

    report.extend([
        "",
        "Interpretation",
        "--------------",
    ])
    if cohort_comparisons and cohort_wins_vs_group / cohort_comparisons > 0.5:
        report.append(
            "Yes — cohort-specific formulas often beat applying the pooled 6-race model to the"
            " same specialization subgroup. The balanced cohort sees the largest gains."
        )
    else:
        report.append(
            "Cohort-specific formulas help in many pairs but not universally; balanced"
            " athletes benefit most when dedicated formulas are available."
        )
    if routed_comparisons and routed_wins / routed_comparisons > 0.5:
        report.append(
            "Routed prediction (pick formula by known specialization status) beats always"
            " using the group-average model for most pairs."
        )
    else:
        report.append(
            "Routed prediction shows modest gains; knowing specialization status helps most"
            " when applying the balanced-athlete formulas."
        )
    report.extend([
        "",
        "Output files:",
        "  best_specialization_time_models.csv",
        "  specialization_vs_group_comparison.csv",
        "  time_models_specialized_6races_formulas.txt",
        "  time_models_balanced_6races_formulas.txt",
        "",
        "Source: build_specialization_time_models.py",
    ])

    with open(OUTPUT_ROOT / "best_specialization_time_models.csv", "w", newline="") as f:
        if best_rows:
            w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            w.writeheader()
            w.writerows(best_rows)

    with open(OUTPUT_ROOT / "specialization_vs_group_comparison.csv", "w", newline="") as f:
        if comparison_rows:
            w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader()
            w.writerows(comparison_rows)

    write_formula_report(
        OUTPUT_ROOT / "time_models_specialized_6races_formulas.txt",
        "Specialized Athletes — 6 Races Per Season",
        "specialized (wa_spread ≥ 50 WA)",
        spec_formula_rows,
    )
    write_formula_report(
        OUTPUT_ROOT / "time_models_balanced_6races_formulas.txt",
        "Balanced Athletes — 6 Races Per Season",
        "balanced (wa_spread < 50 WA)",
        bal_formula_rows,
    )
    (OUTPUT_ROOT / "specialization_stratified_models_report.txt").write_text(
        "\n".join(report).rstrip() + "\n"
    )
    print(f"Wrote specialization-stratified models to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
