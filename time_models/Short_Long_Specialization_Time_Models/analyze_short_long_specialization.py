"""Explore Short/Long (Sprints) and Mid/Long (Distance) specialization for time models.

Definitions (require season PBs in all 3 group events):

Sprints
  Short: |WA(100) − WA(200)| < |WA(200) − WA(400)|
  Long:  |WA(400) − WA(200)| < |WA(100) − WA(200)|

Distance
  Mid:  |WA(800) − WA(1500)| < |WA(1500) − WA(5000)|
  Long: |WA(1500) − WA(5000)| < |WA(1500) − WA(800)|

Runs two populations:
  • exactly 6 races (aligned with prior specialization work)
  • 5 or 6 races (needed for usable Short/Mid sample sizes after the 3-event filter)
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
SPEC_ROOT = TIME_MODELS_ROOT / "specialized_time_models"

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    CandidateResult,
    cross_validate,
    evaluate_chain_models,
    evaluate_multivariate,
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
from analyze_specialization_time_models import (  # noqa: E402
    AthleteSeasonProfile,
    load_athlete_season_profiles,
)

MIN_COHORT_N = 12
MIN_REPORT_N = 20

COHORT_A = {"Sprints": "short_sprints", "Distance": "mid_distance"}
COHORT_B = {"Sprints": "long_sprints", "Distance": "long_distance"}
COHORT_LABELS = {
    "short_sprints": "Short Sprints",
    "long_sprints": "Long Sprints",
    "mid_distance": "Mid Distance",
    "long_distance": "Long Distance",
}

POPULATIONS: list[tuple[str, list[int]]] = [
    ("6", [6]),
    ("5_or_6", [5, 6]),
]

FIT_PRED: dict[str, tuple[Callable | None, Callable | None]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}


def classify_short_long(profile: AthleteSeasonProfile, event_group: str) -> str | None:
    wa = profile.event_wa
    if event_group == "Sprints":
        if not all(e in wa for e in ("100m", "200m", "400m")):
            return None
        gap_short = abs(wa["100m"] - wa["200m"])
        gap_long = abs(wa["200m"] - wa["400m"])
        if gap_short < gap_long:
            return "short_sprints"
        if gap_long < gap_short:
            return "long_sprints"
        return "tie"
    if event_group == "Distance":
        if not all(e in wa for e in ("800m", "1500m", "5000m")):
            return None
        gap_mid = abs(wa["800m"] - wa["1500m"])
        gap_long = abs(wa["1500m"] - wa["5000m"])
        if gap_mid < gap_long:
            return "mid_distance"
        if gap_long < gap_mid:
            return "long_distance"
        return "tie"
    return None


def gap_pair(profile: AthleteSeasonProfile, event_group: str) -> tuple[float, float] | None:
    wa = profile.event_wa
    if event_group == "Sprints":
        if not all(e in wa for e in ("100m", "200m", "400m")):
            return None
        return abs(wa["100m"] - wa["200m"]), abs(wa["200m"] - wa["400m"])
    if event_group == "Distance":
        if not all(e in wa for e in ("800m", "1500m", "5000m")):
            return None
        return abs(wa["800m"] - wa["1500m"]), abs(wa["1500m"] - wa["5000m"])
    return None


def load_profiles_multi(
    folder: str,
    prefix: str,
    gender: str,
    events: dict[str, int],
    race_counts: list[int],
) -> dict[str, AthleteSeasonProfile]:
    merged: dict[str, AthleteSeasonProfile] = {}
    for rc in race_counts:
        for key, p in load_athlete_season_profiles(folder, prefix, gender, events, rc).items():
            if key not in merged:
                merged[key] = p
    return merged


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
    return min(results + extra, key=lambda r: r.cv_median_abs)


def load_pooled_models(filter_label: str) -> dict[tuple[str, str, str, str], dict]:
    # Prefer exact filter; fall back to 6 for 5_or_6 if missing
    path = MODEL_SEARCH_ROOT / "race_count" / "best_time_models_by_race_count.csv"
    out: dict[tuple[str, str, str, str], dict] = {}
    fallback: dict[tuple[str, str, str, str], dict] = {}
    for row in csv.DictReader(open(path)):
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        if row["race_count_filter"] == filter_label:
            out[key] = row
        if row["race_count_filter"] == "6":
            fallback[key] = row
    if out:
        return out
    return fallback


def cv_linear(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    return cross_validate(pairs, fit_linear, pred_linear)[0]


def cv_best(pairs: list[tuple[float, float]]) -> tuple[str | None, float | None]:
    if len(pairs) < MIN_COHORT_N:
        return None, None
    results = evaluate_pair(pairs)
    w = min(results, key=lambda r: r.cv_median_abs)
    return w.name, w.cv_median_abs


def cv_routed(
    labeled_pairs: list[tuple[float, float, str]],
    cohort_a: str,
    cohort_b: str,
) -> tuple[float | None, float | None]:
    if len(labeled_pairs) < MIN_COHORT_N:
        return None, None

    n = len(labeled_pairs)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)

    routed_actual: list[float] = []
    routed_pred: list[float] = []
    group_actual: list[float] = []
    group_pred: list[float] = []

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

        a_train = [(x, y) for x, y, lab in train if lab == cohort_a]
        b_train = [(x, y) for x, y, lab in train if lab == cohort_b]

        a_params = b_params = None
        a_pred_fn = b_pred_fn = None
        if len(a_train) >= MIN_COHORT_N:
            aw = min(evaluate_pair(a_train), key=lambda r: r.cv_median_abs)
            a_fit, a_pred_fn = FIT_PRED.get(aw.name, (fit_linear, pred_linear))
            if a_fit:
                a_params = a_fit(a_train)
        if len(b_train) >= MIN_COHORT_N:
            bw = min(evaluate_pair(b_train), key=lambda r: r.cv_median_abs)
            b_fit, b_pred_fn = FIT_PRED.get(bw.name, (fit_linear, pred_linear))
            if b_fit:
                b_params = b_fit(b_train)

        for x, y, lab in test:
            group_actual.append(y)
            group_pred.append(g_pred(x, g_params))
            if lab == cohort_a and a_params and a_pred_fn:
                routed_actual.append(y)
                routed_pred.append(a_pred_fn(x, a_params))
            elif lab == cohort_b and b_params and b_pred_fn:
                routed_actual.append(y)
                routed_pred.append(b_pred_fn(x, b_params))
            else:
                routed_actual.append(y)
                routed_pred.append(g_pred(x, g_params))

    if not routed_actual:
        return None, None
    return metrics(routed_actual, routed_pred)[0], metrics(group_actual, group_pred)[0]


def write_formula_report(
    path: Path,
    title: str,
    cohort: str,
    pop_label: str,
    rows: list[dict],
) -> None:
    lines = [
        f"Cross-Event Time Models — {title}",
        "=" * (len(title) + 30),
        "",
        f"Population: {pop_label} athlete-seasons with season PBs in all 3 group events.",
        f"Specialization cohort: {cohort}",
        "",
        "Sprints: Short = |WA100−WA200| < |WA200−WA400|; Long = reverse.",
        "Distance: Mid = |WA800−WA1500| < |WA1500−WA5000|; Long = reverse.",
        "",
        "Use when short/long (or mid/long) status is known.",
        "Predict time, then convert to WA in the target event table.",
        "",
    ]
    current = None
    for row in sorted(rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"], r["to_event"])):
        section = f"{row['event_group']} — {row['gender']}"
        if section != current:
            lines.extend(["", section, "-" * len(section), ""])
            current = section
        lines.append(f"{row['from_event']} -> {row['to_event']}  (n={row['n']}, r={row['r']:.3f})")
        lines.append(f"  Model: {row['best_model']}")
        lines.append(f"  Formula: {row['formula']}")
        lines.append(f"  CV median |error|: {row['best_cv']:.3f}s")
        if row.get("group_cv") is not None:
            delta = row["best_cv"] - row["group_cv"]
            lines.append(
                f"  vs pooled model on cohort: {row['group_cv']:.3f}s "
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
    for row in sorted(rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"], r["to_event"])):
        lines.append(
            f"  {row['from_event']:<8} {row['to_event']:<8} {row['n']:>4}  "
            f"{row['best_cv']:>7.3f}s  {row['formula']}"
        )
    lines.append("")
    lines.append("Source: Short_Long_Specialization_Time_Models/analyze_short_long_specialization.py")
    path.write_text("\n".join(lines).rstrip() + "\n")


def run_population(
    pop_label: str,
    race_counts: list[int],
) -> dict:
    pooled = load_pooled_models(pop_label if pop_label != "6" else "6")
    if pop_label == "5_or_6":
        pooled = load_pooled_models("5_or_6")

    profile_rows: list[dict] = []
    comparison_rows: list[dict] = []
    best_rows: list[dict] = []
    routed_rows: list[dict] = []
    formula_by_cohort: dict[str, list[dict]] = defaultdict(list)
    classified_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        cohort_a = COHORT_A[event_group]
        cohort_b = COHORT_B[event_group]

        for gender in ("Men", "Women"):
            profiles = load_profiles_multi(folder, prefix, gender, events, race_counts)
            for p in profiles.values():
                p.event_group = event_group

            for p in profiles.values():
                label = classify_short_long(p, event_group)
                gaps = gap_pair(p, event_group)
                classified_counts[(gender, event_group)]["total"] += 1
                if label is None:
                    classified_counts[(gender, event_group)]["missing_3events"] += 1
                elif label == "tie":
                    classified_counts[(gender, event_group)]["tie"] += 1
                else:
                    classified_counts[(gender, event_group)][label] += 1

                profile_rows.append(
                    {
                        "population": pop_label,
                        "athlete_season_key": p.key,
                        "gender": gender,
                        "event_group": event_group,
                        "race_count": p.race_count,
                        "events_competed": p.events_competed,
                        "short_long_label": label or "unclassifiable",
                        "gap_a": round(gaps[0], 1) if gaps else None,
                        "gap_b": round(gaps[1], 1) if gaps else None,
                        "best_event": p.best_event,
                        "wa_by_event": json.dumps({k: round(v, 1) for k, v in p.event_wa.items()}),
                        "times_by_event": json.dumps(
                            {k: round(v, 3) for k, v in p.event_times.items()}
                        ),
                    }
                )

            for from_ev, to_ev in permutations(order, 2):
                pair_all = [
                    p for p in profiles.values() if from_ev in p.event_times and to_ev in p.event_times
                ]
                if len(pair_all) < MIN_REPORT_N:
                    continue

                classifiable = []
                for p in pair_all:
                    lab = classify_short_long(p, event_group)
                    if lab in (cohort_a, cohort_b):
                        classifiable.append((p, lab))

                a_profs = [p for p, lab in classifiable if lab == cohort_a]
                b_profs = [p for p, lab in classifiable if lab == cohort_b]
                class_profs = [p for p, _ in classifiable]

                all_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_all]
                class_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in class_profs]
                a_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in a_profs]
                b_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in b_profs]

                _, _, r, _, _ = linreg(all_pairs)
                gkey = (gender, event_group, from_ev, to_ev)
                g_row = pooled.get(gkey)
                g_cv = float(g_row["best_cv_median_abs"]) if g_row else None
                g_model = g_row["best_model"] if g_row else None
                g_params = json.loads(g_row["params_json"]) if g_row else {}

                for cohort_name, pairs in (
                    ("all", all_pairs),
                    ("classifiable_3event", class_pairs),
                    (cohort_a, a_pairs),
                    (cohort_b, b_pairs),
                ):
                    if len(pairs) < MIN_COHORT_N:
                        continue
                    lin_cv = cv_linear(pairs)
                    bm, bcv = cv_best(pairs)
                    comparison_rows.append(
                        {
                            "population": pop_label,
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "cohort": cohort_name,
                            "n": len(pairs),
                            "r": round(r, 4),
                            "linear_cv": round(lin_cv, 4) if lin_cv is not None else None,
                            "best_model": bm,
                            "best_cv": round(bcv, 4) if bcv is not None else None,
                            "pooled_cv": g_cv,
                        }
                    )

                for cohort_name, pair_profs, pairs in (
                    (cohort_a, a_profs, a_pairs),
                    (cohort_b, b_profs, b_pairs),
                ):
                    if len(pairs) < MIN_COHORT_N:
                        continue
                    bests = {p.key: dict(p.event_times) for p in pair_profs}
                    winner = pick_winner(pairs, bests, from_ev, to_ev, event_group)
                    if not winner:
                        continue

                    g_fixed = None
                    if g_model and g_params and g_model in FIT_PRED:
                        _, pred_fn = FIT_PRED[g_model]
                        if pred_fn:
                            try:
                                preds = [pred_fn(x, g_params) for x, _ in pairs]
                                g_fixed, _ = metrics([y for _, y in pairs], preds)
                            except (KeyError, TypeError, ValueError):
                                g_fixed = None

                    improve = (g_fixed - winner.cv_median_abs) if g_fixed is not None else None
                    formula = human_formula(winner.name, winner.params, from_ev, to_ev)
                    row = {
                        "population": pop_label,
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "cohort": cohort_name,
                        "n": len(pairs),
                        "r": round(r, 4),
                        "best_model": winner.name,
                        "best_cv": round(winner.cv_median_abs, 4),
                        "pooled_fixed_on_cohort": round(g_fixed, 4) if g_fixed is not None else None,
                        "pooled_cv": g_cv,
                        "improvement_vs_pooled": round(improve, 4) if improve is not None else None,
                        "formula": formula,
                        "params_json": json.dumps(winner.params),
                    }
                    best_rows.append(row)
                    formula_by_cohort[cohort_name].append(
                        {
                            **row,
                            "binned_lines": binned_lines(winner.params, from_ev, to_ev),
                            "group_cv": g_fixed,
                        }
                    )

                if len(classifiable) >= MIN_REPORT_N:
                    labeled = [
                        (p.event_times[from_ev], p.event_times[to_ev], lab)
                        for p, lab in classifiable
                    ]
                    routed_cv, group_cv = cv_routed(labeled, cohort_a, cohort_b)
                    if routed_cv is not None and group_cv is not None:
                        routed_rows.append(
                            {
                                "population": pop_label,
                                "gender": gender,
                                "event_group": event_group,
                                "from_event": from_ev,
                                "to_event": to_ev,
                                "n_classifiable": len(classifiable),
                                "n_cohort_a": len(a_profs),
                                "n_cohort_b": len(b_profs),
                                "cohort_a": cohort_a,
                                "cohort_b": cohort_b,
                                "routed_cv": round(routed_cv, 4),
                                "group_cv": round(group_cv, 4),
                                "routed_improvement": round(group_cv - routed_cv, 4),
                                "pooled_cv": g_cv,
                            }
                        )

    return {
        "pop_label": pop_label,
        "race_counts": race_counts,
        "profile_rows": profile_rows,
        "comparison_rows": comparison_rows,
        "best_rows": best_rows,
        "routed_rows": routed_rows,
        "formula_by_cohort": formula_by_cohort,
        "classified_counts": classified_counts,
    }


def section_report(result: dict) -> list[str]:
    pop = result["pop_label"]
    pop_desc = "exactly 6 races" if pop == "6" else "5 or 6 races"
    lines = [
        f"POPULATION: {pop_desc}",
        "=" * (12 + len(pop_desc)),
        "",
        "Cohort sizes",
        "------------",
    ]
    for (gender, group), counts in sorted(result["classified_counts"].items()):
        a, b = COHORT_A[group], COHORT_B[group]
        lines.append(
            f"  {gender} {group}: total={counts['total']}  "
            f"{COHORT_LABELS[a]}={counts[a]}  {COHORT_LABELS[b]}={counts[b]}  "
            f"tie={counts['tie']}  missing_all_3_events={counts['missing_3events']}"
        )

    lines.extend([
        "",
        "Pair-level linear OLS CV by cohort",
        "----------------------------------",
        f"{'Gender':<7} {'Group':<9} {'Pair':<14} {'nA':>3} {'nB':>3} "
        f"{'CohA':>8} {'CohB':>8} {'Δ(A−B)':>8} Better?",
    ])

    a_better = b_better = comparable = 0
    deltas: list[float] = []
    pair_keys = sorted(
        {
            (r["gender"], r["event_group"], r["from_event"], r["to_event"])
            for r in result["comparison_rows"]
            if r["cohort"] in set(COHORT_A.values()) | set(COHORT_B.values())
        }
    )
    for gender, group, from_ev, to_ev in pair_keys:
        ca, cb = COHORT_A[group], COHORT_B[group]
        by_c = {
            r["cohort"]: r
            for r in result["comparison_rows"]
            if r["gender"] == gender
            and r["event_group"] == group
            and r["from_event"] == from_ev
            and r["to_event"] == to_ev
        }
        if ca not in by_c or cb not in by_c:
            continue
        a_cv, b_cv = by_c[ca]["linear_cv"], by_c[cb]["linear_cv"]
        if a_cv is None or b_cv is None:
            continue
        delta = a_cv - b_cv
        comparable += 1
        deltas.append(delta)
        if delta < -0.01:
            a_better += 1
            better = COHORT_LABELS[ca]
        elif delta > 0.01:
            b_better += 1
            better = COHORT_LABELS[cb]
        else:
            better = "~same"
        lines.append(
            f"{gender:<7} {group:<9} {from_ev}->{to_ev:<6} "
            f"{by_c[ca]['n']:>3} {by_c[cb]['n']:>3} "
            f"{a_cv:>7.3f}s {b_cv:>7.3f}s {delta:>+7.3f}s {better}"
        )

    if comparable:
        lines.extend([
            "",
            f"  Both arms n≥{MIN_COHORT_N}: {comparable} pairs",
            f"  Short/Mid more predictable: {a_better}/{comparable}",
            f"  Long more predictable: {b_better}/{comparable}",
            f"  Mean Δ (Short/Mid − Long): {statistics.mean(deltas):+.3f}s",
            f"  Median Δ: {statistics.median(deltas):+.3f}s",
        ])
    else:
        lines.append("")
        lines.append(f"  No pairs with both cohorts at n≥{MIN_COHORT_N}.")

    lines.extend([
        "",
        "Cohort-specific formulas vs pooled model (same cohort)",
        "------------------------------------------------------",
    ])
    best = result["best_rows"]
    wins = [r for r in best if (r.get("improvement_vs_pooled") or 0) > 0.01]
    losses = [r for r in best if (r.get("improvement_vs_pooled") or 0) < -0.01]
    if best:
        lines.append(
            f"  Compared: {len(best)}  beat pooled: {len(wins)} ({100*len(wins)/len(best):.0f}%)  "
            f"worse: {len(losses)}"
        )
        for row in sorted(wins, key=lambda r: -(r["improvement_vs_pooled"] or 0))[:12]:
            lines.append(
                f"  [{COHORT_LABELS.get(row['cohort'], row['cohort'])}] "
                f"{row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']} "
                f"n={row['n']}: {row['best_cv']:.3f}s vs {row['pooled_fixed_on_cohort']:.3f}s "
                f"(↓{row['improvement_vs_pooled']:.3f}s)"
            )
    else:
        lines.append("  No cohort models.")

    lines.extend([
        "",
        "Routed prediction (status-aware formula assignment)",
        "---------------------------------------------------",
    ])
    routed = result["routed_rows"]
    if routed:
        rw = sum(1 for r in routed if r["routed_improvement"] > 0.01)
        lines.append(f"  Pairs: {len(routed)}  Routed wins: {rw} ({100*rw/len(routed):.0f}%)")
        for row in sorted(routed, key=lambda r: -r["routed_improvement"]):
            lines.append(
                f"  {row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']}: "
                f"routed {row['routed_cv']:.3f}s  group {row['group_cv']:.3f}s  "
                f"Δ={row['routed_improvement']:+.3f}s  "
                f"(n={row['n_classifiable']}, A={row['n_cohort_a']}, B={row['n_cohort_b']})"
            )
    else:
        lines.append("  No routed comparisons.")

    result["_stats"] = {
        "comparable": comparable,
        "a_better": a_better,
        "b_better": b_better,
        "deltas": deltas,
        "wins": len(wins),
        "best_n": len(best),
        "routed_n": len(routed),
        "routed_wins": sum(1 for r in routed if r["routed_improvement"] > 0.01) if routed else 0,
    }
    lines.append("")
    return lines


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [run_population(label, rcs) for label, rcs in POPULATIONS]

    report = [
        "Short/Long (and Mid/Long) Specialization vs Time Model Accuracy",
        "================================================================",
        "",
        "Research question: If we know Short vs Long sprint specialization",
        "(or Mid vs Long distance specialization), do time predictions improve?",
        "",
        "Definitions (require season PBs in all 3 events of the group):",
        "  Sprints — Short: |WA(100)−WA(200)| < |WA(200)−WA(400)|",
        "  Sprints — Long:  |WA(400)−WA(200)| < |WA(100)−WA(200)|",
        "  Distance — Mid:  |WA(800)−WA(1500)| < |WA(1500)−WA(5000)|",
        "  Distance — Long: |WA(1500)−WA(5000)| < |WA(1500)−WA(800)|",
        "",
        "Note: Short/Long (and Mid/Long) are complements except exact gap ties.",
        f"Minimum n={MIN_COHORT_N} for cohort CV; n={MIN_REPORT_N} to report a pair.",
        "",
        "Why two populations?",
        "  Exactly 6 races alone leaves Mid Distance (n=11) and all sprint arms",
        "  below the cohort threshold after requiring all 3 event PBs. The 5-or-6",
        "  race population is the practical sample for this question.",
        "",
    ]

    for result in results:
        report.extend(section_report(result))

    # Interpretation from best-powered population (5_or_6)
    r56 = next(r for r in results if r["pop_label"] == "5_or_6")
    r6 = next(r for r in results if r["pop_label"] == "6")
    s56 = r56["_stats"]
    s6 = r6["_stats"]

    report.extend([
        "Interpretation",
        "--------------",
    ])
    if s6["best_n"] == 0 and s56["best_n"] == 0:
        report.append("Insufficient sample under both populations.")
    else:
        if s6["best_n"]:
            report.append(
                f"At exactly 6 races: {s6['wins']}/{s6['best_n']} cohort models beat pooled; "
                f"routed wins {s6['routed_wins']}/{s6['routed_n']}. "
                "Only Long Distance reaches n≥12; Short/Mid/Sprints are underpowered."
            )
        if s56["comparable"]:
            report.append(
                f"At 5 or 6 races: Short/Mid more predictable on {s56['a_better']}/{s56['comparable']} "
                f"pairs; Long on {s56['b_better']}/{s56['comparable']} "
                f"(mean Δ Short/Mid−Long = {statistics.mean(s56['deltas']):+.3f}s)."
            )
        if s56["best_n"]:
            report.append(
                f"Cohort-specific formulas beat pooled on {s56['wins']}/{s56['best_n']} "
                f"({100*s56['wins']/s56['best_n']:.0f}%) subgroup fits at 5-or-6."
            )
        if s56["routed_n"]:
            report.append(
                f"Routed prediction improves {s56['routed_wins']}/{s56['routed_n']} "
                f"({100*s56['routed_wins']/s56['routed_n']:.0f}%) pairs vs always using the "
                "pooled model among classifiable athletes."
            )
        if s56["routed_n"] and s56["routed_wins"] / s56["routed_n"] > 0.5:
            report.append(
                "Verdict: YES — knowing Short/Long (Mid/Long) status improves accuracy "
                "when you fit and route cohort-specific formulas (best evidence at 5-or-6 races)."
            )
        elif s56["best_n"] and s56["wins"] / s56["best_n"] > 0.5:
            report.append(
                "Verdict: PARTIAL — cohort-specific formulas often beat pooled formulas "
                "within a subgroup, but routing across the full classifiable population "
                "is not consistently better. Status helps most when applying the matching "
                "cohort formula to that athlete."
            )
        else:
            report.append(
                "Verdict: LIMITED — knowing status does not reliably improve time models "
                "beyond pooled high-race formulas for this sample."
            )

    report.extend([
        "",
        "Caveats:",
        "  • Classification requires PBs in all 3 group events (~40–60% of high-race",
        "    athlete-seasons are unclassifiable).",
        "  • 6-race-only Short/Mid/Sprint arms are too small for stable CV.",
        "  • Women's cohorts remain small even at 5-or-6.",
        "",
        "Output files:",
        "  short_long_specialization_report.txt",
        "  short_long_findings.txt",
        "  athlete_season_short_long.csv",
        "  short_long_model_comparison.csv",
        "  best_short_long_time_models.csv",
        "  short_long_vs_group_comparison.csv",
        "  time_models_*_{6,5or6}_formulas.txt (per cohort with enough n)",
        "",
        "Source: Short_Long_Specialization_Time_Models/analyze_short_long_specialization.py",
    ])

    # Findings summary
    findings = [
        "Short/Long Specialization Time Models — Findings",
        "================================================",
        "",
        "Question: Does knowing Short vs Long sprint (or Mid vs Long distance)",
        "specialization improve cross-event time prediction?",
        "",
        "Definitions require season PBs in all 3 events of the group.",
        "",
        "At exactly 6 races:",
    ]
    for (gender, group), counts in sorted(r6["classified_counts"].items()):
        a, b = COHORT_A[group], COHORT_B[group]
        findings.append(
            f"  {gender} {group}: {COHORT_LABELS[a]}={counts[a]}, "
            f"{COHORT_LABELS[b]}={counts[b]}, miss3={counts['missing_3events']}/{counts['total']}"
        )
    findings.append(
        f"  Cohort models: {s6['wins']}/{s6['best_n']} beat pooled; "
        f"routed {s6['routed_wins']}/{s6['routed_n']}."
    )
    findings.extend(["", "At 5 or 6 races (powered sample):"])
    for (gender, group), counts in sorted(r56["classified_counts"].items()):
        a, b = COHORT_A[group], COHORT_B[group]
        findings.append(
            f"  {gender} {group}: {COHORT_LABELS[a]}={counts[a]}, "
            f"{COHORT_LABELS[b]}={counts[b]}, miss3={counts['missing_3events']}/{counts['total']}"
        )
    if s56["comparable"]:
        findings.append(
            f"  Predictability: Short/Mid better {s56['a_better']}/{s56['comparable']}, "
            f"Long better {s56['b_better']}/{s56['comparable']}, "
            f"mean Δ={statistics.mean(s56['deltas']):+.3f}s"
        )
    if s56["best_n"]:
        findings.append(
            f"  Cohort formulas beat pooled: {s56['wins']}/{s56['best_n']} "
            f"({100*s56['wins']/s56['best_n']:.0f}%)"
        )
    if s56["routed_n"]:
        findings.append(
            f"  Routed vs group: {s56['routed_wins']}/{s56['routed_n']} "
            f"({100*s56['routed_wins']/s56['routed_n']:.0f}%)"
        )
    findings.extend([
        "",
        "See short_long_specialization_report.txt for pair-level detail.",
        "Source: analyze_short_long_specialization.py",
    ])

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            (OUTPUT_ROOT / name).write_text("")
            return
        with open(OUTPUT_ROOT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    all_profiles = [r for res in results for r in res["profile_rows"]]
    all_comp = [r for res in results for r in res["comparison_rows"]]
    all_best = [r for res in results for r in res["best_rows"]]
    all_routed = [r for res in results for r in res["routed_rows"]]

    write_csv("athlete_season_short_long.csv", all_profiles)
    write_csv("short_long_model_comparison.csv", all_comp)
    write_csv("best_short_long_time_models.csv", all_best)
    write_csv("short_long_vs_group_comparison.csv", all_routed)
    (OUTPUT_ROOT / "short_long_specialization_report.txt").write_text("\n".join(report).rstrip() + "\n")
    (OUTPUT_ROOT / "short_long_findings.txt").write_text("\n".join(findings).rstrip() + "\n")

    # Formula files for 5_or_6 (primary usable) and 6 where present
    for result in results:
        pop = result["pop_label"]
        pop_desc = "exactly 6 races" if pop == "6" else "5 or 6 races"
        suffix = "6races" if pop == "6" else "5or6_races"
        for cohort, rows in result["formula_by_cohort"].items():
            if not rows:
                continue
            fname = f"time_models_{cohort}_{suffix}_formulas.txt"
            write_formula_report(
                OUTPUT_ROOT / fname,
                f"{COHORT_LABELS[cohort]} — {pop_desc.title()}",
                COHORT_LABELS[cohort],
                pop_desc,
                rows,
            )

    print(f"Wrote Short/Long specialization analysis to {OUTPUT_ROOT}")
    for result in results:
        s = result["_stats"]
        print(
            f"  {result['pop_label']}: profiles={len(result['profile_rows'])} "
            f"cohort_models={s['best_n']} routed={s['routed_n']} "
            f"wins_vs_pooled={s['wins']} routed_wins={s['routed_wins']}"
        )


if __name__ == "__main__":
    main()
