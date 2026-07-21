"""Write indoor-format feature-important time-model reports for steeple-inclusive bands.

Outputs (prefix new_format_):
  new_format_feature_important_time_models_band_{lo}_{hi}_with_steeple.txt

Format matches Indoor_Analysis/.../new_feature_important_time_models_band_750_950.txt:
  feature explanations, formula explanations, per-pair important features,
  and how each recommended feature alters the formula (cohort formulas).
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
TIME_MODELS = PROJECT_ROOT / "time_models"
BAND_FI = TIME_MODELS / "Point_Bands_Time_Models" / "Feature_Importance_Point_Band_Time_Models"

sys.path.insert(0, str(OUTPUT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS))
sys.path.insert(0, str(TIME_MODELS / "model_search"))
sys.path.insert(0, str(TIME_MODELS / "specialized_time_models"))
sys.path.insert(0, str(TIME_MODELS / "Point_Bands_Time_Models"))
sys.path.insert(0, str(BAND_FI))

from build_cross_event_time_models import linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
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
from analyze_feature_importance_point_bands import (  # noqa: E402
    COMBO_STRATEGIES,
    EPS,
    MIN_COHORT_N,
    MIN_REPORT_N,
    SPREAD_THRESHOLD,
    TARGET_BANDS,
    feature_labelers,
)
from analyze_feature_importance_with_steeple import (  # noqa: E402
    DISTANCE_EVENTS,
    EVENT_ORDER,
    SPRINT_EVENTS,
    in_band,
    load_profiles,
)

FIT_PRED: dict[str, tuple[Callable | None, Callable | None]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}

PARAMETRIC = {
    "linear_ols",
    "robust_trimmed_linear",
    "median_ratio",
    "log_linear",
    "ratio_linear",
    "quadratic",
}

STRATEGY_CV_KEYS = {
    "pooled": "cv_pooled",
    "bal_spec": "cv_bal_spec",
    "best_event": "cv_best_event",
    "best_is_from": "cv_best_is_from",
    "pair_wa_gap_50": "cv_pair_wa_gap_50",
    "from_stronger_wa": "cv_from_stronger_wa",
    "bal_x_best_event": "cv_bal_x_best_event",
}

FEATURE_HOW_TO_KNOW = {
    "pooled": "No extra labels — use the single band formula for everyone in this band.",
    "bal_spec": (
        f"Compute wa_spread = max(event WA) − min(event WA) across events with a "
        f"season PB. specialized if ≥ {SPREAD_THRESHOLD:.0f}; else balanced."
    ),
    "best_event": (
        "Identify which event in the group has the athlete's highest season WA; "
        "route to the best_<event> formula."
    ),
    "best_is_from": (
        "Check whether the athlete's best-WA event equals the source (from) event. "
        "Yes → best_is_from formula; no → best_not_from."
    ),
    "pair_wa_gap_50": (
        "Compute |WA(from) − WA(to)|. If ≥ 50, use pair_gap_ge50; else pair_gap_lt50."
    ),
    "from_stronger_wa": (
        "Compare WA(from) to WA(to). If from ≥ to, use from_stronger_wa; "
        "else from_weaker_wa."
    ),
    "bal_x_best_event": (
        "Need both bal/spec (wa_spread ≥ 50) and best_event. Prefer the matching "
        "cell formula when n is large enough; else fall back to best_event, then "
        "bal/spec, then pooled."
    ),
}

FEATURE_DESCRIPTIONS = {
    "bal_spec": f"balanced vs specialized (wa_spread ≥ {SPREAD_THRESHOLD:.0f})",
    "best_event": "Identity of best-WA event in the group",
    "best_is_from": "Athlete's best-WA event is the source event",
    "pair_wa_gap_50": "|WA(from)−WA(to)| ≥ 50",
    "from_stronger_wa": "Source event has higher WA than target",
    "bal_x_best_event": "bal/spec × best_event joint route",
}

FEATURE_LABEL_FNS = {
    "bal_spec": "bal_spec",
    "best_event": "best_event",
    "best_is_from": "best_is_from",
    "pair_wa_gap_50": "pair_wa_gap_50",
    "from_stronger_wa": "from_stronger_wa",
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
        edges = params.get("edges", [])
        ratios = params.get("ratios", [])
        parts = [f"{to_ev} = {from_ev} × ratio_bin({from_ev})"]
        for edge, ratio in zip(edges, ratios):
            parts.append(f"    nearest {from_ev} {edge:.2f}s → {ratio:.3f}")
        return "\n".join(parts)
    if model == "knn_median":
        k = params.get("k", "?")
        n_train = params.get("n_train", "?")
        return (
            f"{to_ev} ≈ median of k={k} nearest {from_ev} neighbors "
            f"(train n={n_train}; non-parametric)"
        )
    return format_params(model, params)


def fit_full(pairs: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    results = evaluate_pair(pairs)
    winner = min(results, key=lambda r: r.cv_median_abs)
    if winner.name not in PARAMETRIC and winner.name != "binned_ratio":
        parametric = [r for r in results if r.name in PARAMETRIC]
        if parametric:
            best_p = min(parametric, key=lambda r: r.cv_median_abs)
            if best_p.cv_median_abs <= winner.cv_median_abs + 0.05:
                winner = best_p
    if winner.name not in FIT_PRED or FIT_PRED[winner.name][0] is None:
        return {
            "model": winner.name,
            "params_raw": winner.params,
            "cv_median_abs": winner.cv_median_abs,
            "full_median_abs": winner.full_median_abs,
            "n": len(pairs),
        }
    fit_fn, pred_fn = FIT_PRED[winner.name]
    params = fit_fn(pairs)
    preds = [pred_fn(x, params) for x, _ in pairs]
    full_med, _ = metrics([y for _, y in pairs], preds)
    return {
        "model": winner.name,
        "params_raw": params,
        "cv_median_abs": winner.cv_median_abs,
        "full_median_abs": full_med,
        "n": len(pairs),
    }


def important_features(cv_row: dict) -> list[tuple[str, float, float]]:
    pooled = float(cv_row["cv_pooled"])
    out = []
    for strat in COMBO_STRATEGIES:
        if strat == "pooled":
            continue
        cv = float(cv_row[STRATEGY_CV_KEYS[strat]])
        delta = pooled - cv
        if delta > EPS:
            out.append((strat, cv, delta))
    out.sort(key=lambda t: -t[2])
    return out


def cohort_formulas_for_strategy(
    profiles: list,
    from_ev: str,
    to_ev: str,
    strategy: str,
    labelers: dict[str, Callable],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if strategy == "bal_x_best_event":
        cells: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for p in profiles:
            bs = labelers["bal_spec"](p)
            be = labelers["best_event"](p)
            cells[(bs, be)].append((p.event_times[from_ev], p.event_times[to_ev]))
        for (bs, be), pairs in sorted(cells.items()):
            fit = fit_full(pairs)
            if not fit:
                continue
            fit["formula"] = human_formula(fit["model"], fit["params_raw"], from_ev, to_ev)
            fit["label"] = f"{bs} ∩ {be}"
            fit["n"] = len(pairs)
            rows.append(fit)
        return rows

    if strategy == "pooled" or strategy not in FEATURE_LABEL_FNS:
        return rows

    lab_key = FEATURE_LABEL_FNS[strategy]
    cells2: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in profiles:
        cells2[labelers[lab_key](p)].append(
            (p.event_times[from_ev], p.event_times[to_ev])
        )
    for lab, pairs in sorted(cells2.items()):
        fit = fit_full(pairs)
        if not fit:
            continue
        fit["formula"] = human_formula(fit["model"], fit["params_raw"], from_ev, to_ev)
        fit["label"] = lab
        fit["n"] = len(pairs)
        rows.append(fit)
    return rows


def alter_explanation(
    strategy: str,
    cohort_rows: list[dict],
    from_ev: str,
    to_ev: str,
    pooled_formula: str,
) -> list[str]:
    lines = [
        f"  How {strategy} alters the formula:",
        f"  • Baseline (pooled band): {pooled_formula}",
    ]
    if not cohort_rows:
        lines.append(
            f"  • No cohort with n≥{MIN_COHORT_N} under this split — keep pooled, "
            "or use CV routing with fallbacks."
        )
        return lines

    if strategy == "bal_x_best_event":
        lines.append(
            "  • Split athletes by balanced/specialized AND best_event; each cell "
            "gets its own formula when n is large enough:"
        )
    elif strategy == "best_is_from":
        lines.append(
            f"  • If best-WA event is {from_ev}, use best_is_from; otherwise best_not_from:"
        )
    elif strategy == "best_event":
        lines.append("  • Route by which event holds the athlete's season-best WA:")
    elif strategy == "bal_spec":
        lines.append(
            f"  • If wa_spread ≥ {SPREAD_THRESHOLD:.0f}, use specialized; else balanced:"
        )
    elif strategy == "pair_wa_gap_50":
        lines.append(
            f"  • If |WA({from_ev})−WA({to_ev})| ≥ 50, use pair_gap_ge50; else pair_gap_lt50:"
        )
    elif strategy == "from_stronger_wa":
        lines.append(
            f"  • If WA({from_ev}) ≥ WA({to_ev}), use from_stronger_wa; else from_weaker_wa:"
        )
    else:
        lines.append(f"  • Route by {strategy} label:")

    for c in cohort_rows:
        formula_lines = c["formula"].split("\n")
        lines.append(
            f"      [{c['label']}] n={c['n']}: {formula_lines[0]}  "
            f"(CV {c['cv_median_abs']:.3f}s)"
        )
        for fl in formula_lines[1:]:
            lines.append(f"                 {fl}")
    return lines


def load_cv_index() -> dict[tuple, dict]:
    path = OUTPUT_ROOT / "pair_strategy_cv_by_band_with_steeple.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path.name} — run analyze_feature_importance_with_steeple.py first."
        )
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row["band"],
                row["gender"],
                row["event_group"],
                row["from_event"],
                row["to_event"],
            )
            out[key] = row
    return out


def write_band_file(lo: int, hi: int, profiles_by_gg: dict, cv_index: dict) -> Path:
    band = f"{lo}-{hi}"
    title = (
        f"Feature-Important Time Models — WA Band {band} "
        f"(Width 200, with Steeplechase)"
    )
    lines = [
        title,
        "=" * len(title),
        "",
        "Method:",
        "  • Data: outdoor CSVs 2024–2026, results on/after March 1.",
        f"  • Population: athlete-seasons with ≥1 individual result in WA [{lo}, {hi}).",
        "  • Excludes relays. Sprints (100/200/400) & Distance",
        "    (800/1500/3000m Steeplechase/5000) — corrected steeple WA from",
        "    New_Steeplechase_Data.",
        "  • Pooled formula: lowest-CV winner among candidate families (same search",
        f"    as prior time models; seed={CV_SEED}, folds={CV_FOLDS}).",
        f"  • Feature formulas: re-fit within feature cohorts (n≥{MIN_COHORT_N}); "
        f"pair needs n≥{MIN_REPORT_N}.",
        "  • Important features per pair come from Feature_Importance CV",
        "    (strategies that beat pooled by >0.01s).",
        "",
        "Feature explanations",
        "--------------------",
        f"  bal_spec: {FEATURE_DESCRIPTIONS['bal_spec']}",
        f"    → {FEATURE_HOW_TO_KNOW['bal_spec']}",
        f"  best_event: {FEATURE_DESCRIPTIONS['best_event']}",
        f"    → {FEATURE_HOW_TO_KNOW['best_event']}",
        f"  best_is_from: {FEATURE_DESCRIPTIONS['best_is_from']}",
        f"    → {FEATURE_HOW_TO_KNOW['best_is_from']}",
        f"  pair_wa_gap_50: {FEATURE_DESCRIPTIONS['pair_wa_gap_50']}",
        f"    → {FEATURE_HOW_TO_KNOW['pair_wa_gap_50']}",
        f"  from_stronger_wa: {FEATURE_DESCRIPTIONS['from_stronger_wa']}",
        f"    → {FEATURE_HOW_TO_KNOW['from_stronger_wa']}",
        f"  bal_x_best_event: {FEATURE_DESCRIPTIONS['bal_x_best_event']}",
        f"    → {FEATURE_HOW_TO_KNOW['bal_x_best_event']}",
        "",
        "Formula explanations",
        "--------------------",
        "  linear_ols / robust_trimmed_linear: to = intercept + slope × from",
        "  median_ratio: to = ratio × from  (ratio = median of to/from)",
        "  log_linear: to = exp(intercept + slope × log(from))",
        "  ratio_linear: to = from × (a + b×from)",
        "  quadratic: to = a + b×from + c×from²",
        "  binned_ratio: to = from × ratio_bin(from); bins shown as nearest from→ratio",
        "  knn_median: non-parametric median of k nearest from-neighbors' to times",
        "",
        "Use:",
        f"  1. Confirm the athlete has a result in WA band {band}.",
        "  2. Start from the pooled formula for the event pair.",
        "  3. If important features are known, switch to the matching cohort",
        "     formula below (prefer the pair's recommended strategy first).",
        "  4. Convert predicted time to World Athletics points in the TARGET event.",
        "",
    ]

    n_pairs = 0
    for event_group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            profiles_all = list(profiles_by_gg[(gender, event_group)].values())
            band_profiles = [p for p in profiles_all if in_band(p.result_was, lo, hi)]
            section = f"{event_group} — {gender}"
            section_lines: list[str] = ["", section, "-" * len(section), ""]
            any_pair = False

            for from_ev, to_ev in permutations(EVENT_ORDER[event_group], 2):
                pair_ps = [
                    p
                    for p in band_profiles
                    if from_ev in p.event_times and to_ev in p.event_times
                ]
                if len(pair_ps) < MIN_REPORT_N:
                    continue
                cv_row = cv_index.get((band, gender, event_group, from_ev, to_ev))
                if not cv_row:
                    continue
                any_pair = True
                n_pairs += 1
                labelers = feature_labelers(pair_ps, from_ev, to_ev)
                pairs = [
                    (p.event_times[from_ev], p.event_times[to_ev]) for p in pair_ps
                ]
                pooled_fit = fit_full(pairs)
                if pooled_fit is None:
                    continue
                pooled_formula = human_formula(
                    pooled_fit["model"], pooled_fit["params_raw"], from_ev, to_ev
                )
                _, _, r, _, _ = linreg(pairs)
                r_med, r_p25, r_p75 = ratio_quartiles(pairs)
                best_strat = cv_row["best_strategy"]
                pooled_cv = float(cv_row["cv_pooled"])
                best_cv = float(cv_row[STRATEGY_CV_KEYS[best_strat]])
                important = important_features(cv_row)

                block = [
                    f"{from_ev} -> {to_ev}  (n={len(pair_ps)} athlete-seasons, r={r:.3f})",
                    f"  Pooled band model: {pooled_fit['model']}",
                    f"  Formula: {pooled_formula.split(chr(10))[0]}",
                ]
                for fl in pooled_formula.split("\n")[1:]:
                    block.append(f"           {fl}")
                block.append(
                    f"  CV median |error| (pooled routing): {pooled_cv:.3f}s"
                )
                if pooled_fit.get("full_median_abs") is not None:
                    block.append(
                        f"  Full-sample median |error|: {pooled_fit['full_median_abs']:.3f}s"
                    )
                block.append(
                    f"  Ratio {to_ev}/{from_ev}: median {r_med:.3f} "
                    f"(IQR {r_p25:.3f}–{r_p75:.3f})"
                )
                block.append(
                    f"  Median times: {from_ev} "
                    f"{statistics.median(p[0] for p in pairs):.2f}s, "
                    f"{to_ev} {statistics.median(p[1] for p in pairs):.2f}s"
                )
                block.append("")
                block.append("  Important features to know")
                block.append("  --------------------------")
                if not important:
                    block.append(
                        "  • None of the tested features beat pooled by >0.01s for "
                        "this pair — use the pooled formula as-is."
                    )
                else:
                    block.append(
                        f"  • Recommended route: {best_strat} "
                        f"(CV {best_cv:.3f}s, Δ {pooled_cv - best_cv:+.3f}s vs pooled)."
                    )
                    block.append(
                        f"  • How to determine it: {FEATURE_HOW_TO_KNOW.get(best_strat, '')}"
                    )
                    block.append("  • Other features that also beat pooled:")
                    for strat, cv, delta in important:
                        mark = " ← recommended" if strat == best_strat else ""
                        block.append(
                            f"      - {strat}: CV {cv:.3f}s (Δ {delta:+.3f}s){mark}"
                        )

                    block.append("")
                    cohorts = cohort_formulas_for_strategy(
                        pair_ps, from_ev, to_ev, best_strat, labelers
                    )
                    block.extend(
                        alter_explanation(
                            best_strat,
                            cohorts,
                            from_ev,
                            to_ev,
                            pooled_formula.split("\n")[0],
                        )
                    )

                    runners = [t for t in important if t[0] != best_strat][:1]
                    for strat, cv, delta in runners:
                        if delta < 0.05:
                            continue
                        cohorts2 = cohort_formulas_for_strategy(
                            pair_ps, from_ev, to_ev, strat, labelers
                        )
                        if not cohorts2:
                            continue
                        block.append("")
                        block.append(
                            f"  Alternate useful split — {strat} "
                            f"(CV {cv:.3f}s, Δ {delta:+.3f}s):"
                        )
                        block.append(f"  • {FEATURE_HOW_TO_KNOW.get(strat, '')}")
                        block.extend(
                            alter_explanation(
                                strat,
                                cohorts2,
                                from_ev,
                                to_ev,
                                pooled_formula.split("\n")[0],
                            )
                        )

                block.append("")
                section_lines.extend(block)

            if any_pair:
                lines.extend(section_lines)
            else:
                lines.extend(
                    [
                        "",
                        section,
                        "-" * len(section),
                        "",
                        f"No pairs met n≥{MIN_REPORT_N} in band {band}.",
                        "",
                    ]
                )

    lines.extend(
        [
            "",
            "Coverage / notes",
            "----------------",
            f"  • {n_pairs} event pairs reported for band {band}.",
            f"  • Cohort formulas require n≥{MIN_COHORT_N}; thin cells fall back to pooled.",
            "  • Prefer the recommended strategy when its labels are available;",
            "    otherwise use the next-best listed feature or the pooled formula.",
            "  • Distance includes corrected 3000m Steeplechase WA.",
            "",
            "Source:",
            "  New_Steeplechase_Data/Feature_Importance_Point_Band_Time_Models/"
            "generate_new_format_feature_important_time_models.py",
            "  Feature importance CV: pair_strategy_cv_by_band_with_steeple.csv",
        ]
    )

    out = OUTPUT_ROOT / f"new_format_feature_important_time_models_band_{lo}_{hi}_with_steeple.txt"
    out.write_text("\n".join(lines).rstrip() + "\n")
    return out


def main() -> None:
    cv_index = load_cv_index()
    profiles_by_gg = {}
    for gender in ("Men", "Women"):
        profiles_by_gg[(gender, "Sprints")] = load_profiles(
            gender, "Sprints", SPRINT_EVENTS
        )
        profiles_by_gg[(gender, "Distance")] = load_profiles(
            gender, "Distance", DISTANCE_EVENTS
        )
        print(
            f"Loaded {gender}: Sprints={len(profiles_by_gg[(gender,'Sprints')])} "
            f"Distance={len(profiles_by_gg[(gender,'Distance')])}"
        )

    for lo, hi in TARGET_BANDS:
        path = write_band_file(lo, hi, profiles_by_gg, cv_index)
        print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
