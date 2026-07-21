"""Balanced / specialized time models for athlete-seasons with 6–10 races.

Population: exactly 6, 7, 8, 9, or 10 valid discipline results in the season.
Specialization (old definition):
  specialized: wa_spread ≥ 50 WA across season PBs in the event group
  balanced:    wa_spread < 50 WA

Outputs (this folder):
  time_models_balanced_6_to_10_races_formulas.txt
  time_models_specialized_6_to_10_races_formulas.txt
  best_bal_spec_6_to_10_time_models.csv
  bal_spec_6_to_10_vs_pooled.csv
  bal_spec_6_to_10_report.txt
  bal_spec_6_to_10_findings.txt
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

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    CandidateResult,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    fit_linear,
    fit_log_linear,
    fit_median_ratio,
    fit_quadratic,
    fit_ratio_linear,
    fit_robust_trimmed,
    metrics,
    pred_binned_ratio,
    pred_linear,
    pred_log_linear,
    pred_median_ratio,
    pred_quadratic,
    pred_ratio_linear,
)
from analyze_specialization_time_models import (  # noqa: E402
    SPREAD_THRESHOLD,
    AthleteSeasonProfile,
    load_athlete_season_profiles,
    specialization_label,
)
from build_specialization_time_models import (  # noqa: E402
    human_formula,
    pick_winner,
    profiles_to_bests,
    binned_lines,
)

MIN_COHORT_N = 12
MIN_REPORT_N = 20
RACE_COUNTS = [6, 7, 8, 9, 10]
POOLED_CSV = OUTPUT_ROOT / "best_time_models_by_race_count.csv"

FIT_PRED: dict[str, tuple[Callable | None, Callable | None]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}


def load_profiles_6_to_10(
    folder: str,
    prefix: str,
    gender: str,
    events: dict[str, int],
) -> dict[str, AthleteSeasonProfile]:
    merged: dict[str, AthleteSeasonProfile] = {}
    for rc in RACE_COUNTS:
        for key, p in load_athlete_season_profiles(folder, prefix, gender, events, rc).items():
            if key not in merged:
                merged[key] = p
    return merged


def load_pooled_6_to_10() -> dict[tuple[str, str, str, str], dict]:
    out: dict[tuple[str, str, str, str], dict] = {}
    if not POOLED_CSV.exists():
        return out
    for row in csv.DictReader(open(POOLED_CSV)):
        if row["race_count_filter"] != "6_to_10":
            continue
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        out[key] = row
    return out


def cv_routed(
    labeled_pairs: list[tuple[float, float, str]],
) -> tuple[float | None, float | None]:
    if len(labeled_pairs) < MIN_REPORT_N:
        return None, None
    n = len(labeled_pairs)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)

    routed_a: list[float] = []
    routed_p: list[float] = []
    group_a: list[float] = []
    group_p: list[float] = []

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else n
        test_idx = set(idx[start:end])
        train = [labeled_pairs[i] for i in range(n) if i not in test_idx]
        test = [labeled_pairs[i] for i in test_idx]
        if len(train) < 10:
            continue

        train_all = [(a, b) for a, b, _ in train]
        group_w = min(evaluate_pair(train_all), key=lambda r: r.cv_median_abs)
        g_fit, g_pred = FIT_PRED.get(group_w.name, (fit_linear, pred_linear))
        if g_fit is None:
            continue
        g_params = g_fit(train_all)

        cohort_fits: dict[str, tuple] = {}
        for lab in ("balanced", "specialized"):
            subset = [(a, b) for a, b, l in train if l == lab]
            if len(subset) < MIN_COHORT_N:
                continue
            w = min(evaluate_pair(subset), key=lambda r: r.cv_median_abs)
            fit_fn, pred_fn = FIT_PRED.get(w.name, (fit_linear, pred_linear))
            if fit_fn and pred_fn:
                cohort_fits[lab] = (fit_fn(subset), pred_fn)

        for x, y, lab in test:
            group_a.append(y)
            group_p.append(g_pred(x, g_params))
            if lab in cohort_fits:
                params, pred_fn = cohort_fits[lab]
                routed_a.append(y)
                routed_p.append(pred_fn(x, params))
            else:
                routed_a.append(y)
                routed_p.append(g_pred(x, g_params))

    if not routed_a:
        return None, None
    return metrics(routed_a, routed_p)[0], metrics(group_a, group_p)[0]


def write_formula_file(
    path: Path,
    title: str,
    cohort: str,
    rows: list[dict],
) -> None:
    mean_cv = statistics.mean(r["best_cv"] for r in rows) if rows else 0.0
    lines = [
        f"Cross-Event Time Models — {title}",
        "=" * (len(title) + 30),
        "",
        "Population: athlete-seasons with 6–10 valid results in the event group",
        f"  AND specialization = {cohort}.",
        f"  specialized: wa_spread ≥ {SPREAD_THRESHOLD:.0f} WA",
        f"  balanced: wa_spread < {SPREAD_THRESHOLD:.0f} WA",
        "",
        "Use when race count is 6–10 and specialization status is known.",
        "Predict time, then convert to WA in the target event table.",
        f"Mean CV median |error| across reported pairs: {mean_cv:.2f}s.",
        f"Minimum n = {MIN_COHORT_N} for a cohort formula.",
        "",
    ]
    current = None
    for row in sorted(
        rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"], r["to_event"])
    ):
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
        if row.get("pooled_fixed") is not None:
            delta = row["best_cv"] - row["pooled_fixed"]
            lines.append(
                f"  vs pooled 6–10 formula on this cohort: {row['pooled_fixed']:.3f}s "
                f"({'↓' if delta < -0.01 else '↑'}{abs(delta):.3f}s)"
            )
        if row.get("ratio_med") is not None:
            lines.append(
                f"  Ratio {row['to_event']}/{row['from_event']}: median {row['ratio_med']:.3f} "
                f"(IQR {row['ratio_p25']:.3f}–{row['ratio_p75']:.3f})"
            )
        if row.get("from_med") is not None:
            lines.append(
                f"  Median times in sample: {row['from_event']} {row['from_med']:.2f}s, "
                f"{row['to_event']} {row['to_med']:.2f}s"
            )
        for bl in row.get("binned_lines", []):
            lines.append(bl)
        lines.append("")

    lines.extend([
        "Pair summary",
        "------------",
        f"  {'Gender':<7} {'Group':<9} {'From':<8} {'To':<8} {'n':>4}  {'CV':>8}  Formula",
    ])
    for row in sorted(
        rows, key=lambda r: (r["event_group"], r["gender"], r["from_event"], r["to_event"])
    ):
        lines.append(
            f"  {row['gender']:<7} {row['event_group']:<9} {row['from_event']:<8} "
            f"{row['to_event']:<8} {row['n']:>4}  {row['best_cv']:>7.3f}s  {row['formula']}"
        )
    lines.append("")
    lines.append("Source: 6-10 races_time_models/build_bal_spec_6_to_10_models.py")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pooled = load_pooled_6_to_10()

    best_rows: list[dict] = []
    routed_rows: list[dict] = []
    bal_formula: list[dict] = []
    spec_formula: list[dict] = []
    cohort_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            profiles = load_profiles_6_to_10(folder, prefix, gender, events)
            for p in profiles.values():
                p.event_group = event_group
                lab = specialization_label(p.wa_spread)
                cohort_counts[(gender, event_group)]["total"] += 1
                cohort_counts[(gender, event_group)][lab] += 1

            for from_ev, to_ev in permutations(order, 2):
                pair_all = [
                    p
                    for p in profiles.values()
                    if from_ev in p.event_times and to_ev in p.event_times
                ]
                if len(pair_all) < MIN_REPORT_N:
                    continue

                bal = [
                    p for p in pair_all if specialization_label(p.wa_spread) == "balanced"
                ]
                spec = [
                    p for p in pair_all if specialization_label(p.wa_spread) == "specialized"
                ]

                all_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_all]
                _, _, r, _, _ = linreg(all_pairs)

                gkey = (gender, event_group, from_ev, to_ev)
                g = pooled.get(gkey)
                g_cv = float(g["best_cv_median_abs"]) if g else None
                g_model = g["best_model"] if g else None
                g_params = json.loads(g["params_json"]) if g else {}

                labeled = [
                    (
                        p.event_times[from_ev],
                        p.event_times[to_ev],
                        specialization_label(p.wa_spread),
                    )
                    for p in pair_all
                ]
                routed_cv, group_cv = cv_routed(labeled)
                if routed_cv is not None and group_cv is not None:
                    routed_rows.append(
                        {
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "n_all": len(pair_all),
                            "n_balanced": len(bal),
                            "n_specialized": len(spec),
                            "pooled_6_to_10_cv": g_cv,
                            "routed_cv": round(routed_cv, 4),
                            "group_cv_in_fold": round(group_cv, 4),
                            "routed_improvement": round(group_cv - routed_cv, 4),
                        }
                    )

                for cohort_name, pair_profs in (("balanced", bal), ("specialized", spec)):
                    if len(pair_profs) < MIN_COHORT_N:
                        continue
                    pairs = [
                        (p.event_times[from_ev], p.event_times[to_ev]) for p in pair_profs
                    ]
                    bests = profiles_to_bests(pair_profs)
                    winner = pick_winner(pairs, bests, from_ev, to_ev, event_group)
                    if not winner:
                        continue

                    pooled_fixed = None
                    if g_model and g_params and g_model in FIT_PRED:
                        _, pred_fn = FIT_PRED[g_model]
                        if pred_fn:
                            try:
                                preds = [pred_fn(x, g_params) for x, _ in pairs]
                                pooled_fixed, _ = metrics([y for _, y in pairs], preds)
                            except (KeyError, TypeError, ValueError):
                                pooled_fixed = None

                    improve = (
                        (pooled_fixed - winner.cv_median_abs) if pooled_fixed is not None else None
                    )
                    formula = human_formula(winner.name, winner.params, from_ev, to_ev)
                    r_med, r_p25, r_p75 = ratio_quartiles(pairs)
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
                        "pooled_6_to_10_cv": g_cv,
                        "pooled_fixed_on_cohort": round(pooled_fixed, 4)
                        if pooled_fixed is not None
                        else None,
                        "improvement_vs_pooled": round(improve, 4) if improve is not None else None,
                        "formula": formula,
                        "params_json": json.dumps(winner.params),
                    }
                    best_rows.append(row)

                    fr = {
                        **row,
                        "formula": formula,
                        "binned_lines": binned_lines(winner.params, from_ev, to_ev),
                        "pooled_fixed": pooled_fixed,
                        "ratio_med": r_med,
                        "ratio_p25": r_p25,
                        "ratio_p75": r_p75,
                        "from_med": statistics.median(x for x, _ in pairs),
                        "to_med": statistics.median(y for _, y in pairs),
                    }
                    if cohort_name == "balanced":
                        bal_formula.append(fr)
                    else:
                        spec_formula.append(fr)

    # Reports
    report = [
        "Balanced / Specialized Time Models — 6 to 10 Races",
        "==================================================",
        "",
        "Question: Within the 6–10 race population, do balanced vs specialized",
        "cohort-specific formulas beat the pooled 6–10 models?",
        "",
        f"Population: athlete-seasons with {RACE_COUNTS[0]}–{RACE_COUNTS[-1]} races.",
        f"Specialized: wa_spread ≥ {SPREAD_THRESHOLD:.0f}. Balanced: wa_spread < {SPREAD_THRESHOLD:.0f}.",
        f"Min cohort n={MIN_COHORT_N}; pair n={MIN_REPORT_N} for routed comparisons.",
        "",
        "Cohort sizes",
        "------------",
    ]
    for (gender, group), counts in sorted(cohort_counts.items()):
        total = counts["total"]
        bal = counts["balanced"]
        spec = counts["specialized"]
        report.append(
            f"  {gender} {group}: total={total}  balanced={bal} ({100*bal/total:.0f}%)  "
            f"specialized={spec} ({100*spec/total:.0f}%)"
        )

    wins = [r for r in best_rows if (r.get("improvement_vs_pooled") or 0) > 0.01]
    report.extend([
        "",
        "Cohort formulas vs pooled 6–10 formula (same cohort)",
        "---------------------------------------------------",
        f"  Cohort models: {len(best_rows)}  beat pooled on cohort: {len(wins)} "
        f"({100*len(wins)/len(best_rows):.0f}%)" if best_rows else "  No cohort models.",
    ])
    for row in sorted(wins, key=lambda r: -(r["improvement_vs_pooled"] or 0))[:15]:
        report.append(
            f"  [{row['cohort']}] {row['gender']} {row['event_group']} "
            f"{row['from_event']}->{row['to_event']} n={row['n']}: "
            f"{row['best_cv']:.3f}s vs {row['pooled_fixed_on_cohort']:.3f}s "
            f"(↓{row['improvement_vs_pooled']:.3f}s)"
        )

    report.extend([
        "",
        "Routed prediction (apply bal or spec formula by known status)",
        "-------------------------------------------------------------",
    ])
    if routed_rows:
        rw = sum(1 for r in routed_rows if r["routed_improvement"] > 0.01)
        report.append(
            f"  Pairs: {len(routed_rows)}  Routed wins: {rw} ({100*rw/len(routed_rows):.0f}%)"
        )
        for row in sorted(routed_rows, key=lambda r: -r["routed_improvement"]):
            report.append(
                f"  {row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']}: "
                f"routed {row['routed_cv']:.3f}s  group {row['group_cv_in_fold']:.3f}s  "
                f"Δ={row['routed_improvement']:+.3f}s  "
                f"(n={row['n_all']}, bal={row['n_balanced']}, spec={row['n_specialized']})"
            )
    else:
        report.append("  No routed comparisons.")

    report.extend(["", "Interpretation", "--------------"])
    if best_rows and len(wins) / len(best_rows) > 0.5:
        report.append(
            "Yes — within the 6–10 race pool, knowing balanced vs specialized status"
            " yields more accurate cohort-specific formulas for most fitted pairs."
        )
    else:
        report.append(
            "Gains from bal/spec stratification in the 6–10 pool are mixed;"
            " see pair tables."
        )
    if routed_rows:
        rw = sum(1 for r in routed_rows if r["routed_improvement"] > 0.01)
        report.append(
            f"Routed prediction improves {rw}/{len(routed_rows)} pairs vs always using"
            " the pooled model among 6–10-race athletes."
        )
    report.extend([
        "",
        "Output files:",
        "  time_models_balanced_6_to_10_races_formulas.txt",
        "  time_models_specialized_6_to_10_races_formulas.txt",
        "  best_bal_spec_6_to_10_time_models.csv",
        "  bal_spec_6_to_10_vs_pooled.csv",
        "  bal_spec_6_to_10_report.txt",
        "  bal_spec_6_to_10_findings.txt",
        "",
        "Source: build_bal_spec_6_to_10_models.py",
    ])

    findings = [
        "Balanced / Specialized — 6 to 10 Races — Findings",
        "=================================================",
        "",
        "Question: Do bal/spec-stratified formulas improve 6–10 race time models?",
        "",
    ]
    for (gender, group), counts in sorted(cohort_counts.items()):
        findings.append(
            f"  {gender} {group}: bal={counts['balanced']}, spec={counts['specialized']}, "
            f"total={counts['total']}"
        )
    findings.append("")
    if best_rows:
        findings.append(
            f"Cohort vs pooled-on-cohort: {len(wins)}/{len(best_rows)} beat pooled "
            f"({100*len(wins)/len(best_rows):.0f}%)"
        )
    if routed_rows:
        rw = sum(1 for r in routed_rows if r["routed_improvement"] > 0.01)
        findings.append(
            f"Routed vs group: {rw}/{len(routed_rows)} "
            f"({100*rw/len(routed_rows):.0f}%)"
        )
        findings.append(
            f"Balanced formulas: {len(bal_formula)} pairs; "
            f"Specialized formulas: {len(spec_formula)} pairs"
        )
    findings.extend([
        "",
        "See bal_spec_6_to_10_report.txt and formula files.",
        "Source: build_bal_spec_6_to_10_models.py",
    ])

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            (OUTPUT_ROOT / name).write_text("")
            return
        # drop non-serializable
        clean = []
        for r in rows:
            clean.append({k: v for k, v in r.items() if k not in ("binned_lines", "pairs")})
        with open(OUTPUT_ROOT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
            w.writeheader()
            w.writerows(clean)

    write_csv("best_bal_spec_6_to_10_time_models.csv", best_rows)
    write_csv("bal_spec_6_to_10_vs_pooled.csv", routed_rows)
    write_formula_file(
        OUTPUT_ROOT / "time_models_balanced_6_to_10_races_formulas.txt",
        "Balanced Athletes — 6 to 10 Races Per Season",
        "balanced (wa_spread < 50 WA)",
        bal_formula,
    )
    write_formula_file(
        OUTPUT_ROOT / "time_models_specialized_6_to_10_races_formulas.txt",
        "Specialized Athletes — 6 to 10 Races Per Season",
        "specialized (wa_spread ≥ 50 WA)",
        spec_formula,
    )
    (OUTPUT_ROOT / "bal_spec_6_to_10_report.txt").write_text("\n".join(report).rstrip() + "\n")
    (OUTPUT_ROOT / "bal_spec_6_to_10_findings.txt").write_text("\n".join(findings).rstrip() + "\n")

    print(f"Wrote bal/spec 6–10 models to {OUTPUT_ROOT}")
    print(f"  balanced formulas: {len(bal_formula)}")
    print(f"  specialized formulas: {len(spec_formula)}")
    print(f"  cohort model rows: {len(best_rows)}  wins vs pooled: {len(wins)}")
    print(f"  routed pairs: {len(routed_rows)}")


if __name__ == "__main__":
    main()
