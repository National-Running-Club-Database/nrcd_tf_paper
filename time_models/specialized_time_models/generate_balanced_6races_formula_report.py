"""Generate balanced-athlete formula reports (6 races and/or 5-or-6 races).

Outputs in specialized_time_models/:
  time_models_balanced_6races_formulas.txt
  time_models_balanced_5_6_race_formulas.txt
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
OUTPUT_ROOT = TIME_MODELS_ROOT / "specialized_time_models"
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(OUTPUT_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    EVENT_ORDER,
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
from analyze_specialization_time_models import (  # noqa: E402
    SPREAD_THRESHOLD,
    load_athlete_season_profiles,
    specialization_label,
)
from build_specialization_time_models import (  # noqa: E402
    MIN_COHORT_N,
    human_formula,
    pick_winner,
    profiles_to_bests,
)

MIN_PAIR_N = MIN_COHORT_N  # 12

PRED = {
    "linear_ols": (fit_linear, pred_linear),
    "log_linear": (fit_log_linear, pred_log_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "binned_ratio": (None, pred_binned_ratio),
}


def model_label(model: str) -> str:
    labels = {
        "linear_ols": "linear OLS",
        "robust_trimmed_linear": "robust_trimmed_linear",
        "log_linear": "log_linear (multiplicative)",
        "median_ratio": "median_ratio",
        "ratio_linear": "ratio_linear (speed-dependent)",
        "quadratic": "quadratic",
        "binned_ratio": "binned_ratio (speed bins)",
        "multivariate_ols": "multivariate_ols (two inputs required)",
        "chain_residual_adjusted": "chain_residual_adjusted (via intermediate event)",
    }
    return labels.get(model, model)


def binned_table(params: dict, from_ev: str, to_ev: str) -> list[str]:
    edges = params.get("edges", [])
    ratios = params.get("ratios", [])
    if not edges:
        return []
    lines = [f"  Binned lookup for {from_ev} -> {to_ev} (use ratio at nearest {from_ev} edge):"]
    for edge, ratio in zip(edges, ratios):
        lines.append(f"    nearest {from_ev} edge {edge:.2f}s -> ratio {ratio:.3f}")
    return lines


def load_profiles_multi(
    folder: str,
    prefix: str,
    gender: str,
    events: dict[str, int],
    race_counts: list[int],
):
    merged = {}
    for rc in race_counts:
        for key, p in load_athlete_season_profiles(folder, prefix, gender, events, rc).items():
            if key not in merged:
                merged[key] = p
    return merged


def load_pooled_models(filter_label: str) -> dict[tuple[str, str, str, str], dict]:
    path = MODEL_SEARCH_ROOT / "race_count" / "best_time_models_by_race_count.csv"
    out: dict[tuple[str, str, str, str], dict] = {}
    fallback: dict[tuple[str, str, str, str], dict] = {}
    for row in csv.DictReader(open(path)):
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        if row["race_count_filter"] == filter_label:
            out[key] = row
        if row["race_count_filter"] == "6":
            fallback[key] = row
    return out if out else fallback


def pair_stats(
    pairs: list[tuple[float, float]],
    model: str,
    params: dict,
    from_ev: str,
    to_ev: str,
    cv_med: float,
) -> dict[str, Any]:
    _, _, r, _, _ = linreg(pairs)
    r_med, r_p25, r_p75 = ratio_quartiles(pairs)
    from_times = [p[0] for p in pairs]
    to_times = [p[1] for p in pairs]

    full_med = None
    if model in PRED and PRED[model][1] is not None:
        pred_fn = PRED[model][1]
        preds = [pred_fn(x, params) for x, _ in pairs]
        actual = [y for _, y in pairs]
        full_med, _ = metrics(actual, preds)
    if model == "binned_ratio":
        full_med = None

    return {
        "n": len(pairs),
        "r": r,
        "formula": human_formula(model, params, from_ev, to_ev),
        "model": model,
        "cv_med": cv_med,
        "full_med": full_med,
        "r_med": r_med,
        "r_p25": r_p25,
        "r_p75": r_p75,
        "from_med": statistics.median(from_times),
        "to_med": statistics.median(to_times),
        "params": params,
        "binned_lines": binned_table(params, from_ev, to_ev),
    }


def build_balanced_rows(
    race_counts: list[int],
    pooled_filter: str,
) -> tuple[list[dict], dict[tuple, int]]:
    pooled = load_pooled_models(pooled_filter)
    rows: list[dict] = []
    balanced_n: dict[tuple, int] = {}

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            profiles = load_profiles_multi(folder, prefix, gender, events, race_counts)
            bal_profs = [
                p for p in profiles.values() if specialization_label(p.wa_spread) == "balanced"
            ]

            for from_ev, to_ev in permutations(order, 2):
                pair_profs = [
                    p for p in bal_profs if from_ev in p.event_times and to_ev in p.event_times
                ]
                key = (gender, event_group, from_ev, to_ev)
                balanced_n[key] = len(pair_profs)
                if len(pair_profs) < MIN_PAIR_N:
                    continue

                pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_profs]
                bests = profiles_to_bests(pair_profs)
                winner = pick_winner(pairs, bests, from_ev, to_ev, event_group)
                if not winner:
                    continue

                g = pooled.get(key)
                g_cv = float(g["best_cv_median_abs"]) if g else None
                g_model = g["best_model"] if g else None
                g_params = json.loads(g["params_json"]) if g else {}
                g_fixed = None
                if g_model and g_params and g_model in PRED:
                    _, pred_fn = PRED[g_model]
                    if pred_fn:
                        try:
                            preds = [pred_fn(x, g_params) for x, _ in pairs]
                            g_fixed, _ = metrics([y for _, y in pairs], preds)
                        except (KeyError, TypeError, ValueError):
                            g_fixed = None

                rows.append(
                    {
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n": len(pairs),
                        "best_model": winner.name,
                        "best_cv": winner.cv_median_abs,
                        "params": winner.params,
                        "pooled_cv": g_cv,
                        "pooled_fixed_on_cohort": g_fixed,
                        "pairs": pairs,
                    }
                )
    return rows, balanced_n


def write_report(
    rows: list[dict],
    balanced_n: dict[tuple, int],
    output_path: Path,
    *,
    pop_title: str,
    pop_desc: str,
    use_line: str,
    pooled_label: str,
) -> None:
    mean_cv = statistics.mean(r["best_cv"] for r in rows) if rows else 0.0
    n_beat = sum(
        1
        for r in rows
        if r.get("pooled_fixed_on_cohort") is not None
        and r["best_cv"] < r["pooled_fixed_on_cohort"] - 0.01
    )
    n_cmp = sum(1 for r in rows if r.get("pooled_fixed_on_cohort") is not None)

    lines = [
        f"Cross-Event Time Models — Balanced Athletes — {pop_title} (Sprints & Distance)",
        "=" * (len(pop_title) + 70),
        "",
        "Method:",
        "  • Data: relay-inclusive outdoor CSVs, 2024–2026, results on or after March 1.",
        f"  • Population: athlete-seasons with {pop_desc} in the event group's",
        "    discipline CSV AND wa_spread < 50 WA across season PBs in that group.",
        "  • Specialization: wa_spread = max(season PB WA) − min(season PB WA) across events",
        "    in the group with a PB that season.",
        "  • Balanced: wa_spread < 50 WA (well-rounded multi-event profile within the group).",
        "  • Athlete-season PB: fastest valid mark per event within that season (WA > 0).",
        "  • Model selection: 5-fold cross-validation across 8+ candidate model families;",
        "    the formula below is the lowest-CV-median-|error| winner for each pair.",
        "  • 3000m Steeplechase excluded from distance.",
        f"  • Minimum n = {MIN_PAIR_N} balanced athlete-season pairs required to report a model.",
        "",
        "Use:",
        f"  • {use_line}",
        f"  • Mean CV median |error| across reported pairs: {mean_cv:.2f}s.",
    ]
    if n_cmp:
        lines.append(
            f"  • Balanced-specific formulas beat pooled {pooled_label} formulas on "
            f"{n_beat} of {n_cmp} reported pairs."
        )
    lines.extend([
        "  • Predict time in the unknown event from known PB(s), then convert predicted",
        "    time to World Athletics points in the TARGET event table.",
        "  • Treat CV median |error| as the typical ± uncertainty band (cross-validated).",
        "  • Do not compare WA across events directly.",
        "  • If specialization is unknown or wa_spread ≥ 50, use the pooled race-count",
        "    formula files or specialized-cohort formulas instead.",
        "",
        "Coverage note:",
        f"  {len(rows)} of 24 event pairs met n ≥ {MIN_PAIR_N} among balanced athletes",
        f"  with {pop_desc}.",
        "",
    ])

    reported_keys: set[tuple[str, str, str, str]] = set()
    summary_rows: list[tuple[str, str, str, str, dict]] = []

    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            section_rows = [r for r in rows if r["gender"] == gender and r["event_group"] == group]
            section = f"{group} — {gender}"
            lines.extend(["", section, "-" * len(section), ""])

            if not section_rows:
                lines.append(
                    f"No pairs met the minimum sample size (n ≥ {MIN_PAIR_N} balanced athlete-seasons)."
                )
                lines.append(
                    "Use time_models_5_or_6_races_formulas.txt or cross_event_time_models_report.txt."
                )
                lines.append("")
                continue

            section_rows.sort(key=lambda r: (r["from_event"], r["to_event"]))
            for row in section_rows:
                from_ev, to_ev = row["from_event"], row["to_event"]
                key = (gender, group, from_ev, to_ev)
                reported_keys.add(key)
                pairs = row["pairs"]
                stats = pair_stats(
                    pairs, row["best_model"], row["params"], from_ev, to_ev, row["best_cv"]
                )
                summary_rows.append((gender, group, from_ev, to_ev, stats))

                lines.append(
                    f"{from_ev} -> {to_ev}  (n={stats['n']} balanced athlete-seasons, r={stats['r']:.3f})"
                )
                lines.append(f"  Model: {model_label(stats['model'])}")
                formula_lines = stats["formula"].split("\n")
                lines.append(f"  Formula: {formula_lines[0]}")
                for fl in formula_lines[1:]:
                    lines.append(f"           {fl}")
                if stats["full_med"] is not None:
                    lines.append(
                        f"  CV median |error|: {stats['cv_med']:.3f}s | "
                        f"full-sample median |error|: {stats['full_med']:.3f}s"
                    )
                else:
                    lines.append(f"  CV median |error|: {stats['cv_med']:.3f}s")
                if row.get("pooled_cv") is not None:
                    lines.append(f"  vs pooled {pooled_label} model CV: {row['pooled_cv']:.3f}s")
                if row.get("pooled_fixed_on_cohort") is not None:
                    delta = stats["cv_med"] - row["pooled_fixed_on_cohort"]
                    lines.append(
                        f"  vs pooled {pooled_label} formula on balanced cohort: "
                        f"{row['pooled_fixed_on_cohort']:.3f}s "
                        f"({'↓' if delta < -0.01 else '↑'}{abs(delta):.3f}s)"
                    )
                lines.extend(stats["binned_lines"])
                lines.append(
                    f"  Ratio {to_ev}/{from_ev}: median {stats['r_med']:.3f} "
                    f"(IQR {stats['r_p25']:.3f}–{stats['r_p75']:.3f})"
                )
                lines.append(
                    f"  Median times in sample: {from_ev} {stats['from_med']:.2f}s, "
                    f"{to_ev} {stats['to_med']:.2f}s"
                )
                lines.append("")

            order = EVENT_ORDER[group]
            missing = []
            for from_ev, to_ev in permutations(order, 2):
                if (gender, group, from_ev, to_ev) not in reported_keys:
                    n_bal = balanced_n.get((gender, group, from_ev, to_ev), 0)
                    missing.append(f"{from_ev} -> {to_ev} (n={n_bal})")
            if missing:
                lines.extend([
                    f"Not reported for {group} — {gender} (n < {MIN_PAIR_N} balanced athlete-season pairs):",
                    "  " + ", ".join(missing),
                    "",
                ])

    lines.extend(["Pair summary tables", "-------------------", ""])
    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            sub = [s for s in summary_rows if s[1] == group and s[0] == gender]
            if not sub:
                continue
            lines.append(f"{group} — {gender} (balanced, {pop_desc})")
            lines.append(
                f"  {'From':<8} {'To':<8} {'n':>5} {'r':>6}  {'CV Med|err|':>11}  Formula"
            )
            for _, _, from_ev, to_ev, stats in sorted(sub, key=lambda x: (x[2], x[3])):
                formula_one_line = stats["formula"].split("\n")[0]
                lines.append(
                    f"  {from_ev:<8} {to_ev:<8} {stats['n']:>5} {stats['r']:>6.3f}  "
                    f"{stats['cv_med']:>10.3f}s  {formula_one_line}"
                )
            lines.append("")

    improvements = []
    for row in rows:
        if row.get("pooled_fixed_on_cohort") is not None:
            improvements.append(
                (
                    row["gender"],
                    row["event_group"],
                    row["from_event"],
                    row["to_event"],
                    row["pooled_fixed_on_cohort"],
                    row["best_cv"],
                )
            )
    if improvements:
        lines.extend([
            f"Accuracy vs pooled {pooled_label} models (applied to balanced cohort)",
            "-" * (48 + len(pooled_label)),
            "",
            "  Pair                    Pooled on balanced   Balanced CV   Improvement",
            "  ----------------------  ------------------   -----------   -----------",
        ])
        improvements.sort(key=lambda x: x[4] - x[5], reverse=True)
        for gender, group, from_ev, to_ev, pooled, bal in improvements:
            delta = bal - pooled
            lines.append(
                f"  {gender} {group} {from_ev}->{to_ev}  "
                f"{pooled:>17.3f}s  {bal:>10.3f}s       {delta:+.3f}s"
            )
        lines.append("")

    lines.extend([
        "When to use these vs other formula sets",
        "---------------------------------------",
        "  • Athlete has 5 or 6 races AND wa_spread < 50 WA → time_models_balanced_5_6_race_formulas.txt",
        "  • Athlete has exactly 6 races AND wa_spread < 50 WA → time_models_balanced_6races_formulas.txt",
        "  • Athlete has 6 races AND wa_spread ≥ 50 WA → time_models_specialized_6races_formulas.txt",
        "  • Specialization unknown → pooled race-count formula files",
        "",
        "Source: time_models/specialized_time_models/generate_balanced_6races_formula_report.py",
    ])
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Exactly 6 races (existing file)
    rows6, n6 = build_balanced_rows([6], "6")
    out6 = OUTPUT_ROOT / "time_models_balanced_6races_formulas.txt"
    write_report(
        rows6,
        n6,
        out6,
        pop_title="6 Races Per Season",
        pop_desc="exactly 6 valid results",
        use_line=(
            "Apply these formulas when the athlete has exactly 6 season races in the event "
            "group AND is balanced (wa_spread < 50 WA)."
        ),
        pooled_label="6-race",
    )
    print(f"Wrote {out6} ({len(rows6)} pairs)")

    # 5 or 6 races (requested file)
    rows56, n56 = build_balanced_rows([5, 6], "5_or_6")
    out56 = OUTPUT_ROOT / "time_models_balanced_5_6_race_formulas.txt"
    write_report(
        rows56,
        n56,
        out56,
        pop_title="5 or 6 Races Per Season",
        pop_desc="exactly 5 or 6 valid results",
        use_line=(
            "Apply these formulas when the athlete has 5 or 6 season races in the event "
            "group AND is balanced (wa_spread < 50 WA)."
        ),
        pooled_label="5-or-6-race",
    )
    print(f"Wrote {out56} ({len(rows56)} pairs)")


if __name__ == "__main__":
    main()
