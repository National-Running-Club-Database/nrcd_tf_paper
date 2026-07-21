"""Combine Short/Long (Mid/Long) specialization with balanced/specialized.

Question: Does knowing BOTH axes improve cross-event time models beyond
knowing either axis alone?

Axes
----
  Short/Long (requires season PBs in all 3 group events):
    Sprints — Short: |WA100−WA200| < |WA200−WA400|
    Sprints — Long:  reverse
    Distance — Mid (short-distance): |WA800−WA1500| < |WA1500−WA5000|
    Distance — Long: reverse

  Balanced/Specialized (old definition):
    specialized: wa_spread ≥ 50 across season PBs in the group
    balanced: wa_spread < 50

Routing strategies compared (on classifiable athletes):
  1. pooled          — one formula for all classifiable
  2. bal_spec_only   — route by balanced vs specialized
  3. short_long_only — route by Short/Mid vs Long
  4. joint           — route by (bal/spec × short/long) when train n ≥ 12,
                       else fall back to short/long, then bal/spec, then pooled
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
SHORT_LONG_ROOT = TIME_MODELS_ROOT / "Short_Long_Specialization_Time_Models"

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(SHORT_LONG_ROOT))

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
    specialization_label,
)
from analyze_short_long_specialization import (  # noqa: E402
    COHORT_A,
    COHORT_B,
    COHORT_LABELS,
    classify_short_long,
    load_profiles_multi,
)

MIN_COHORT_N = 12
MIN_REPORT_N = 20
SPREAD_THRESHOLD = 50.0

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


def fit_best(pairs: list[tuple[float, float]]) -> tuple[str, dict, Callable] | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    winner = min(evaluate_pair(pairs), key=lambda r: r.cv_median_abs)
    fit_fn, pred_fn = FIT_PRED.get(winner.name, (fit_linear, pred_linear))
    if fit_fn is None or pred_fn is None:
        return None
    return winner.name, fit_fn(pairs), pred_fn


def predict_with(fit: tuple[str, dict, Callable] | None, x: float, fallback) -> float:
    if fit is None:
        return fallback(x)
    _, params, pred_fn = fit
    return pred_fn(x, params)


def cv_strategies(
    records: list[tuple[float, float, str, str]],
) -> dict[str, float] | None:
    """records: (from_t, to_t, bal_spec_label, short_long_label)"""
    if len(records) < MIN_REPORT_N:
        return None

    n = len(records)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)

    buckets: dict[str, tuple[list[float], list[float]]] = {
        "pooled": ([], []),
        "bal_spec_only": ([], []),
        "short_long_only": ([], []),
        "joint": ([], []),
    }

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else n
        test_idx = set(idx[start:end])
        train = [records[i] for i in range(n) if i not in test_idx]
        test = [records[i] for i in test_idx]
        if len(train) < 10:
            continue

        train_pairs = [(a, b) for a, b, _, _ in train]
        pooled = fit_best(train_pairs)
        if pooled is None:
            continue
        _, pooled_params, pooled_pred = pooled

        def pooled_fn(x: float) -> float:
            return pooled_pred(x, pooled_params)

        # bal/spec fits
        bal_spec_fits: dict[str, tuple] = {}
        for lab in ("balanced", "specialized"):
            subset = [(a, b) for a, b, bs, _ in train if bs == lab]
            fit = fit_best(subset)
            if fit:
                bal_spec_fits[lab] = fit

        # short/long fits
        sl_fits: dict[str, tuple] = {}
        for _, _, _, sl in train:
            if sl not in sl_fits:
                subset = [(a, b) for a, b, _, s in train if s == sl]
                fit = fit_best(subset)
                if fit:
                    sl_fits[sl] = fit

        # joint fits
        joint_fits: dict[tuple[str, str], tuple] = {}
        for bs in ("balanced", "specialized"):
            for sl in {s for _, _, _, s in train}:
                subset = [(a, b) for a, b, b2, s2 in train if b2 == bs and s2 == sl]
                fit = fit_best(subset)
                if fit:
                    joint_fits[(bs, sl)] = fit

        for x, y, bs, sl in test:
            # pooled
            buckets["pooled"][0].append(y)
            buckets["pooled"][1].append(pooled_fn(x))

            # bal_spec_only
            buckets["bal_spec_only"][0].append(y)
            buckets["bal_spec_only"][1].append(
                predict_with(bal_spec_fits.get(bs), x, pooled_fn)
            )

            # short_long_only
            buckets["short_long_only"][0].append(y)
            buckets["short_long_only"][1].append(
                predict_with(sl_fits.get(sl), x, pooled_fn)
            )

            # joint with fallbacks: joint → short/long → bal/spec → pooled
            buckets["joint"][0].append(y)
            if (bs, sl) in joint_fits:
                buckets["joint"][1].append(predict_with(joint_fits[(bs, sl)], x, pooled_fn))
            elif sl in sl_fits:
                buckets["joint"][1].append(predict_with(sl_fits[sl], x, pooled_fn))
            elif bs in bal_spec_fits:
                buckets["joint"][1].append(predict_with(bal_spec_fits[bs], x, pooled_fn))
            else:
                buckets["joint"][1].append(pooled_fn(x))

    out: dict[str, float] = {}
    for name, (actual, pred) in buckets.items():
        if not actual:
            return None
        out[name], _ = metrics(actual, pred)
    return out


def cv_linear_subgroup(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    from compare_time_model_candidates import cross_validate

    return cross_validate(pairs, fit_linear, pred_linear)[0]


def run_population(pop_label: str, race_counts: list[int]) -> dict:
    profile_rows: list[dict] = []
    crosstab_rows: list[dict] = []
    strategy_rows: list[dict] = []
    within_rows: list[dict] = []
    classified_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        ca, cb = COHORT_A[event_group], COHORT_B[event_group]

        for gender in ("Men", "Women"):
            profiles = load_profiles_multi(folder, prefix, gender, events, race_counts)
            for p in profiles.values():
                p.event_group = event_group

            cell_counts: dict[str, int] = defaultdict(int)
            for p in profiles.values():
                bs = specialization_label(p.wa_spread)
                sl = classify_short_long(p, event_group)
                classified_counts[(gender, event_group)]["total"] += 1
                classified_counts[(gender, event_group)][bs] += 1
                if sl is None:
                    classified_counts[(gender, event_group)]["missing_3events"] += 1
                    joint = "unclassifiable"
                elif sl == "tie":
                    classified_counts[(gender, event_group)]["tie"] += 1
                    joint = f"{bs}|tie"
                else:
                    classified_counts[(gender, event_group)][sl] += 1
                    joint = f"{bs}|{sl}"
                    cell_counts[joint] += 1

                profile_rows.append(
                    {
                        "population": pop_label,
                        "athlete_season_key": p.key,
                        "gender": gender,
                        "event_group": event_group,
                        "race_count": p.race_count,
                        "events_competed": p.events_competed,
                        "bal_spec": bs,
                        "wa_spread": round(p.wa_spread, 1),
                        "short_long": sl or "unclassifiable",
                        "joint_label": joint,
                        "best_event": p.best_event,
                        "wa_by_event": json.dumps({k: round(v, 1) for k, v in p.event_wa.items()}),
                    }
                )

            for joint, n in sorted(cell_counts.items()):
                crosstab_rows.append(
                    {
                        "population": pop_label,
                        "gender": gender,
                        "event_group": event_group,
                        "joint_label": joint,
                        "n": n,
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

                # Classifiable on short/long
                classifiable = []
                for p in pair_profs:
                    sl = classify_short_long(p, event_group)
                    if sl in (ca, cb):
                        classifiable.append(
                            (
                                p.event_times[from_ev],
                                p.event_times[to_ev],
                                specialization_label(p.wa_spread),
                                sl,
                                p,
                            )
                        )

                if len(classifiable) < MIN_REPORT_N:
                    continue

                _, _, r, _, _ = linreg([(a, b) for a, b, _, _, _ in classifiable])
                records = [(a, b, bs, sl) for a, b, bs, sl, _ in classifiable]
                cvs = cv_strategies(records)
                if not cvs:
                    continue

                n_bal = sum(1 for _, _, bs, _, _ in classifiable if bs == "balanced")
                n_spec = sum(1 for _, _, bs, _, _ in classifiable if bs == "specialized")
                n_a = sum(1 for _, _, _, sl, _ in classifiable if sl == ca)
                n_b = sum(1 for _, _, _, sl, _ in classifiable if sl == cb)

                strategy_rows.append(
                    {
                        "population": pop_label,
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n_classifiable": len(classifiable),
                        "n_balanced": n_bal,
                        "n_specialized": n_spec,
                        "n_short_mid": n_a,
                        "n_long": n_b,
                        "r": round(r, 4),
                        "cv_pooled": round(cvs["pooled"], 4),
                        "cv_bal_spec_only": round(cvs["bal_spec_only"], 4),
                        "cv_short_long_only": round(cvs["short_long_only"], 4),
                        "cv_joint": round(cvs["joint"], 4),
                        "delta_bal_spec_vs_pooled": round(cvs["pooled"] - cvs["bal_spec_only"], 4),
                        "delta_short_long_vs_pooled": round(cvs["pooled"] - cvs["short_long_only"], 4),
                        "delta_joint_vs_pooled": round(cvs["pooled"] - cvs["joint"], 4),
                        "delta_joint_vs_short_long": round(
                            cvs["short_long_only"] - cvs["joint"], 4
                        ),
                        "delta_joint_vs_bal_spec": round(cvs["bal_spec_only"] - cvs["joint"], 4),
                        "best_strategy": min(
                            [
                                ("pooled", cvs["pooled"]),
                                ("bal_spec_only", cvs["bal_spec_only"]),
                                ("short_long_only", cvs["short_long_only"]),
                                ("joint", cvs["joint"]),
                            ],
                            key=lambda x: x[1],
                        )[0],
                    }
                )

                # Within short/long arms: does bal/spec still matter?
                for sl_lab in (ca, cb):
                    for bs_lab in ("balanced", "specialized"):
                        pairs = [
                            (a, b)
                            for a, b, bs, sl, _ in classifiable
                            if sl == sl_lab and bs == bs_lab
                        ]
                        lin = cv_linear_subgroup(pairs)
                        if lin is None:
                            continue
                        within_rows.append(
                            {
                                "population": pop_label,
                                "gender": gender,
                                "event_group": event_group,
                                "from_event": from_ev,
                                "to_event": to_ev,
                                "short_long": sl_lab,
                                "bal_spec": bs_lab,
                                "n": len(pairs),
                                "linear_cv": round(lin, 4),
                            }
                        )

    return {
        "pop_label": pop_label,
        "profile_rows": profile_rows,
        "crosstab_rows": crosstab_rows,
        "strategy_rows": strategy_rows,
        "within_rows": within_rows,
        "classified_counts": classified_counts,
    }


def summarize_strategies(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "joint_wins_vs_pooled": 0,
            "joint_wins_vs_sl": 0,
            "sl_wins_vs_pooled": 0,
            "bs_wins_vs_pooled": 0,
            "best_counts": {},
            "mean_joint_vs_sl": None,
            "mean_joint_vs_pooled": None,
        }
    best_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        best_counts[r["best_strategy"]] += 1
    return {
        "n": len(rows),
        "joint_wins_vs_pooled": sum(1 for r in rows if r["delta_joint_vs_pooled"] > 0.01),
        "joint_wins_vs_sl": sum(1 for r in rows if r["delta_joint_vs_short_long"] > 0.01),
        "sl_wins_vs_pooled": sum(1 for r in rows if r["delta_short_long_vs_pooled"] > 0.01),
        "bs_wins_vs_pooled": sum(1 for r in rows if r["delta_bal_spec_vs_pooled"] > 0.01),
        "best_counts": dict(best_counts),
        "mean_joint_vs_sl": statistics.mean(r["delta_joint_vs_short_long"] for r in rows),
        "mean_joint_vs_pooled": statistics.mean(r["delta_joint_vs_pooled"] for r in rows),
        "mean_sl_vs_pooled": statistics.mean(r["delta_short_long_vs_pooled"] for r in rows),
        "mean_bs_vs_pooled": statistics.mean(r["delta_bal_spec_vs_pooled"] for r in rows),
    }


def section_report(result: dict) -> list[str]:
    pop = result["pop_label"]
    pop_desc = "exactly 6 races" if pop == "6" else "5 or 6 races"
    lines = [
        f"POPULATION: {pop_desc}",
        "=" * (12 + len(pop_desc)),
        "",
        "Joint crosstab (bal/spec × short/long) among classifiable athletes",
        "------------------------------------------------------------------",
    ]
    by_group: dict[tuple, list] = defaultdict(list)
    for row in result["crosstab_rows"]:
        by_group[(row["gender"], row["event_group"])].append(row)
    for key in sorted(by_group):
        gender, group = key
        lines.append(f"  {gender} {group}:")
        for row in sorted(by_group[key], key=lambda r: r["joint_label"]):
            lines.append(f"    {row['joint_label']}: n={row['n']}")

    # Association note
    total_class = 0
    bal_class = 0
    for row in result["crosstab_rows"]:
        total_class += row["n"]
        if row["joint_label"].startswith("balanced|"):
            bal_class += row["n"]
    if total_class:
        lines.append("")
        lines.append(
            f"  Classifiable total={total_class}; balanced among them={bal_class} "
            f"({100*bal_class/total_class:.0f}%). "
            "Short/Long classification (needs all 3 events) heavily overlaps with "
            "old 'specialized' (≥50 WA spread)."
        )

    lines.extend([
        "",
        "Strategy CV on classifiable athletes (median |error|)",
        "-----------------------------------------------------",
        f"{'Pair':<36} {'n':>3} {'Pool':>7} {'BalSp':>7} {'Sh/L':>7} {'Joint':>7} Best",
    ])
    for row in sorted(
        result["strategy_rows"],
        key=lambda r: (r["event_group"], r["gender"], r["from_event"], r["to_event"]),
    ):
        pair = f"{row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']}"
        lines.append(
            f"{pair:<36} {row['n_classifiable']:>3} "
            f"{row['cv_pooled']:>6.3f}s {row['cv_bal_spec_only']:>6.3f}s "
            f"{row['cv_short_long_only']:>6.3f}s {row['cv_joint']:>6.3f}s "
            f"{row['best_strategy']}"
        )

    stats = summarize_strategies(result["strategy_rows"])
    result["_stats"] = stats
    lines.append("")
    if stats["n"]:
        lines.extend([
            "Strategy summary",
            "----------------",
            f"  Pairs compared: {stats['n']}",
            f"  Best strategy counts: {stats['best_counts']}",
            f"  short_long beats pooled: {stats['sl_wins_vs_pooled']}/{stats['n']} "
            f"(mean Δ {stats['mean_sl_vs_pooled']:+.3f}s)",
            f"  bal_spec beats pooled: {stats['bs_wins_vs_pooled']}/{stats['n']} "
            f"(mean Δ {stats['mean_bs_vs_pooled']:+.3f}s)",
            f"  joint beats pooled: {stats['joint_wins_vs_pooled']}/{stats['n']} "
            f"(mean Δ {stats['mean_joint_vs_pooled']:+.3f}s)",
            f"  joint beats short_long alone: {stats['joint_wins_vs_sl']}/{stats['n']} "
            f"(mean Δ {stats['mean_joint_vs_sl']:+.3f}s)",
        ])
    else:
        lines.append("  No pairs with enough classifiable athletes.")

    # Within-arm bal/spec
    within = result["within_rows"]
    lines.extend([
        "",
        "Within short/long arms: balanced vs specialized (linear CV, n≥12)",
        "-----------------------------------------------------------------",
    ])
    if within:
        for row in sorted(
            within,
            key=lambda r: (
                r["event_group"],
                r["gender"],
                r["short_long"],
                r["from_event"],
                r["to_event"],
                r["bal_spec"],
            ),
        ):
            lines.append(
                f"  {row['gender']} {row['event_group']} {row['from_event']}->{row['to_event']} "
                f"[{COHORT_LABELS.get(row['short_long'], row['short_long'])} ∩ {row['bal_spec']}] "
                f"n={row['n']}  CV={row['linear_cv']:.3f}s"
            )
    else:
        lines.append(
            "  No joint cells reach n≥12. Balanced ∩ Short/Long is essentially empty;"
            " within-arm bal/spec splits cannot be estimated."
        )
    lines.append("")
    return lines


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [run_population(label, rcs) for label, rcs in POPULATIONS]

    report = [
        "Combined Specialization Axes vs Time Model Accuracy",
        "====================================================",
        "",
        "Research question: If we know BOTH",
        "  (1) Short vs Long sprints / Mid (short-distance) vs Long distance, AND",
        "  (2) Balanced vs Specialized (wa_spread ≥ 50),",
        "do time predictions improve beyond knowing either axis alone?",
        "",
        "Routing strategies (classifiable athletes = have all 3 event PBs):",
        "  pooled          — one model for all classifiable",
        "  bal_spec_only   — route by balanced vs specialized",
        "  short_long_only — route by Short/Mid vs Long",
        "  joint           — route by both axes (fallback: short/long → bal/spec → pooled)",
        "",
        f"Minimum cohort train n={MIN_COHORT_N}; pair report n={MIN_REPORT_N}.",
        "",
    ]
    for result in results:
        report.extend(section_report(result))

    r56 = next(r for r in results if r["pop_label"] == "5_or_6")
    r6 = next(r for r in results if r["pop_label"] == "6")
    s56, s6 = r56["_stats"], r6["_stats"]

    report.extend(["Interpretation", "--------------"])
    report.append(
        "Among athletes classifiable on Short/Long (all 3 event PBs), nearly all are"
        " already 'specialized' under the old wa_spread≥50 rule. Balanced∩Short/Long"
        " cells are tiny (often n=0–3). The two axes are highly overlapping, not"
        " independent stratifiers."
    )
    if s56["n"]:
        bc = s56["best_counts"]
        report.append(
            f"At 5-or-6 races ({s56['n']} pairs): best strategy counts {bc}. "
            f"Joint beats short/long alone on {s56['joint_wins_vs_sl']}/{s56['n']} pairs "
            f"(mean Δ {s56['mean_joint_vs_sl']:+.3f}s)."
        )
        if s56["joint_wins_vs_sl"] / s56["n"] > 0.5 and (s56["mean_joint_vs_sl"] or 0) > 0.05:
            report.append(
                "Verdict: YES — knowing both axes adds meaningful accuracy beyond"
                " Short/Long alone."
            )
        elif s56["sl_wins_vs_pooled"] > s56["bs_wins_vs_pooled"] and s56["joint_wins_vs_sl"] <= max(
            1, s56["n"] // 4
        ):
            report.append(
                "Verdict: LIMITED — Short/Long status drives most of the gain over pooled"
                " models. Adding balanced/specialized on top rarely helps, because almost"
                " all Short/Long-classifiable athletes are already specialized."
            )
        elif s56["joint_wins_vs_pooled"] / s56["n"] > 0.5:
            report.append(
                "Verdict: PARTIAL — joint routing helps vs pooled for many pairs, but"
                " usually matches Short/Long-only routing (old axis adds little)."
            )
        else:
            report.append(
                "Verdict: NO CLEAR GAIN — combining axes does not reliably beat the"
                " better single-axis strategy (Short/Long)."
            )
    if s6["n"]:
        report.append(
            f"At exactly 6 races: {s6['n']} pairs; joint vs short/long wins "
            f"{s6['joint_wins_vs_sl']}/{s6['n']} (underpowered; Mid/Short arms small)."
        )

    report.extend([
        "",
        "Practical recommendation",
        "------------------------",
        "  • If athlete has all 3 event PBs: use Short/Long (Mid/Long) routing.",
        "  • Old balanced/specialized remains useful for athletes who raced only 2",
        "    events in the group (not Short/Long-classifiable).",
        "  • Do not expect a 4-way formula grid — balanced∩Short/Long is too rare.",
        "",
        "Caveats:",
        "  • Joint cells with balanced athletes lack n for dedicated formulas.",
        "  • Women's joint samples remain too small.",
        "  • Analysis restricted to athletes classifiable on Short/Long for fair",
        "    comparison of strategies that use that axis.",
        "",
        "Output files:",
        "  combined_specialization_report.txt",
        "  combined_specialization_findings.txt",
        "  athlete_season_combined_specialization.csv",
        "  combined_crosstab.csv",
        "  combined_strategy_comparison.csv",
        "  combined_within_arm_comparison.csv",
        "",
        "Source: Combined_Specialization_Time_Models/analyze_combined_specialization.py",
    ])

    findings = [
        "Combined Specialization Axes — Findings",
        "=======================================",
        "",
        "Question: Does knowing Short/Long (Mid/Long) AND balanced/specialized",
        "together improve time models beyond either alone?",
        "",
        "Core finding: the axes heavily overlap. Athletes with all 3 event PBs",
        "(required for Short/Long) are almost always 'specialized' (wa_spread≥50).",
        "",
    ]
    for result in results:
        pop = result["pop_label"]
        stats = result["_stats"]
        findings.append(f"{'5 or 6 races' if pop == '5_or_6' else 'Exactly 6 races'}:")
        # balanced share
        tot = bal = 0
        for row in result["crosstab_rows"]:
            tot += row["n"]
            if row["joint_label"].startswith("balanced|"):
                bal += row["n"]
        if tot:
            findings.append(
                f"  Classifiable: {tot}; balanced among them: {bal} ({100*bal/tot:.0f}%)"
            )
        if stats["n"]:
            findings.append(
                f"  Strategy pairs: {stats['n']}; best={stats['best_counts']}; "
                f"joint>short_long {stats['joint_wins_vs_sl']}/{stats['n']} "
                f"(mean Δ {stats['mean_joint_vs_sl']:+.3f}s); "
                f"short_long>pooled {stats['sl_wins_vs_pooled']}/{stats['n']}; "
                f"bal_spec>pooled {stats['bs_wins_vs_pooled']}/{stats['n']}"
            )
        findings.append("")
    findings.extend([
        "Recommendation: use Short/Long when all 3 PBs exist; use balanced/specialized",
        "when only 2 events are known. Combining both rarely adds a third gain.",
        "",
        "See combined_specialization_report.txt for pair-level CV tables.",
        "Source: analyze_combined_specialization.py",
    ])

    def write_csv(name: str, rows: list[dict]) -> None:
        path = OUTPUT_ROOT / name
        if not rows:
            path.write_text("")
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(
        "athlete_season_combined_specialization.csv",
        [r for res in results for r in res["profile_rows"]],
    )
    write_csv("combined_crosstab.csv", [r for res in results for r in res["crosstab_rows"]])
    write_csv(
        "combined_strategy_comparison.csv",
        [r for res in results for r in res["strategy_rows"]],
    )
    write_csv(
        "combined_within_arm_comparison.csv",
        [r for res in results for r in res["within_rows"]],
    )
    (OUTPUT_ROOT / "combined_specialization_report.txt").write_text("\n".join(report).rstrip() + "\n")
    (OUTPUT_ROOT / "combined_specialization_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )

    print(f"Wrote combined specialization analysis to {OUTPUT_ROOT}")
    for result in results:
        s = result["_stats"]
        print(
            f"  {result['pop_label']}: strategy_pairs={s['n']} "
            f"joint>sl={s['joint_wins_vs_sl']} best={s['best_counts']}"
        )


if __name__ == "__main__":
    main()
