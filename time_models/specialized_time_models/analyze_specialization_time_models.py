"""Test whether WA-based event-group specialization affects cross-event time model accuracy.

Population: athlete-seasons with exactly 6 valid results in the event group.
Specialization: spread of season PB World Athletics points across events in the group.

Outputs to time_models/specialized_time_models/
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[1]
RELAYS_ROOT = PROJECT_ROOT / "Relays_Findings"
OUTPUT_ROOT = TIME_MODELS_ROOT / "specialized_time_models"
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"

sys.path.insert(0, str(RELAYS_ROOT))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))

from relay_rq1_data import points_col, season_rows  # noqa: E402
from build_cross_event_time_models import GROUP_CONFIG, linreg, parse_performance  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
    cross_validate,
    evaluate_pair,
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
)
from compare_time_models_by_race_count import load_athlete_season_pbs  # noqa: E402

RACE_COUNT = 6
SPREAD_THRESHOLD = 50.0
MIN_PAIR_N = 20
MIN_SUBGROUP_N = 12

PRED_MAP = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}


@dataclass
class AthleteSeasonProfile:
    key: str
    gender: str
    event_group: str
    race_count: int
    event_times: dict[str, float]
    event_wa: dict[str, float]
    wa_spread: float
    best_event: str
    events_competed: int


def load_athlete_season_profiles(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
    race_count: int,
) -> dict[str, AthleteSeasonProfile]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    pcol = points_col(gender)
    season_results: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)

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
            event_name = id_to_name[event_id]
            season_results[key][result_id] = (event_name, t, wa)

    profiles: dict[str, AthleteSeasonProfile] = {}
    for key, results in season_results.items():
        if len(results) != race_count:
            continue

        event_times: dict[str, float] = {}
        event_wa: dict[str, float] = {}
        for event_name, t, wa in results.values():
            prev_t = event_times.get(event_name)
            if prev_t is None or t < prev_t:
                event_times[event_name] = t
                event_wa[event_name] = wa

        if len(event_times) < 2:
            continue

        wa_vals = list(event_wa.values())
        spread = max(wa_vals) - min(wa_vals)
        best_event = max(event_wa, key=lambda e: event_wa[e])
        profiles[key] = AthleteSeasonProfile(
            key=key,
            gender=gender,
            event_group="",
            race_count=race_count,
            event_times=event_times,
            event_wa=event_wa,
            wa_spread=spread,
            best_event=best_event,
            events_competed=len(event_times),
        )
    return profiles


def specialization_label(spread: float, threshold: float = SPREAD_THRESHOLD) -> str:
    return "specialized" if spread >= threshold else "balanced"


def pair_wa_gap(profile: AthleteSeasonProfile, from_ev: str, to_ev: str) -> float:
    return profile.event_wa[from_ev] - profile.event_wa[to_ev]


def load_best6_models() -> dict[tuple[str, str, str, str], dict]:
    path = MODEL_SEARCH_ROOT / "race_count" / "best_time_models_by_race_count.csv"
    out: dict[tuple[str, str, str, str], dict] = {}
    for row in csv.DictReader(open(path)):
        if row["race_count_filter"] != "6":
            continue
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        out[key] = row
    return out


def cv_for_pairs(
    pairs: list[tuple[float, float]],
    model_name: str = "linear_ols",
) -> float | None:
    if len(pairs) < MIN_SUBGROUP_N:
        return None
    fit_fn, pred_fn = PRED_MAP.get(model_name, (fit_linear, pred_linear))
    if fit_fn is None:
        return None
    cv_med, _ = cross_validate(pairs, fit_fn, pred_fn)
    return cv_med


def cv_best_model(pairs: list[tuple[float, float]]) -> tuple[str | None, float | None]:
    if len(pairs) < MIN_SUBGROUP_N:
        return None, None
    results = evaluate_pair(pairs)
    winner = min(results, key=lambda r: r.cv_median_abs)
    return winner.name, winner.cv_median_abs


def median_abs_error_with_params(
    pairs: list[tuple[float, float]],
    model_name: str,
    params: dict,
) -> float | None:
    _, pred_fn = PRED_MAP.get(model_name, (fit_linear, pred_linear))
    if pred_fn is None:
        return None
    preds = [pred_fn(x, params) for x, _ in pairs]
    actual = [y for _, y in pairs]
    med, _ = metrics(actual, preds)
    return med


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    best6 = load_best6_models()

    profile_rows: list[dict] = []
    comparison_rows: list[dict] = []
    report_lines = [
        "Specialization vs. Cross-Event Time Model Accuracy",
        "===================================================",
        "",
        "Research question: Among athlete-seasons with exactly 6 races in an event",
        "group, do athletes with stronger WA-based specialization (larger spread",
        "between their best and weakest events) have worse or better cross-event",
        "time predictions?",
        "",
        "Population:",
        f"  • Exactly {RACE_COUNT} valid discipline results in the event group (2024–2026).",
        "  • Athlete-season PB per event (fastest time, WA > 0).",
        "",
        "Specialization metrics (per athlete-season, within group):",
        "  • wa_spread = max(WA) − min(WA) across season PBs in group events",
        "  • best_event = event with highest season PB WA",
        f"  • specialized: wa_spread ≥ {SPREAD_THRESHOLD:.0f} WA points",
        f"  • balanced: wa_spread < {SPREAD_THRESHOLD:.0f} WA points",
        "  • pair_wa_gap = WA(from) − WA(to) for a specific prediction pair",
        "",
        "Model evaluation:",
        f"  • 5-fold CV (seed={CV_SEED}), same candidates as model_search",
        f"  • Compare linear OLS and best 6-race model per pair on each cohort",
        f"  • Minimum n={MIN_SUBGROUP_N} for subgroup CV; n={MIN_PAIR_N} for pair reporting",
        "",
    ]

    all_profiles_by_group: dict[tuple[str, str], list[AthleteSeasonProfile]] = defaultdict(list)

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            profiles = load_athlete_season_profiles(folder, prefix, gender, events, RACE_COUNT)
            for p in profiles.values():
                p.event_group = event_group
            all_profiles_by_group[(gender, event_group)].extend(profiles.values())

            for p in profiles.values():
                profile_rows.append(
                    {
                        "athlete_season_key": p.key,
                        "gender": gender,
                        "event_group": event_group,
                        "race_count": p.race_count,
                        "events_competed": p.events_competed,
                        "wa_spread": round(p.wa_spread, 1),
                        "best_event": p.best_event,
                        "specialization": specialization_label(p.wa_spread),
                        "wa_by_event": json.dumps({k: round(v, 1) for k, v in p.event_wa.items()}),
                    }
                )

            for from_ev, to_ev in permutations(order, 2):
                pair_profiles = [
                    p
                    for p in profiles.values()
                    if from_ev in p.event_times and to_ev in p.event_times
                ]
                if len(pair_profiles) < MIN_PAIR_N:
                    continue

                all_pairs = [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_profiles]
                spec_pairs = [
                    (p.event_times[from_ev], p.event_times[to_ev])
                    for p in pair_profiles
                    if specialization_label(p.wa_spread) == "specialized"
                ]
                bal_pairs = [
                    (p.event_times[from_ev], p.event_times[to_ev])
                    for p in pair_profiles
                    if specialization_label(p.wa_spread) == "balanced"
                ]

                # Predicting toward non-best event vs toward best event
                to_non_best = [
                    (p.event_times[from_ev], p.event_times[to_ev])
                    for p in pair_profiles
                    if to_ev != p.best_event
                ]
                to_best = [
                    (p.event_times[from_ev], p.event_times[to_ev])
                    for p in pair_profiles
                    if to_ev == p.best_event
                ]

                # Large pair gap: from event is much stronger WA than to (|gap| >= 50)
                large_gap_pairs = [
                    (p.event_times[from_ev], p.event_times[to_ev])
                    for p in pair_profiles
                    if pair_wa_gap(p, from_ev, to_ev) >= SPREAD_THRESHOLD
                ]
                small_gap_pairs = [
                    (p.event_times[from_ev], p.event_times[to_ev])
                    for p in pair_profiles
                    if abs(pair_wa_gap(p, from_ev, to_ev)) < SPREAD_THRESHOLD
                ]

                linear_all = cv_for_pairs(all_pairs, "linear_ols")
                linear_spec = cv_for_pairs(spec_pairs, "linear_ols")
                linear_bal = cv_for_pairs(bal_pairs, "linear_ols")
                best_name_all, best_cv_all = cv_best_model(all_pairs)

                model_key = (gender, event_group, from_ev, to_ev)
                best6_row = best6.get(model_key)
                best6_model = best6_row["best_model"] if best6_row else None
                best6_cv = float(best6_row["best_cv_median_abs"]) if best6_row else None

                row_base = {
                    "gender": gender,
                    "event_group": event_group,
                    "from_event": from_ev,
                    "to_event": to_ev,
                    "n_all": len(all_pairs),
                    "n_specialized": len(spec_pairs),
                    "n_balanced": len(bal_pairs),
                    "linear_cv_all": linear_all,
                    "linear_cv_specialized": linear_spec,
                    "linear_cv_balanced": linear_bal,
                    "best_model_all": best_name_all,
                    "best_cv_all": best_cv_all,
                    "best6_race_model": best6_model,
                    "best6_race_cv": best6_cv,
                }

                cohorts = [
                    ("all", all_pairs, linear_all, best_cv_all),
                    ("specialized", spec_pairs, linear_spec, cv_for_pairs(spec_pairs, "linear_ols")),
                    ("balanced", bal_pairs, linear_bal, cv_for_pairs(bal_pairs, "linear_ols")),
                    ("predict_to_non_best", to_non_best, cv_for_pairs(to_non_best, "linear_ols"), None),
                    ("predict_to_best", to_best, cv_for_pairs(to_best, "linear_ols"), None),
                    ("pair_gap_large", large_gap_pairs, cv_for_pairs(large_gap_pairs, "linear_ols"), None),
                    ("pair_gap_small", small_gap_pairs, cv_for_pairs(small_gap_pairs, "linear_ols"), None),
                ]

                for cohort_name, pairs, lin_cv, _ in cohorts:
                    if len(pairs) < MIN_SUBGROUP_N:
                        continue
                    bm, bcv = cv_best_model(pairs)
                    comparison_rows.append(
                        {
                            **row_base,
                            "cohort": cohort_name,
                            "n": len(pairs),
                            "linear_cv": round(lin_cv, 4) if lin_cv is not None else None,
                            "best_model": bm,
                            "best_cv": round(bcv, 4) if bcv is not None else None,
                        }
                    )

    # Profile summary
    report_lines.extend(["Cohort sizes (6-race athlete-seasons)", "--------------------------------"])
    for (gender, group), profs in sorted(all_profiles_by_group.items()):
        if not profs:
            continue
        spreads = [p.wa_spread for p in profs]
        n_spec = sum(1 for p in profs if specialization_label(p.wa_spread) == "specialized")
        report_lines.append(
            f"  {gender} {group}: n={len(profs)}  "
            f"median spread={statistics.median(spreads):.1f} WA  "
            f"specialized={n_spec} ({100*n_spec/len(profs):.0f}%)  "
            f"balanced={len(profs)-n_spec}"
        )

    report_lines.extend(["", "Pair-level comparison (linear OLS CV median |error|)", "-" * 55])
    report_lines.append(
        f"{'Gender':<7} {'Group':<9} {'Pair':<12} {'n':>4} "
        f"{'All':>7} {'Spec':>7} {'Bal':>7} {'Δ(S-B)':>8} {'Better?'}"
    )

    pair_keys = sorted(
        {(r["gender"], r["event_group"], r["from_event"], r["to_event"]) for r in comparison_rows}
    )
    spec_better_count = 0
    bal_better_count = 0
    comparable = 0
    deltas = []

    for gender, group, from_ev, to_ev in pair_keys:
        rows = [
            r
            for r in comparison_rows
            if r["gender"] == gender
            and r["event_group"] == group
            and r["from_event"] == from_ev
            and r["to_event"] == to_ev
        ]
        by_cohort = {r["cohort"]: r for r in rows}
        if "specialized" not in by_cohort or "balanced" not in by_cohort:
            continue
        spec_cv = by_cohort["specialized"]["linear_cv"]
        bal_cv = by_cohort["balanced"]["linear_cv"]
        all_cv = by_cohort.get("all", {}).get("linear_cv")
        n_all = by_cohort.get("all", {}).get("n", "")
        if spec_cv is None or bal_cv is None:
            continue
        delta = spec_cv - bal_cv
        comparable += 1
        deltas.append(delta)
        if delta < -0.01:
            spec_better_count += 1
            better = "specialized"
        elif delta > 0.01:
            bal_better_count += 1
            better = "balanced"
        else:
            better = "~same"
        pair = f"{from_ev}->{to_ev}"
        report_lines.append(
            f"{gender:<7} {group:<9} {pair:<12} {n_all:>4} "
            f"{all_cv:>6.3f}s {spec_cv:>6.3f}s {bal_cv:>6.3f}s {delta:>+7.3f}s {better}"
        )

    if comparable:
        report_lines.extend(
            [
                "",
                "Headline — specialization spread ≥50 WA vs balanced:",
                f"  Pairs with both cohorts n≥{MIN_SUBGROUP_N}: {comparable}",
                f"  Balanced athletes more predictable: {bal_better_count}/{comparable}",
                f"  Specialized athletes more predictable: {spec_better_count}/{comparable}",
                f"  Mean Δ (specialized − balanced) linear CV: {statistics.mean(deltas):+.3f}s",
                f"  Median Δ: {statistics.median(deltas):+.3f}s",
            ]
        )

    report_lines.extend(["", "Predicting toward best vs non-best event (linear CV)", "-" * 50])
    for gender, group, from_ev, to_ev in pair_keys:
        rows = {
            r["cohort"]: r
            for r in comparison_rows
            if r["gender"] == gender
            and r["event_group"] == group
            and r["from_event"] == from_ev
            and r["to_event"] == to_ev
        }
        if "predict_to_non_best" not in rows or "predict_to_best" not in rows:
            continue
        nb = rows["predict_to_non_best"]["linear_cv"]
        b = rows["predict_to_best"]["linear_cv"]
        pair = f"{gender} {group} {from_ev}->{to_ev}"
        report_lines.append(
            f"  {pair}: to non-best {nb:.3f}s (n={rows['predict_to_non_best']['n']})  "
            f"to best {b:.3f}s (n={rows['predict_to_best']['n']})  "
            f"Δ={nb-b:+.3f}s"
        )

    report_lines.extend(["", "Interpretation", "--------------"])
    if deltas and statistics.median(deltas) > 0.05:
        report_lines.append(
            "Specialized athletes (large WA spread within the group) show LARGER cross-event"
            " time prediction errors. Their weaker events are harder to project from"
            " stronger ones — WA specialization signals misfit with group-average time"
            " relationships."
        )
    elif deltas and statistics.median(deltas) < -0.05:
        report_lines.append(
            "Specialized athletes show SMALLER prediction errors than balanced athletes."
            " High-volume specialists may still follow consistent time scaling within"
            " the group despite WA concentration in one event."
        )
    else:
        report_lines.append(
            "Little systematic difference between specialized and balanced athletes at"
            " the 50-WA spread threshold. Specialization within the group does not"
            " materially change time-model accuracy for the 6-race cohort."
        )

    report_lines.extend(
        [
            "",
            "Caveats:",
            "  • WA spread uses PB marks in the group; it does not require running all 3 events.",
            "  • Subgroup sample sizes are small after splitting 6-race seasons.",
            "  • 50-WA threshold is a rule of thumb from prior WA specialization discussion.",
            "",
            "Source: time_models/specialized_time_models/analyze_specialization_time_models.py",
        ]
    )

    with open(OUTPUT_ROOT / "athlete_season_specialization_6races.csv", "w", newline="") as f:
        if profile_rows:
            w = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
            w.writeheader()
            w.writerows(profile_rows)

    with open(OUTPUT_ROOT / "specialization_model_comparison.csv", "w", newline="") as f:
        if comparison_rows:
            w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader()
            w.writerows(comparison_rows)

    (OUTPUT_ROOT / "specialization_time_model_report.txt").write_text(
        "\n".join(report_lines).rstrip() + "\n"
    )
    print(f"Wrote specialization analysis to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
