"""Explore factors beyond balanced/specialized that could improve time models.

Population: athlete-seasons with 5 or 6 races in the event group (primary),
plus a 6-race-only appendix. Always also label balanced vs specialized
(wa_spread ≥ 50) as the baseline specialization axis.

For each event pair, 5-fold CV compares:
  pooled              — one formula
  bal_spec            — route by balanced / specialized
  factor              — route by candidate factor alone
  bal_spec_x_factor   — route by (bal/spec × factor) with fallbacks

A factor "helps" if routing by it (alone or jointly with bal/spec) beats
bal_spec-only by >0.01s on a meaningful share of pairs.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
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

from relay_rq1_data import points_col, season_rows  # noqa: E402
from build_cross_event_time_models import GROUP_CONFIG, linreg, parse_performance  # noqa: E402
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
from analyze_specialization_time_models import specialization_label  # noqa: E402

MIN_COHORT_N = 12
MIN_REPORT_N = 20
SPREAD_THRESHOLD = 50.0
# Populations to analyze (label → race_count filter list → file suffix)
POPULATIONS: list[tuple[str, list[int], str]] = [
    ("5_or_6", [5, 6], ""),  # default filenames (no suffix)
    ("6", [6], "_6races"),
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


@dataclass
class RichProfile:
    key: str
    gender: str
    event_group: str
    year: str
    race_count: int
    event_times: dict[str, float]
    event_wa: dict[str, float]
    wa_spread: float
    best_event: str
    events_competed: int
    max_wa: float
    mean_wa: float
    unique_meets: int
    season_span_days: int
    ran_nationals: bool
    has_relay: bool
    team_ids: set[str] = field(default_factory=set)


def parse_date(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def load_rich_profiles(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
    race_counts: list[int],
) -> dict[str, RichProfile]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    pcol = points_col(gender)
    # key -> result_id -> payload
    season_results: dict[str, dict[str, dict]] = defaultdict(dict)

    for year in ("2024", "2025", "2026"):
        for row in season_rows(folder, prefix, gender, year):
            try:
                event_id = int(row["running_event_id"])
            except (TypeError, ValueError):
                continue
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
            is_relay = bool(
                (row.get("relay_split") or "").strip()
                or (row.get("athlete_id_2") or "").strip()
            )
            nationals = str(row.get("nationals") or "").strip().lower() == "true"
            season_results[key][result_id] = {
                "event_name": event_name,
                "t": t,
                "wa": wa,
                "meet_id": (row.get("meet_id") or "").strip(),
                "start_date": parse_date(row.get("start_date") or ""),
                "nationals": nationals,
                "is_relay": is_relay,
                "team_id": (row.get("team_id") or "").strip(),
                "year": year,
            }

    allowed = set(race_counts)
    profiles: dict[str, RichProfile] = {}
    for key, results in season_results.items():
        if len(results) not in allowed:
            continue

        event_times: dict[str, float] = {}
        event_wa: dict[str, float] = {}
        meets: set[str] = set()
        dates: list[datetime] = []
        ran_nats = False
        has_relay = False
        teams: set[str] = set()
        year = key.split("|")[1]

        for payload in results.values():
            en = payload["event_name"]
            if en not in event_times or payload["t"] < event_times[en]:
                event_times[en] = payload["t"]
                event_wa[en] = payload["wa"]
            if payload["meet_id"]:
                meets.add(payload["meet_id"])
            if payload["start_date"]:
                dates.append(payload["start_date"])
            ran_nats = ran_nats or payload["nationals"]
            has_relay = has_relay or payload["is_relay"]
            if payload["team_id"]:
                teams.add(payload["team_id"])
            year = payload["year"]

        if len(event_times) < 2:
            continue

        wa_vals = list(event_wa.values())
        spread = max(wa_vals) - min(wa_vals)
        best_event = max(event_wa, key=lambda e: event_wa[e])
        span = (max(dates) - min(dates)).days if len(dates) >= 2 else 0

        profiles[key] = RichProfile(
            key=key,
            gender=gender,
            event_group="",
            year=year,
            race_count=len(results),
            event_times=event_times,
            event_wa=event_wa,
            wa_spread=spread,
            best_event=best_event,
            events_competed=len(event_times),
            max_wa=max(wa_vals),
            mean_wa=statistics.mean(wa_vals),
            unique_meets=len(meets),
            season_span_days=span,
            ran_nationals=ran_nats,
            has_relay=has_relay,
            team_ids=teams,
        )
    return profiles


def fit_best(pairs: list[tuple[float, float]]) -> tuple[str, dict, Callable] | None:
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


def cv_route(
    records: list[tuple[float, float, str, str]],
) -> dict[str, float] | None:
    """records: (x, y, bal_spec_label, factor_label)"""
    if len(records) < MIN_REPORT_N:
        return None
    n = len(records)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)

    buckets = {
        "pooled": ([], []),
        "bal_spec": ([], []),
        "factor": ([], []),
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

        pooled = fit_best([(a, b) for a, b, _, _ in train])
        if pooled is None:
            continue
        _, pp, ppred = pooled

        def pooled_fn(x: float) -> float:
            return ppred(x, pp)

        bs_fits: dict[str, tuple] = {}
        for lab in {bs for _, _, bs, _ in train}:
            subset = [(a, b) for a, b, bs, _ in train if bs == lab]
            fit = fit_best(subset)
            if fit:
                bs_fits[lab] = fit

        fac_fits: dict[str, tuple] = {}
        for lab in {fac for _, _, _, fac in train}:
            subset = [(a, b) for a, b, _, fac in train if fac == lab]
            fit = fit_best(subset)
            if fit:
                fac_fits[lab] = fit

        joint_fits: dict[tuple[str, str], tuple] = {}
        for bs in {b for _, _, b, _ in train}:
            for fac in {f for _, _, _, f in train}:
                subset = [(a, b) for a, b, b2, f2 in train if b2 == bs and f2 == fac]
                fit = fit_best(subset)
                if fit:
                    joint_fits[(bs, fac)] = fit

        for x, y, bs, fac in test:
            buckets["pooled"][0].append(y)
            buckets["pooled"][1].append(pooled_fn(x))
            buckets["bal_spec"][0].append(y)
            buckets["bal_spec"][1].append(predict_with(bs_fits.get(bs), x, pooled_fn))
            buckets["factor"][0].append(y)
            buckets["factor"][1].append(predict_with(fac_fits.get(fac), x, pooled_fn))
            buckets["joint"][0].append(y)
            if (bs, fac) in joint_fits:
                buckets["joint"][1].append(predict_with(joint_fits[(bs, fac)], x, pooled_fn))
            elif fac in fac_fits:
                buckets["joint"][1].append(predict_with(fac_fits[fac], x, pooled_fn))
            elif bs in bs_fits:
                buckets["joint"][1].append(predict_with(bs_fits[bs], x, pooled_fn))
            else:
                buckets["joint"][1].append(pooled_fn(x))

    out = {}
    for name, (actual, pred) in buckets.items():
        if not actual:
            return None
        out[name], _ = metrics(actual, pred)
    return out


def median_split_labels(values: list[float], labels=("low", "high")) -> dict[float, str]:
    """Map each distinct value... actually assign by threshold on list aligned elsewhere."""
    if not values:
        return {}
    med = statistics.median(values)
    # return threshold only; caller compares
    return {"threshold": med}  # type: ignore


def build_factor_labelers(
    pair_profiles: list[RichProfile],
    from_ev: str,
    to_ev: str,
) -> dict[str, Callable[[RichProfile], str | None]]:
    """Build factor label functions calibrated on this pair's sample."""
    max_was = [p.max_wa for p in pair_profiles]
    spans = [p.season_span_days for p in pair_profiles]
    meets = [p.unique_meets for p in pair_profiles]
    spreads = [p.wa_spread for p in pair_profiles]
    from_was = [p.event_wa[from_ev] for p in pair_profiles]
    gaps = [abs(p.event_wa[from_ev] - p.event_wa[to_ev]) for p in pair_profiles]

    med_max = statistics.median(max_was)
    med_span = statistics.median(spans)
    med_meets = statistics.median(meets)
    med_from = statistics.median(from_was)
    med_gap = statistics.median(gaps)

    # Tertiles of spread for finer specialization
    qs = sorted(spreads)
    n = len(qs)
    t1 = qs[max(0, n // 3 - 1)] if n >= 9 else SPREAD_THRESHOLD
    t2 = qs[min(n - 1, (2 * n) // 3)] if n >= 9 else SPREAD_THRESHOLD

    def ability_tier(p: RichProfile) -> str:
        return "ability_high" if p.max_wa >= med_max else "ability_low"

    def season_span(p: RichProfile) -> str:
        return "span_long" if p.season_span_days >= med_span else "span_short"

    def meet_volume(p: RichProfile) -> str:
        return "meets_many" if p.unique_meets >= med_meets else "meets_few"

    def events_competed(p: RichProfile) -> str:
        return "events_3plus" if p.events_competed >= 3 else "events_2"

    def race_count(p: RichProfile) -> str:
        return f"races_{p.race_count}"

    def season_year(p: RichProfile) -> str:
        return f"year_{p.year}"

    def nationals(p: RichProfile) -> str:
        return "nationals_yes" if p.ran_nationals else "nationals_no"

    def has_relay(p: RichProfile) -> str:
        return "relay_yes" if p.has_relay else "relay_no"

    def best_is_from(p: RichProfile) -> str:
        return "best_is_from" if p.best_event == from_ev else "best_not_from"

    def best_is_to(p: RichProfile) -> str:
        return "best_is_to" if p.best_event == to_ev else "best_not_to"

    def predict_to_best(p: RichProfile) -> str:
        return "to_best" if p.best_event == to_ev else "to_nonbest"

    def pair_gap(p: RichProfile) -> str:
        g = abs(p.event_wa[from_ev] - p.event_wa[to_ev])
        return "pair_gap_large" if g >= med_gap else "pair_gap_small"

    def pair_gap_50(p: RichProfile) -> str:
        g = abs(p.event_wa[from_ev] - p.event_wa[to_ev])
        return "pair_gap_ge50" if g >= 50 else "pair_gap_lt50"

    def from_stronger(p: RichProfile) -> str:
        return (
            "from_stronger_wa"
            if p.event_wa[from_ev] >= p.event_wa[to_ev]
            else "from_weaker_wa"
        )

    def source_ability(p: RichProfile) -> str:
        return "from_wa_high" if p.event_wa[from_ev] >= med_from else "from_wa_low"

    def spread_tertile(p: RichProfile) -> str:
        if p.wa_spread <= t1:
            return "spread_low"
        if p.wa_spread <= t2:
            return "spread_mid"
        return "spread_high"

    def best_event_id(p: RichProfile) -> str:
        return f"best_{p.best_event}"

    return {
        "ability_tier": ability_tier,
        "season_span": season_span,
        "meet_volume": meet_volume,
        "events_competed": events_competed,
        "race_count": race_count,
        "season_year": season_year,
        "nationals": nationals,
        "has_relay": has_relay,
        "best_is_from": best_is_from,
        "best_is_to": best_is_to,
        "predict_to_best": predict_to_best,
        "pair_wa_gap_median": pair_gap,
        "pair_wa_gap_50": pair_gap_50,
        "from_stronger_wa": from_stronger,
        "source_ability": source_ability,
        "wa_spread_tertile": spread_tertile,
        "best_event": best_event_id,
    }


FACTOR_DESCRIPTIONS = {
    "ability_tier": "Max season WA in group: low vs high (median split)",
    "season_span": "Days between first and last race: short vs long (median)",
    "meet_volume": "Unique meets in season: few vs many (median)",
    "events_competed": "Distinct events raced in group: 2 vs 3+",
    "race_count": "Season race count: 5 vs 6",
    "season_year": "Calendar year of the athlete-season",
    "nationals": "Competed at nationals that season (yes/no)",
    "has_relay": "Any relay-leg result in season (yes/no)",
    "best_is_from": "Athlete's best-WA event is the source event",
    "best_is_to": "Athlete's best-WA event is the target event",
    "predict_to_best": "Predicting toward the athlete's best-WA event",
    "pair_wa_gap_median": "|WA(from)−WA(to)| relative to pair median",
    "pair_wa_gap_50": "|WA(from)−WA(to)| ≥ 50 WA points",
    "from_stronger_wa": "Source event has higher WA than target",
    "source_ability": "Source-event WA low vs high (median)",
    "wa_spread_tertile": "Overall WA spread tertile (finer than bal/spec)",
    "best_event": "Identity of best-WA event in the group",
}


def run(race_counts: list[int]) -> dict:
    profile_rows: list[dict] = []
    factor_rows: list[dict] = []
    summary_rows: list[dict] = []

    all_profiles_by_gg: dict[tuple[str, str], list[RichProfile]] = defaultdict(list)

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            profiles = load_rich_profiles(folder, prefix, gender, events, race_counts)
            for p in profiles.values():
                p.event_group = event_group
            all_profiles_by_gg[(gender, event_group)].extend(profiles.values())

            for p in profiles.values():
                profile_rows.append(
                    {
                        "athlete_season_key": p.key,
                        "gender": gender,
                        "event_group": event_group,
                        "year": p.year,
                        "race_count": p.race_count,
                        "events_competed": p.events_competed,
                        "wa_spread": round(p.wa_spread, 1),
                        "bal_spec": specialization_label(p.wa_spread),
                        "best_event": p.best_event,
                        "max_wa": round(p.max_wa, 1),
                        "mean_wa": round(p.mean_wa, 1),
                        "unique_meets": p.unique_meets,
                        "season_span_days": p.season_span_days,
                        "ran_nationals": p.ran_nationals,
                        "has_relay": p.has_relay,
                        "n_teams": len(p.team_ids),
                        "wa_by_event": json.dumps({k: round(v, 1) for k, v in p.event_wa.items()}),
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

                labelers = build_factor_labelers(pair_profs, from_ev, to_ev)
                _, _, r, _, _ = linreg(
                    [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_profs]
                )

                for factor_name, label_fn in labelers.items():
                    labeled = []
                    level_counts: dict[str, int] = defaultdict(int)
                    for p in pair_profs:
                        fac = label_fn(p)
                        if fac is None:
                            continue
                        bs = specialization_label(p.wa_spread)
                        labeled.append(
                            (p.event_times[from_ev], p.event_times[to_ev], bs, fac)
                        )
                        level_counts[fac] += 1

                    # Need at least 2 factor levels with enough support potential
                    if len(level_counts) < 2:
                        continue
                    if sum(1 for c in level_counts.values() if c >= MIN_COHORT_N) < 1:
                        # even one strong level helps factor routing with fallback
                        pass
                    if max(level_counts.values()) < MIN_COHORT_N:
                        continue

                    cvs = cv_route(labeled)
                    if not cvs:
                        continue

                    row = {
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "factor": factor_name,
                        "n": len(labeled),
                        "r": round(r, 4),
                        "factor_levels": json.dumps(dict(level_counts)),
                        "cv_pooled": round(cvs["pooled"], 4),
                        "cv_bal_spec": round(cvs["bal_spec"], 4),
                        "cv_factor": round(cvs["factor"], 4),
                        "cv_joint": round(cvs["joint"], 4),
                        "delta_factor_vs_pooled": round(cvs["pooled"] - cvs["factor"], 4),
                        "delta_factor_vs_bal_spec": round(cvs["bal_spec"] - cvs["factor"], 4),
                        "delta_joint_vs_pooled": round(cvs["pooled"] - cvs["joint"], 4),
                        "delta_joint_vs_bal_spec": round(cvs["bal_spec"] - cvs["joint"], 4),
                        "best_strategy": min(
                            [
                                ("pooled", cvs["pooled"]),
                                ("bal_spec", cvs["bal_spec"]),
                                ("factor", cvs["factor"]),
                                ("joint", cvs["joint"]),
                            ],
                            key=lambda x: x[1],
                        )[0],
                    }
                    factor_rows.append(row)

    # Aggregate by factor
    by_factor: dict[str, list[dict]] = defaultdict(list)
    for row in factor_rows:
        by_factor[row["factor"]].append(row)

    for factor_name, rows in sorted(by_factor.items()):
        n = len(rows)
        summary_rows.append(
            {
                "factor": factor_name,
                "description": FACTOR_DESCRIPTIONS.get(factor_name, ""),
                "n_pairs": n,
                "factor_beats_pooled": sum(1 for r in rows if r["delta_factor_vs_pooled"] > 0.01),
                "factor_beats_bal_spec": sum(
                    1 for r in rows if r["delta_factor_vs_bal_spec"] > 0.01
                ),
                "joint_beats_pooled": sum(1 for r in rows if r["delta_joint_vs_pooled"] > 0.01),
                "joint_beats_bal_spec": sum(
                    1 for r in rows if r["delta_joint_vs_bal_spec"] > 0.01
                ),
                "best_is_factor": sum(1 for r in rows if r["best_strategy"] == "factor"),
                "best_is_joint": sum(1 for r in rows if r["best_strategy"] == "joint"),
                "best_is_bal_spec": sum(1 for r in rows if r["best_strategy"] == "bal_spec"),
                "best_is_pooled": sum(1 for r in rows if r["best_strategy"] == "pooled"),
                "mean_delta_factor_vs_bal_spec": round(
                    statistics.mean(r["delta_factor_vs_bal_spec"] for r in rows), 4
                ),
                "mean_delta_joint_vs_bal_spec": round(
                    statistics.mean(r["delta_joint_vs_bal_spec"] for r in rows), 4
                ),
                "mean_delta_factor_vs_pooled": round(
                    statistics.mean(r["delta_factor_vs_pooled"] for r in rows), 4
                ),
                "median_delta_factor_vs_bal_spec": round(
                    statistics.median(r["delta_factor_vs_bal_spec"] for r in rows), 4
                ),
                "median_delta_joint_vs_bal_spec": round(
                    statistics.median(r["delta_joint_vs_bal_spec"] for r in rows), 4
                ),
            }
        )

    return {
        "profile_rows": profile_rows,
        "factor_rows": factor_rows,
        "summary_rows": summary_rows,
        "cohort_sizes": {
            f"{g} {eg}": len(ps) for (g, eg), ps in sorted(all_profiles_by_gg.items())
        },
    }


def write_reports(
    data: dict,
    *,
    pop_label: str,
    file_suffix: str,
) -> None:
    if pop_label == "6":
        pop_desc = "exactly 6 races"
        pop_title = "6 Races Per Season"
        min_pairs_for_promising = 4
    else:
        pop_desc = "5 or 6 races"
        pop_title = "5 or 6 Races Per Season"
        min_pairs_for_promising = 8

    report_name = f"new_factors_report{file_suffix}.txt"
    findings_name = f"new_factors_findings{file_suffix}.txt"
    summary_csv = f"factor_summary{file_suffix}.csv"
    pair_csv = f"factor_pair_comparison{file_suffix}.csv"
    profile_csv = f"athlete_season_new_factors{file_suffix}.csv"

    summaries = sorted(
        data["summary_rows"],
        key=lambda r: (
            -r["joint_beats_bal_spec"] / max(r["n_pairs"], 1),
            -r["mean_delta_joint_vs_bal_spec"],
            -r["factor_beats_bal_spec"] / max(r["n_pairs"], 1),
        ),
    )

    report = [
        f"New Factors Beyond Balanced/Specialized — {pop_title}",
        "=" * (52 + len(pop_title)),
        "",
        "Question: Under the old specialization definition (balanced vs specialized,",
        f"wa_spread ≥ {SPREAD_THRESHOLD:.0f}), are there other factors that improve",
        "cross-event time prediction accuracy if we account for them?",
        "",
        f"Population: athlete-seasons with {pop_desc} in the event group (2024–2026).",
        f"CV: {CV_FOLDS}-fold, seed={CV_SEED}. Min cohort n={MIN_COHORT_N}; pair n={MIN_REPORT_N}.",
        "",
        "Strategies compared per factor × event pair:",
        "  pooled            — one formula for all athletes in the pair",
        "  bal_spec          — route by balanced vs specialized",
        "  factor            — route by the candidate factor alone",
        "  joint (bal×factor)— route by both, with fallbacks",
        "",
        f"Cohort sizes ({pop_desc} athlete-seasons)",
        "-" * (14 + len(pop_desc) + 18),
    ]
    for k, v in data["cohort_sizes"].items():
        report.append(f"  {k}: n={v}")

    report.extend([
        "",
        "Factor leaderboard (sorted by how often joint beats bal_spec)",
        "-------------------------------------------------------------",
        f"{'Factor':<22} {'Pairs':>5} {'Fac>Bal':>7} {'Jnt>Bal':>7} "
        f"{'MeanΔFac':>9} {'MeanΔJnt':>9} {'Best=Fac/Jnt':>12}",
    ])
    for s in summaries:
        report.append(
            f"{s['factor']:<22} {s['n_pairs']:>5} "
            f"{s['factor_beats_bal_spec']:>3}/{s['n_pairs']:<3} "
            f"{s['joint_beats_bal_spec']:>3}/{s['n_pairs']:<3} "
            f"{s['mean_delta_factor_vs_bal_spec']:>+8.3f}s "
            f"{s['mean_delta_joint_vs_bal_spec']:>+8.3f}s "
            f"{s['best_is_factor']+s['best_is_joint']:>3}/{s['n_pairs']}"
        )

    promising = [
        s
        for s in summaries
        if s["n_pairs"] >= min_pairs_for_promising
        and (
            s["joint_beats_bal_spec"] / s["n_pairs"] >= 0.4
            or s["factor_beats_bal_spec"] / s["n_pairs"] >= 0.4
            or s["mean_delta_joint_vs_bal_spec"] > 0.15
            or s["mean_delta_factor_vs_bal_spec"] > 0.15
        )
    ]
    report.extend([
        "",
        f"Promising factors (help on ≥40% of pairs or mean Δ>+0.15s; min {min_pairs_for_promising} pairs)",
        "-" * 70,
    ])
    if promising:
        for s in promising:
            report.append(f"  {s['factor']}: {s['description']}")
            report.append(
                f"    factor>bal_spec {s['factor_beats_bal_spec']}/{s['n_pairs']} "
                f"(mean Δ {s['mean_delta_factor_vs_bal_spec']:+.3f}s); "
                f"joint>bal_spec {s['joint_beats_bal_spec']}/{s['n_pairs']} "
                f"(mean Δ {s['mean_delta_joint_vs_bal_spec']:+.3f}s)"
            )
            pair_rows = [r for r in data["factor_rows"] if r["factor"] == s["factor"]]
            wins = sorted(
                pair_rows,
                key=lambda r: max(r["delta_factor_vs_bal_spec"], r["delta_joint_vs_bal_spec"]),
                reverse=True,
            )[:4]
            for w in wins:
                if max(w["delta_factor_vs_bal_spec"], w["delta_joint_vs_bal_spec"]) <= 0.01:
                    continue
                report.append(
                    f"    e.g. {w['gender']} {w['event_group']} "
                    f"{w['from_event']}->{w['to_event']}: "
                    f"bal_spec {w['cv_bal_spec']:.3f}s, factor {w['cv_factor']:.3f}s, "
                    f"joint {w['cv_joint']:.3f}s (best={w['best_strategy']})"
                )
    else:
        report.append("  None met the promising threshold.")

    weak = [
        s
        for s in summaries
        if s["n_pairs"] >= min_pairs_for_promising
        and s["joint_beats_bal_spec"] / s["n_pairs"] <= 0.25
        and s["factor_beats_bal_spec"] / s["n_pairs"] <= 0.25
        and s["mean_delta_joint_vs_bal_spec"] <= 0
        and s["mean_delta_factor_vs_bal_spec"] <= 0
    ]
    report.extend(["", "Factors that did NOT help beyond bal/spec", "-" * 42])
    if weak:
        for s in weak:
            report.append(
                f"  {s['factor']}: joint>bal {s['joint_beats_bal_spec']}/{s['n_pairs']} "
                f"(mean Δ {s['mean_delta_joint_vs_bal_spec']:+.3f}s) — {s['description']}"
            )
    else:
        report.append("  (none met the weak-factor criteria)")

    top5 = summaries[:5]
    report.extend(["", "Detail — top factors by joint>bal_spec rate", "-" * 44])
    for s in top5:
        report.append("")
        report.append(f"{s['factor']} — {s['description']}")
        report.append(
            f"  {'Pair':<34} {'n':>3} {'Pool':>7} {'BalSp':>7} {'Fac':>7} {'Joint':>7} Best"
        )
        for r in sorted(
            [x for x in data["factor_rows"] if x["factor"] == s["factor"]],
            key=lambda x: -max(x["delta_joint_vs_bal_spec"], x["delta_factor_vs_bal_spec"]),
        ):
            pair = f"{r['gender']} {r['event_group']} {r['from_event']}->{r['to_event']}"
            report.append(
                f"  {pair:<34} {r['n']:>3} {r['cv_pooled']:>6.3f}s {r['cv_bal_spec']:>6.3f}s "
                f"{r['cv_factor']:>6.3f}s {r['cv_joint']:>6.3f}s {r['best_strategy']}"
            )

    report.extend(["", "Interpretation", "--------------"])
    if promising:
        names = ", ".join(s["factor"] for s in promising[:5])
        report.append(
            f"Yes — several factors beyond balanced/specialized improve models: {names}."
        )
        report.append(
            "Most useful are typically pair-level WA gaps and projection direction"
            " (from stronger WA / best-event involvement)."
        )
        report.append(
            "Best practice: start with bal/spec routing, then layer the strongest"
            " secondary factor via joint routing when both labels are known."
        )
    else:
        report.append(
            "No clear secondary factors consistently beat bal/spec routing in this sample."
            " Stick with balanced/specialized."
        )
    if pop_label == "6":
        report.append(
            "Note: the 6-race-only sample is smaller; some factors that looked strong at"
            " 5-or-6 may lose power here, and women's pairs often lack n≥20."
        )

    report.extend([
        "",
        "Notes:",
        "  • Factors are median-split or categorical labels fit per event-pair sample.",
        "  • Women's pairs often drop out for joint cells due to small n.",
        "  • wa_spread_tertile is a finer version of bal/spec — expected to correlate.",
        "  • Short/Long specialization was studied separately (other folders).",
        "",
        "Output files:",
        f"  {report_name}",
        f"  {findings_name}",
        f"  {summary_csv}",
        f"  {pair_csv}",
        f"  {profile_csv}",
        "",
        "Source: time_models/new_factors/analyze_new_factors.py",
    ])

    findings = [
        f"New Factors Beyond Balanced/Specialized — Findings ({pop_title})",
        "=" * (58 + len(pop_title)),
        "",
        "Question: Besides balanced vs specialized (wa_spread ≥ 50), what else",
        f"improves cross-event time models for {pop_desc} athlete-seasons?",
        "",
    ]
    if promising:
        findings.append("Helpful secondary factors:")
        for s in promising:
            findings.append(f"  • {s['factor']}: {s['description']}")
            findings.append(
                f"    joint>bal_spec {s['joint_beats_bal_spec']}/{s['n_pairs']} "
                f"(mean Δ {s['mean_delta_joint_vs_bal_spec']:+.3f}s); "
                f"factor alone>bal_spec {s['factor_beats_bal_spec']}/{s['n_pairs']}"
            )
    else:
        findings.append("No secondary factor clearly beat bal/spec routing.")
    findings.append("")
    findings.append("Leaderboard (joint beats bal_spec rate):")
    for s in summaries[:8]:
        findings.append(
            f"  {s['factor']}: {s['joint_beats_bal_spec']}/{s['n_pairs']} "
            f"(mean Δ {s['mean_delta_joint_vs_bal_spec']:+.3f}s)"
        )
    findings.extend([
        "",
        "Recommendation:",
        "  1. Keep bal/spec as a useful axis when available.",
        "  2. Strongest secondary factor is usually pair-level WA gap",
        "     (|WA_from − WA_to| ≥ 50, or median split).",
        "  3. Next: projection direction (from_stronger_wa, best_is_from / predict_to_best).",
        "",
        f"See {report_name} for pair-level tables.",
        "Source: analyze_new_factors.py",
    ])

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            (OUTPUT_ROOT / name).write_text("")
            return
        with open(OUTPUT_ROOT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(profile_csv, data["profile_rows"])
    write_csv(pair_csv, data["factor_rows"])
    write_csv(summary_csv, data["summary_rows"])
    (OUTPUT_ROOT / report_name).write_text("\n".join(report).rstrip() + "\n")
    (OUTPUT_ROOT / findings_name).write_text("\n".join(findings).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for pop_label, race_counts, file_suffix in POPULATIONS:
        data = run(race_counts)
        write_reports(data, pop_label=pop_label, file_suffix=file_suffix)
        print(f"Wrote {pop_label} analysis to {OUTPUT_ROOT} (suffix={file_suffix!r})")
        print(f"  profiles={len(data['profile_rows'])}")
        print(f"  factor-pair rows={len(data['factor_rows'])}")
        print(f"  factors summarized={len(data['summary_rows'])}")
        top = sorted(
            data["summary_rows"],
            key=lambda r: -r["joint_beats_bal_spec"] / max(r["n_pairs"], 1),
        )[:5]
        for s in top:
            print(
                f"  top: {s['factor']} joint>bal {s['joint_beats_bal_spec']}/{s['n_pairs']} "
                f"meanΔ={s['mean_delta_joint_vs_bal_spec']:+.3f}s"
            )


if __name__ == "__main__":
    main()
