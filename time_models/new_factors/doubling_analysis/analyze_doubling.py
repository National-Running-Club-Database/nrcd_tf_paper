"""Doubling analysis: multi-event meets vs WA scores and time-model accuracy.

Doubling = an athlete records ≥2 distinct individual events from the event group
in the same meet (same meet_id). Example: 100m + 200m at one meet.

Questions:
  1) Is doubling associated with higher World Athletics scores?
  2) Does knowing whether an athlete doubled improve cross-event time models?

Population (primary): athlete-seasons with 5 or 6 races (aligned with new_factors).
Appendix: all athlete-seasons with ≥2 events for descriptive WA stats.

Sprints & Distance only. Relays / steeple excluded. Outdoor 2024–2026, March 1+.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[2]
NEW_FACTORS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
SPEC_ROOT = TIME_MODELS_ROOT / "specialized_time_models"

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(NEW_FACTORS_ROOT))

from relay_rq1_data import (  # noqa: E402
    STANDARD_RELAY_IDS,
    is_in_season,
    points_col,
    season_rows,
)
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
from analyze_new_factors import cv_route  # noqa: E402

MIN_COHORT_N = 12
MIN_REPORT_N = 20
SPREAD_THRESHOLD = 50.0
STEEPLE_ID = 20
EPS = 0.01

PRIMARY_RACE_COUNTS = [5, 6]

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
class MeetBundle:
    meet_id: str
    results: list[dict] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len({r["event_name"] for r in self.results})

    @property
    def is_double(self) -> bool:
        return self.n_events >= 2

    @property
    def mean_wa(self) -> float:
        return statistics.mean(r["wa"] for r in self.results)

    @property
    def max_wa(self) -> float:
        return max(r["wa"] for r in self.results)


@dataclass
class DoublingProfile:
    key: str
    gender: str
    event_group: str
    year: str
    race_count: int
    event_times: dict[str, float]
    event_wa: dict[str, float]
    wa_spread: float
    best_event: str
    max_wa: float
    mean_wa: float
    meets: list[MeetBundle]
    n_meets: int
    n_double_meets: int
    ever_doubled: bool
    double_meet_share: float
    # WA of results recorded at double meets vs solo meets
    mean_wa_at_doubles: float | None
    mean_wa_at_solos: float | None
    n_results_at_doubles: int
    n_results_at_solos: int
    # Was each event's season PB set at a double meet?
    pb_from_double: dict[str, bool]


def parse_date(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def load_doubling_profiles(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
    race_counts: list[int] | None,
) -> dict[str, DoublingProfile]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    allowed = set(event_name_to_id.values())
    pcol = points_col(gender)
    season_results: dict[str, dict[str, dict]] = defaultdict(dict)

    for year in ("2024", "2025", "2026"):
        for row in season_rows(folder, prefix, gender, year):
            date_str = (row.get("start_date") or "").strip()
            if date_str and not is_in_season(date_str):
                continue
            try:
                event_id = int(row["running_event_id"])
            except (TypeError, ValueError):
                continue
            if event_id in STANDARD_RELAY_IDS or event_id == STEEPLE_ID:
                continue
            if event_id not in allowed:
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
            meet_id = (row.get("meet_id") or "").strip()
            if not meet_id:
                continue
            key = f"{aid}|{year}"
            season_results[key][result_id] = {
                "event_name": id_to_name[event_id],
                "t": t,
                "wa": wa,
                "meet_id": meet_id,
                "start_date": parse_date(date_str),
                "year": year,
            }

    allowed_rc = set(race_counts) if race_counts is not None else None
    out: dict[str, DoublingProfile] = {}
    for key, results in season_results.items():
        if allowed_rc is not None and len(results) not in allowed_rc:
            continue

        by_meet: dict[str, MeetBundle] = {}
        event_times: dict[str, float] = {}
        event_wa: dict[str, float] = {}
        pb_meet: dict[str, str] = {}

        for payload in results.values():
            mid = payload["meet_id"]
            if mid not in by_meet:
                by_meet[mid] = MeetBundle(meet_id=mid)
            by_meet[mid].results.append(payload)
            en = payload["event_name"]
            if en not in event_times or payload["t"] < event_times[en]:
                event_times[en] = payload["t"]
                event_wa[en] = payload["wa"]
                pb_meet[en] = mid

        if len(event_times) < 2:
            continue

        meets = list(by_meet.values())
        n_meets = len(meets)
        n_double = sum(1 for m in meets if m.is_double)
        double_meet_ids = {m.meet_id for m in meets if m.is_double}

        wa_at_double = [r["wa"] for m in meets if m.is_double for r in m.results]
        wa_at_solo = [r["wa"] for m in meets if not m.is_double for r in m.results]

        wa_vals = list(event_wa.values())
        out[key] = DoublingProfile(
            key=key,
            gender=gender,
            event_group="",
            year=key.split("|")[1],
            race_count=len(results),
            event_times=event_times,
            event_wa=event_wa,
            wa_spread=max(wa_vals) - min(wa_vals),
            best_event=max(event_wa, key=lambda e: event_wa[e]),
            max_wa=max(wa_vals),
            mean_wa=statistics.mean(wa_vals),
            meets=meets,
            n_meets=n_meets,
            n_double_meets=n_double,
            ever_doubled=n_double > 0,
            double_meet_share=(n_double / n_meets) if n_meets else 0.0,
            mean_wa_at_doubles=(
                statistics.mean(wa_at_double) if wa_at_double else None
            ),
            mean_wa_at_solos=statistics.mean(wa_at_solo) if wa_at_solo else None,
            n_results_at_doubles=len(wa_at_double),
            n_results_at_solos=len(wa_at_solo),
            pb_from_double={
                e: (pb_meet.get(e) in double_meet_ids) for e in event_times
            },
        )
    return out


def mean_or_none(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def median_or_none(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def summarize_wa(
    profiles: list[DoublingProfile],
    population: str,
) -> list[dict]:
    """Cross-sectional + within-athlete WA comparisons."""
    rows = []
    groups = defaultdict(list)
    for p in profiles:
        groups[(p.gender, p.event_group)].append(p)
    groups[("All", "All")] = profiles

    for (gender, eg), ps in sorted(groups.items()):
        doubled = [p for p in ps if p.ever_doubled]
        never = [p for p in ps if not p.ever_doubled]
        # Within athletes who did both double and solo meets
        both = [
            p
            for p in doubled
            if p.mean_wa_at_doubles is not None and p.mean_wa_at_solos is not None
        ]
        within_deltas = [
            p.mean_wa_at_doubles - p.mean_wa_at_solos  # type: ignore[operator]
            for p in both
        ]
        pb_double_flags = [
            flag for p in ps for flag in p.pb_from_double.values()
        ]
        mean_max_d = mean_or_none([p.max_wa for p in doubled])
        mean_max_n = mean_or_none([p.max_wa for p in never])
        mean_mean_d = mean_or_none([p.mean_wa for p in doubled])
        mean_mean_n = mean_or_none([p.mean_wa for p in never])
        rows.append(
            {
                "population": population,
                "gender": gender,
                "event_group": eg,
                "n_athlete_seasons": len(ps),
                "n_ever_doubled": len(doubled),
                "n_never_doubled": len(never),
                "pct_ever_doubled": round(100 * len(doubled) / len(ps), 1) if ps else None,
                "mean_max_wa_doubled": round(mean_max_d, 2) if mean_max_d is not None else None,
                "mean_max_wa_never": round(mean_max_n, 2) if mean_max_n is not None else None,
                "delta_mean_max_wa": (
                    round(mean_max_d - mean_max_n, 2)
                    if mean_max_d is not None and mean_max_n is not None
                    else None
                ),
                "mean_mean_wa_doubled": round(mean_mean_d, 2) if mean_mean_d is not None else None,
                "mean_mean_wa_never": round(mean_mean_n, 2) if mean_mean_n is not None else None,
                "n_within_both_meet_types": len(both),
                "mean_within_delta_wa_double_minus_solo": round(
                    mean_or_none(within_deltas), 2
                )
                if within_deltas
                else None,
                "median_within_delta_wa_double_minus_solo": round(
                    median_or_none(within_deltas), 2
                )
                if within_deltas
                else None,
                "pct_within_delta_positive": (
                    round(100 * sum(1 for d in within_deltas if d > 0) / len(within_deltas), 1)
                    if within_deltas
                    else None
                ),
                "pct_season_pbs_from_double_meet": (
                    round(100 * sum(pb_double_flags) / len(pb_double_flags), 1)
                    if pb_double_flags
                    else None
                ),
                "mean_double_meets_among_doublers": round(
                    mean_or_none([p.n_double_meets for p in doubled]), 2
                )
                if doubled
                else None,
                "mean_double_share_among_doublers": round(
                    mean_or_none([p.double_meet_share for p in doubled]), 3
                )
                if doubled
                else None,
            }
        )
    return rows


def doubling_labelers(
    pair_profiles: list[DoublingProfile],
) -> dict[str, Callable[[DoublingProfile], str]]:
    shares = [p.double_meet_share for p in pair_profiles]
    n_dubs = [p.n_double_meets for p in pair_profiles]
    med_share = statistics.median(shares) if shares else 0.0
    med_n = statistics.median(n_dubs) if n_dubs else 0.0

    return {
        "ever_doubled": lambda p: "doubled_yes" if p.ever_doubled else "doubled_no",
        "double_meet_share": lambda p: (
            "double_share_high" if p.double_meet_share >= med_share else "double_share_low"
        ),
        "n_double_meets": lambda p: (
            "n_doubles_high" if p.n_double_meets >= med_n else "n_doubles_low"
        ),
        "pair_pb_from_double": lambda p: (
            # At least one of the two events' season PB came from a double meet
            "pb_from_double"
            if any(p.pb_from_double.get(e, False) for e in p.event_times)
            else "pb_from_solo_only"
        ),
    }


FACTOR_DESCRIPTIONS = {
    "ever_doubled": "≥1 meet with 2+ distinct group events (yes/no)",
    "double_meet_share": "Share of season meets that were doubles (median split)",
    "n_double_meets": "Count of double meets in season (median split)",
    "pair_pb_from_double": "Any season PB set at a double meet (yes/no)",
}


def run_time_model_factors(
    profiles_by_gg: dict[tuple[str, str], list[DoublingProfile]],
    population: str,
) -> tuple[list[dict], list[dict]]:
    pair_rows: list[dict] = []
    for (gender, event_group), profiles in profiles_by_gg.items():
        order = EVENT_ORDER[event_group]
        for from_ev, to_ev in permutations(order, 2):
            pair_ps = [
                p
                for p in profiles
                if from_ev in p.event_times and to_ev in p.event_times
            ]
            if len(pair_ps) < MIN_REPORT_N:
                continue
            labelers = doubling_labelers(pair_ps)
            _, _, r, _, _ = linreg(
                [(p.event_times[from_ev], p.event_times[to_ev]) for p in pair_ps]
            )
            for factor_name, label_fn in labelers.items():
                labeled = []
                level_counts: dict[str, int] = defaultdict(int)
                for p in pair_ps:
                    fac = label_fn(p)
                    bs = specialization_label(p.wa_spread, SPREAD_THRESHOLD)
                    labeled.append(
                        (p.event_times[from_ev], p.event_times[to_ev], bs, fac)
                    )
                    level_counts[fac] += 1
                if len(level_counts) < 2:
                    continue
                if max(level_counts.values()) < MIN_COHORT_N:
                    continue
                cvs = cv_route(labeled)
                if not cvs:
                    continue
                best = min(
                    ("pooled", "bal_spec", "factor", "joint"),
                    key=lambda k: cvs[k],
                )
                pair_rows.append(
                    {
                        "population": population,
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
                        "delta_factor_vs_pooled": round(
                            cvs["pooled"] - cvs["factor"], 4
                        ),
                        "delta_factor_vs_bal_spec": round(
                            cvs["bal_spec"] - cvs["factor"], 4
                        ),
                        "delta_joint_vs_pooled": round(
                            cvs["pooled"] - cvs["joint"], 4
                        ),
                        "delta_joint_vs_bal_spec": round(
                            cvs["bal_spec"] - cvs["joint"], 4
                        ),
                        "best_strategy": best,
                        "factor_beats_pooled": cvs["factor"] < cvs["pooled"] - EPS,
                        "joint_beats_pooled": cvs["joint"] < cvs["pooled"] - EPS,
                        "factor_beats_bal_spec": cvs["factor"] < cvs["bal_spec"] - EPS,
                        "joint_beats_bal_spec": cvs["joint"] < cvs["bal_spec"] - EPS,
                    }
                )

    # Factor summaries
    by_factor: dict[str, list[dict]] = defaultdict(list)
    for r in pair_rows:
        by_factor[r["factor"]].append(r)
    summaries = []
    for fname, rows in by_factor.items():
        summaries.append(
            {
                "population": population,
                "factor": fname,
                "description": FACTOR_DESCRIPTIONS.get(fname, ""),
                "n_pairs": len(rows),
                "factor_beats_pooled": sum(1 for r in rows if r["factor_beats_pooled"]),
                "joint_beats_pooled": sum(1 for r in rows if r["joint_beats_pooled"]),
                "factor_beats_bal_spec": sum(
                    1 for r in rows if r["factor_beats_bal_spec"]
                ),
                "joint_beats_bal_spec": sum(
                    1 for r in rows if r["joint_beats_bal_spec"]
                ),
                "mean_delta_factor_vs_pooled": round(
                    statistics.mean(r["delta_factor_vs_pooled"] for r in rows), 4
                ),
                "mean_delta_joint_vs_pooled": round(
                    statistics.mean(r["delta_joint_vs_pooled"] for r in rows), 4
                ),
                "mean_delta_factor_vs_bal_spec": round(
                    statistics.mean(r["delta_factor_vs_bal_spec"] for r in rows), 4
                ),
                "mean_delta_joint_vs_bal_spec": round(
                    statistics.mean(r["delta_joint_vs_bal_spec"] for r in rows), 4
                ),
                "best_is_factor": sum(1 for r in rows if r["best_strategy"] == "factor"),
                "best_is_joint": sum(1 for r in rows if r["best_strategy"] == "joint"),
                "best_is_bal_spec": sum(
                    1 for r in rows if r["best_strategy"] == "bal_spec"
                ),
                "best_is_pooled": sum(1 for r in rows if r["best_strategy"] == "pooled"),
            }
        )
    summaries.sort(
        key=lambda r: (
            -r["factor_beats_pooled"],
            -r["mean_delta_factor_vs_pooled"],
        )
    )
    return pair_rows, summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_reports(
    wa_rows: list[dict],
    pair_rows: list[dict],
    factor_sums: list[dict],
    cohort_sizes: list[dict],
) -> None:
    primary_wa = [r for r in wa_rows if r["population"] == "5_or_6" and r["gender"] == "All"]
    all_wa = [r for r in wa_rows if r["population"] == "all" and r["gender"] == "All"]
    primary_factors = [r for r in factor_sums if r["population"] == "5_or_6"]

    lines = [
        "Doubling Analysis — Report",
        "==========================",
        "",
        "Questions:",
        "  1) Does doubling (2+ events in one meet) associate with higher WA scores?",
        "  2) Does knowing doubling status make cross-event time models more accurate?",
        "",
        "Definition:",
        "  A meet is a double if the athlete has ≥2 distinct individual events from",
        "  the event group (Sprints: 100/200/400; Distance: 800/1500/5000) with the",
        "  same meet_id. Relays and 3000m steeplechase excluded.",
        "",
        "Method:",
        "  • Outdoor 2024–2026, results on/after March 1.",
        "  • Primary population: 5 or 6 races per athlete-season (new_factors aligned).",
        "  • WA appendix: all athlete-seasons with ≥2 group events.",
        "  • Time-model test: 5-fold CV routing (pooled / bal_spec / doubling factor /",
        f"    joint); pair n≥{MIN_REPORT_N}; cohort fit n≥{MIN_COHORT_N}.",
        f"  • “Beats” = CV median |error| improves by >{EPS}s.",
        "",
        "Cohort sizes",
        "------------",
    ]
    for c in cohort_sizes:
        lines.append(
            f"  {c['population']:<10} {c['gender']:<6} {c['event_group']:<10} "
            f"n={c['n']:<5} ever_doubled={c['n_ever_doubled']} "
            f"({c['pct_ever_doubled']}%)"
        )

    lines.extend(["", "1) Doubling and World Athletics scores", "-" * 38])
    for label, rows in (("5 or 6 races", primary_wa), ("All seasons (≥2 events)", all_wa)):
        if not rows:
            continue
        r = rows[0]
        lines.append(f"\n  Population: {label}")
        lines.append(
            f"    Ever doubled: {r['n_ever_doubled']}/{r['n_athlete_seasons']} "
            f"({r['pct_ever_doubled']}%)"
        )
        lines.append(
            f"    Mean season max WA — doubled: {r['mean_max_wa_doubled']}  "
            f"never: {r['mean_max_wa_never']}  "
            f"Δ (doubled−never): {r['delta_mean_max_wa']:+}"
            if r["delta_mean_max_wa"] is not None
            else "    (insufficient groups)"
        )
        lines.append(
            f"    Mean season mean WA — doubled: {r['mean_mean_wa_doubled']}  "
            f"never: {r['mean_mean_wa_never']}"
        )
        if r["n_within_both_meet_types"]:
            lines.append(
                f"    Within athletes with both double & solo meets "
                f"(n={r['n_within_both_meet_types']}):"
            )
            lines.append(
                f"      mean(WA at double meets − WA at solo meets) = "
                f"{r['mean_within_delta_wa_double_minus_solo']:+} "
                f"(median {r['median_within_delta_wa_double_minus_solo']:+}); "
                f"{r['pct_within_delta_positive']}% have higher mean WA at doubles"
            )
        lines.append(
            f"    Season PBs set at a double meet: "
            f"{r['pct_season_pbs_from_double_meet']}%"
        )

    # Gender × group brief for primary
    lines.append("\n  Primary (5_or_6) by gender × group — Δ mean max WA (doubled−never):")
    for r in wa_rows:
        if r["population"] != "5_or_6" or r["gender"] == "All":
            continue
        if r["delta_mean_max_wa"] is None:
            continue
        lines.append(
            f"    {r['gender']} {r['event_group']}: "
            f"Δ={r['delta_mean_max_wa']:+}  "
            f"(doubled n={r['n_ever_doubled']}, never n={r['n_never_doubled']})"
        )

    lines.extend(
        [
            "",
            "WA interpretation notes:",
            "  • Cross-sectional Δ may reflect selection (stronger / busier athletes",
            "    double more) rather than a causal effect of doubling.",
            "  • Within-athlete double-vs-solo WA compares the same athlete's results",
            "    at multi-event vs single-event meets — closer to a load tradeoff test.",
            "",
            "2) Doubling and time-model accuracy (5 or 6 races)",
            "-" * 50,
        ]
    )
    if primary_factors:
        lines.append(
            f"  {'Factor':<22} {'Beats pooled':>12} {'Mean Δ':>10} "
            f"{'Beats bal/spec':>14} {'Mean Δ':>10}"
        )
        for s in primary_factors:
            lines.append(
                f"  {s['factor']:<22} "
                f"{s['factor_beats_pooled']:>3}/{s['n_pairs']:<3} "
                f"{s['mean_delta_factor_vs_pooled']:>+9.3f}s "
                f"{s['factor_beats_bal_spec']:>4}/{s['n_pairs']:<3} "
                f"{s['mean_delta_factor_vs_bal_spec']:>+9.3f}s"
            )
            lines.append(f"    {s['description']}")

        top = primary_factors[0]
        helpful = [
            s
            for s in primary_factors
            if s["factor_beats_pooled"] / max(s["n_pairs"], 1) >= 0.5
            and s["mean_delta_factor_vs_pooled"] > EPS
        ]
        lines.extend(["", "  Pair-level bests (ever_doubled factor):"])
        for r in pair_rows:
            if r["population"] != "5_or_6" or r["factor"] != "ever_doubled":
                continue
            lines.append(
                f"    {r['gender']} {r['event_group']} {r['from_event']}->"
                f"{r['to_event']}: pooled={r['cv_pooled']:.3f}s "
                f"factor={r['cv_factor']:.3f}s "
                f"(Δ {r['delta_factor_vs_pooled']:+.3f}s) best={r['best_strategy']}"
            )

        lines.extend(["", "Time-model verdict", "-" * 17])
        if helpful:
            names = ", ".join(s["factor"] for s in helpful)
            lines.append(
                f"  YES for accuracy — {names} beat pooled on ≥50% of pairs "
                f"with mean CV improvement >{EPS}s."
            )
        elif primary_factors and primary_factors[0]["mean_delta_factor_vs_pooled"] > 0:
            lines.append(
                f"  MIXED — best factor {top['factor']} beats pooled "
                f"{top['factor_beats_pooled']}/{top['n_pairs']} "
                f"(mean Δ {top['mean_delta_factor_vs_pooled']:+.3f}s); "
                "not a clear majority win."
            )
        else:
            lines.append(
                "  NO — doubling labels do not consistently improve time-model "
                "accuracy vs the pooled model."
            )
    else:
        lines.append("  No pairs met reporting thresholds.")

    # Overall verdicts
    wa_r = primary_wa[0] if primary_wa else None
    lines.extend(["", "Overall findings", "-" * 16])
    if wa_r and wa_r["delta_mean_max_wa"] is not None:
        if wa_r["delta_mean_max_wa"] > 5:
            lines.append(
                f"  WA scores: Doublers have higher season max WA on average "
                f"(Δ {wa_r['delta_mean_max_wa']:+} vs never-doubled at 5/6 races)."
            )
        elif wa_r["delta_mean_max_wa"] < -5:
            lines.append(
                f"  WA scores: Doublers have lower season max WA on average "
                f"(Δ {wa_r['delta_mean_max_wa']:+})."
            )
        else:
            lines.append(
                f"  WA scores: Little cross-sectional difference in season max WA "
                f"(Δ {wa_r['delta_mean_max_wa']:+})."
            )
        if wa_r.get("mean_within_delta_wa_double_minus_solo") is not None:
            d = wa_r["mean_within_delta_wa_double_minus_solo"]
            if d > 5:
                lines.append(
                    f"  Within-athlete: results at double meets score higher WA "
                    f"than solo meets (mean Δ {d:+})."
                )
            elif d < -5:
                lines.append(
                    f"  Within-athlete: results at double meets score lower WA "
                    f"than solo meets (mean Δ {d:+}) — possible fatigue tradeoff."
                )
            else:
                lines.append(
                    f"  Within-athlete: double vs solo meet WA nearly similar "
                    f"(mean Δ {d:+})."
                )

    lines.extend(
        [
            "",
            "Source: new_factors/doubling_analysis/analyze_doubling.py",
        ]
    )
    (OUTPUT_ROOT / "doubling_report.txt").write_text("\n".join(lines).rstrip() + "\n")

    # Findings short
    findings = [
        "Doubling Analysis — Findings",
        "============================",
        "",
        "Doubling = ≥2 distinct group events in the same meet.",
        "Primary population: 5 or 6 races per athlete-season.",
        "",
    ]
    if wa_r:
        findings.append(
            f"WA (5_or_6): ever_doubled={wa_r['pct_ever_doubled']}%  "
            f"Δ mean max WA (doubled−never)={wa_r['delta_mean_max_wa']:+}  "
            f"within Δ (double−solo meets)="
            f"{wa_r['mean_within_delta_wa_double_minus_solo']}"
        )
    findings.append("")
    findings.append("Time models (factor vs pooled):")
    for s in primary_factors:
        findings.append(
            f"  {s['factor']}: beats {s['factor_beats_pooled']}/{s['n_pairs']} "
            f"(mean Δ {s['mean_delta_factor_vs_pooled']:+.3f}s)"
        )
    findings.append("")
    # Short verdicts
    if wa_r and wa_r["delta_mean_max_wa"] is not None:
        never_n = wa_r.get("n_never_doubled") or 0
        if never_n < 20:
            findings.append(
                "WA verdict (cross-sectional): Almost everyone at 5/6 races doubles "
                f"(never-doubled n={never_n}) — comparing doubled vs never is unreliable."
            )
        elif wa_r["delta_mean_max_wa"] > 5:
            findings.append(
                "WA verdict: Doubling is associated with higher season WA scores "
                "(cross-sectional); interpret cautiously (selection)."
            )
        elif (
            wa_r.get("mean_within_delta_wa_double_minus_solo") is not None
            and wa_r["mean_within_delta_wa_double_minus_solo"] < -5
        ):
            findings.append(
                "WA verdict: Within athletes, double-meet results tend to score "
                "lower WA than solo-meet results (possible fatigue)."
            )
        else:
            findings.append(
                "WA verdict: No large, consistent WA boost from doubling alone."
            )
        if (
            wa_r.get("mean_within_delta_wa_double_minus_solo") is not None
            and never_n < 20
        ):
            findings.append(
                f"WA verdict (within-athlete): Double-meet results average "
                f"{wa_r['mean_within_delta_wa_double_minus_solo']:+} WA vs solo "
                f"meets for the same athlete — suggests doubling does not raise "
                f"scores (more often a small drop)."
            )
    if primary_factors:
        helpful = [
            s
            for s in primary_factors
            if s["factor_beats_pooled"] / max(s["n_pairs"], 1) >= 0.5
            and s["mean_delta_factor_vs_pooled"] > EPS
        ]
        if helpful:
            findings.append(
                "Time-model verdict: YES — "
                + ", ".join(s["factor"] for s in helpful)
                + " improve accuracy vs pooled."
            )
        else:
            findings.append(
                "Time-model verdict: NO clear accuracy gain from doubling factors "
                "vs pooled (or only weak/mixed gains)."
            )
    findings.extend(
        [
            "",
            "See doubling_report.txt for detail.",
            "Source: analyze_doubling.py",
        ]
    )
    (OUTPUT_ROOT / "doubling_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )


def load_population(
    race_counts: list[int] | None,
) -> dict[tuple[str, str], list[DoublingProfile]]:
    out: dict[tuple[str, str], list[DoublingProfile]] = defaultdict(list)
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            profiles = load_doubling_profiles(
                folder, prefix, gender, events, race_counts
            )
            for p in profiles.values():
                p.event_group = event_group
                p.gender = gender
            out[(gender, event_group)].extend(profiles.values())
    return out


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    populations = {
        "5_or_6": PRIMARY_RACE_COUNTS,
        "all": None,
    }

    all_wa_rows: list[dict] = []
    all_pair_rows: list[dict] = []
    all_factor_sums: list[dict] = []
    cohort_sizes: list[dict] = []
    profile_rows: list[dict] = []

    for pop_label, race_counts in populations.items():
        print(f"Loading population {pop_label}...")
        by_gg = load_population(race_counts)
        flat: list[DoublingProfile] = []
        for (gender, eg), ps in by_gg.items():
            flat.extend(ps)
            n_dub = sum(1 for p in ps if p.ever_doubled)
            cohort_sizes.append(
                {
                    "population": pop_label,
                    "gender": gender,
                    "event_group": eg,
                    "n": len(ps),
                    "n_ever_doubled": n_dub,
                    "pct_ever_doubled": round(100 * n_dub / len(ps), 1) if ps else 0,
                }
            )
            print(f"  {gender} {eg}: {len(ps)} (doubled {n_dub})")

        for p in flat:
            profile_rows.append(
                {
                    "population": pop_label,
                    "athlete_season_key": p.key,
                    "gender": p.gender,
                    "event_group": p.event_group,
                    "year": p.year,
                    "race_count": p.race_count,
                    "n_meets": p.n_meets,
                    "n_double_meets": p.n_double_meets,
                    "ever_doubled": p.ever_doubled,
                    "double_meet_share": round(p.double_meet_share, 3),
                    "max_wa": round(p.max_wa, 1),
                    "mean_wa": round(p.mean_wa, 1),
                    "wa_spread": round(p.wa_spread, 1),
                    "mean_wa_at_doubles": (
                        round(p.mean_wa_at_doubles, 1)
                        if p.mean_wa_at_doubles is not None
                        else None
                    ),
                    "mean_wa_at_solos": (
                        round(p.mean_wa_at_solos, 1)
                        if p.mean_wa_at_solos is not None
                        else None
                    ),
                    "n_results_at_doubles": p.n_results_at_doubles,
                    "n_results_at_solos": p.n_results_at_solos,
                    "pct_pbs_from_double": round(
                        100
                        * sum(p.pb_from_double.values())
                        / max(len(p.pb_from_double), 1),
                        1,
                    ),
                }
            )

        all_wa_rows.extend(summarize_wa(flat, pop_label))

        if pop_label == "5_or_6":
            print("  Running time-model factor CV...")
            pairs, sums = run_time_model_factors(by_gg, pop_label)
            all_pair_rows.extend(pairs)
            all_factor_sums.extend(sums)
            for s in sums:
                print(
                    f"    {s['factor']}: beats pooled "
                    f"{s['factor_beats_pooled']}/{s['n_pairs']} "
                    f"Δ {s['mean_delta_factor_vs_pooled']:+.3f}s"
                )

    write_csv(OUTPUT_ROOT / "doubling_wa_summary.csv", all_wa_rows)
    write_csv(OUTPUT_ROOT / "doubling_time_model_pairs.csv", all_pair_rows)
    write_csv(OUTPUT_ROOT / "doubling_factor_summary.csv", all_factor_sums)
    write_csv(OUTPUT_ROOT / "doubling_cohort_sizes.csv", cohort_sizes)
    write_csv(OUTPUT_ROOT / "athlete_season_doubling_profiles.csv", profile_rows)
    write_reports(all_wa_rows, all_pair_rows, all_factor_sums, cohort_sizes)
    print(f"Wrote doubling analysis to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
