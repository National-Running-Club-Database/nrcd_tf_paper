"""Feature importance within WA point-band time models.

Question: For athletes in bands 750–950, 800–1000, and 850–1050 (width 200),
does routing by previously important features (balanced/specialized, best_event,
best_is_from, pair WA gap, from_stronger_wa, bal×best_event) beat the pooled
band time model?

Population & filters match Point_Bands_Time_Models:
  outdoor 2024–2026, on/after March 1, no relays, no steeple; Sprints & Distance.
  Band inclusion: ≥1 individual result WA in [lo, hi).
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[2]
BAND_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
SPEC_ROOT = TIME_MODELS_ROOT / "specialized_time_models"

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(BAND_ROOT))

from relay_rq1_data import (  # noqa: E402
    STANDARD_RELAY_IDS,
    is_in_season,
    points_col,
    season_rows,
)
from build_cross_event_time_models import GROUP_CONFIG, parse_performance  # noqa: E402
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
from analyze_point_bands_time_models import STEEPLE_ID, in_band  # noqa: E402

MIN_COHORT_N = 12
MIN_REPORT_N = 20
SPREAD_THRESHOLD = 50.0
EPS = 0.01  # improvement threshold (seconds)

TARGET_BANDS = (
    (750, 950),
    (800, 1000),
    (850, 1050),
)

# Features previously found helpful (plus bal/spec as the main prior axis)
FEATURE_NAMES = (
    "bal_spec",
    "best_event",
    "best_is_from",
    "best_is_to",
    "pair_wa_gap_50",
    "pair_wa_gap_median",
    "from_stronger_wa",
    "events_competed",
)

# Multi-label strategies evaluated together (one CV pass per pair)
COMBO_STRATEGIES = (
    "pooled",
    "bal_spec",
    "best_event",
    "best_is_from",
    "pair_wa_gap_50",
    "from_stronger_wa",
    "bal_x_best_event",
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


@dataclass
class BandProfile:
    key: str
    gender: str
    event_group: str
    event_times: dict[str, float]
    event_wa: dict[str, float]
    result_was: list[float]
    wa_spread: float
    best_event: str
    events_competed: int
    max_wa: float


def load_profiles(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
) -> dict[str, BandProfile]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    allowed = set(event_name_to_id.values())
    pcol = points_col(gender)
    season_results: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)

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
            key = f"{aid}|{year}"
            season_results[key][result_id] = (id_to_name[event_id], t, wa)

    out: dict[str, BandProfile] = {}
    for key, results in season_results.items():
        event_times: dict[str, float] = {}
        event_wa: dict[str, float] = {}
        result_was: list[float] = []
        for event_name, t, wa in results.values():
            result_was.append(wa)
            if event_name not in event_times or t < event_times[event_name]:
                event_times[event_name] = t
                event_wa[event_name] = wa
        if len(event_times) < 2:
            continue
        wa_vals = list(event_wa.values())
        out[key] = BandProfile(
            key=key,
            gender=gender,
            event_group="",
            event_times=event_times,
            event_wa=event_wa,
            result_was=result_was,
            wa_spread=max(wa_vals) - min(wa_vals),
            best_event=max(event_wa, key=lambda e: event_wa[e]),
            events_competed=len(event_times),
            max_wa=max(wa_vals),
        )
    return out


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


def feature_labelers(
    profiles: list[BandProfile],
    from_ev: str,
    to_ev: str,
) -> dict[str, Callable[[BandProfile], str]]:
    gaps = [abs(p.event_wa[from_ev] - p.event_wa[to_ev]) for p in profiles]
    med_gap = statistics.median(gaps) if gaps else 50.0

    return {
        "bal_spec": lambda p: specialization_label(p.wa_spread, SPREAD_THRESHOLD),
        "best_event": lambda p: f"best_{p.best_event}",
        "best_is_from": lambda p: (
            "best_is_from" if p.best_event == from_ev else "best_not_from"
        ),
        "best_is_to": lambda p: (
            "best_is_to" if p.best_event == to_ev else "best_not_to"
        ),
        "pair_wa_gap_50": lambda p: (
            "pair_gap_ge50"
            if abs(p.event_wa[from_ev] - p.event_wa[to_ev]) >= 50
            else "pair_gap_lt50"
        ),
        "pair_wa_gap_median": lambda p: (
            "pair_gap_large"
            if abs(p.event_wa[from_ev] - p.event_wa[to_ev]) >= med_gap
            else "pair_gap_small"
        ),
        "from_stronger_wa": lambda p: (
            "from_stronger_wa"
            if p.event_wa[from_ev] >= p.event_wa[to_ev]
            else "from_weaker_wa"
        ),
        "events_competed": lambda p: (
            "events_3plus" if p.events_competed >= 3 else "events_2"
        ),
    }


def cv_feature_vs_pooled(
    records: list[tuple[float, float, str]],
) -> dict[str, float] | None:
    """records: (x, y, feature_label). Returns pooled / feature CV median abs."""
    if len(records) < MIN_REPORT_N:
        return None
    n = len(records)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)
    buckets = {"pooled": ([], []), "feature": ([], [])}

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else n
        test_idx = set(idx[start:end])
        train = [records[i] for i in range(n) if i not in test_idx]
        test = [records[i] for i in test_idx]
        if len(train) < 10:
            continue
        pooled = fit_best([(a, b) for a, b, _ in train])
        if pooled is None:
            continue
        _, pp, ppred = pooled

        def pooled_fn(x: float) -> float:
            return ppred(x, pp)

        fac_fits: dict[str, tuple] = {}
        for lab in {lab for _, _, lab in train}:
            subset = [(a, b) for a, b, lab2 in train if lab2 == lab]
            fit = fit_best(subset)
            if fit:
                fac_fits[lab] = fit

        for x, y, lab in test:
            buckets["pooled"][0].append(y)
            buckets["pooled"][1].append(pooled_fn(x))
            buckets["feature"][0].append(y)
            buckets["feature"][1].append(predict_with(fac_fits.get(lab), x, pooled_fn))

    out = {}
    for name, (actual, pred) in buckets.items():
        if not actual:
            return None
        out[name], _ = metrics(actual, pred)
    return out


def cv_combo_strategies(
    records: list[tuple[float, float, str, str, str, str, str]],
) -> dict[str, float] | None:
    """
    records: (x, y, bal_spec, best_event, best_is_from, pair_gap_50, from_stronger)
    """
    if len(records) < MIN_REPORT_N:
        return None
    n = len(records)
    rng = random.Random(CV_SEED)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = max(1, n // CV_FOLDS)
    buckets: dict[str, tuple[list[float], list[float]]] = {
        s: ([], []) for s in COMBO_STRATEGIES
    }

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else n
        test_idx = set(idx[start:end])
        train = [records[i] for i in range(n) if i not in test_idx]
        test = [records[i] for i in test_idx]
        if len(train) < 10:
            continue

        pooled = fit_best([(a, b) for a, b, *_ in train])
        if pooled is None:
            continue
        _, pp, ppred = pooled

        def pooled_fn(x: float) -> float:
            return ppred(x, pp)

        def fit_by(getter):
            fits = {}
            for lab in {getter(r) for r in train}:
                subset = [(r[0], r[1]) for r in train if getter(r) == lab]
                fit = fit_best(subset)
                if fit:
                    fits[lab] = fit
            return fits

        bs_fits = fit_by(lambda r: r[2])
        be_fits = fit_by(lambda r: r[3])
        bif_fits = fit_by(lambda r: r[4])
        gap_fits = fit_by(lambda r: r[5])
        fs_fits = fit_by(lambda r: r[6])

        joint_fits: dict[tuple[str, str], tuple] = {}
        for bs in {r[2] for r in train}:
            for be in {r[3] for r in train}:
                subset = [(r[0], r[1]) for r in train if r[2] == bs and r[3] == be]
                fit = fit_best(subset)
                if fit:
                    joint_fits[(bs, be)] = fit

        for x, y, bs, be, bif, gap, fs in test:
            buckets["pooled"][0].append(y)
            buckets["pooled"][1].append(pooled_fn(x))
            buckets["bal_spec"][0].append(y)
            buckets["bal_spec"][1].append(predict_with(bs_fits.get(bs), x, pooled_fn))
            buckets["best_event"][0].append(y)
            buckets["best_event"][1].append(predict_with(be_fits.get(be), x, pooled_fn))
            buckets["best_is_from"][0].append(y)
            buckets["best_is_from"][1].append(
                predict_with(bif_fits.get(bif), x, pooled_fn)
            )
            buckets["pair_wa_gap_50"][0].append(y)
            buckets["pair_wa_gap_50"][1].append(
                predict_with(gap_fits.get(gap), x, pooled_fn)
            )
            buckets["from_stronger_wa"][0].append(y)
            buckets["from_stronger_wa"][1].append(
                predict_with(fs_fits.get(fs), x, pooled_fn)
            )
            buckets["bal_x_best_event"][0].append(y)
            if (bs, be) in joint_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(joint_fits[(bs, be)], x, pooled_fn)
                )
            elif be in be_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(be_fits[be], x, pooled_fn)
                )
            elif bs in bs_fits:
                buckets["bal_x_best_event"][1].append(
                    predict_with(bs_fits[bs], x, pooled_fn)
                )
            else:
                buckets["bal_x_best_event"][1].append(pooled_fn(x))

    out = {}
    for name, (actual, pred) in buckets.items():
        if not actual:
            return None
        out[name], _ = metrics(actual, pred)
    return out


FEATURE_DESCRIPTIONS = {
    "bal_spec": f"balanced vs specialized (wa_spread ≥ {SPREAD_THRESHOLD:.0f})",
    "best_event": "Identity of best-WA event in the group",
    "best_is_from": "Athlete's best-WA event is the source event",
    "best_is_to": "Athlete's best-WA event is the target event",
    "pair_wa_gap_50": "|WA(from)−WA(to)| ≥ 50",
    "pair_wa_gap_median": "|WA(from)−WA(to)| relative to band-pair median",
    "from_stronger_wa": "Source event has higher WA than target",
    "events_competed": "Distinct events raced: 2 vs 3+",
    "bal_x_best_event": "bal/spec × best_event joint route",
}


def analyze_band(
    lo: int,
    hi: int,
    profiles_by_gg: dict[tuple[str, str], dict[str, BandProfile]],
) -> dict:
    label = f"{lo}-{hi}"
    combo_rows: list[dict] = []
    feature_rows: list[dict] = []
    summary_feature: list[dict] = []

    for (gender, event_group), profiles in profiles_by_gg.items():
        band_profiles = [
            p for p in profiles.values() if in_band(p.result_was, lo, hi)
        ]
        order = EVENT_ORDER[event_group]
        for from_ev, to_ev in permutations(order, 2):
            pair_ps = [
                p
                for p in band_profiles
                if from_ev in p.event_times and to_ev in p.event_times
            ]
            if len(pair_ps) < MIN_REPORT_N:
                continue

            labelers = feature_labelers(pair_ps, from_ev, to_ev)

            # Combo strategies
            combo_records = [
                (
                    p.event_times[from_ev],
                    p.event_times[to_ev],
                    labelers["bal_spec"](p),
                    labelers["best_event"](p),
                    labelers["best_is_from"](p),
                    labelers["pair_wa_gap_50"](p),
                    labelers["from_stronger_wa"](p),
                )
                for p in pair_ps
            ]
            combo = cv_combo_strategies(combo_records)
            if combo:
                best_name = min(
                    (s for s in COMBO_STRATEGIES if s in combo),
                    key=lambda s: combo[s],
                )
                row = {
                    "band": label,
                    "band_lo": lo,
                    "band_hi": hi,
                    "gender": gender,
                    "event_group": event_group,
                    "from_event": from_ev,
                    "to_event": to_ev,
                    "n": len(pair_ps),
                    "best_strategy": best_name,
                    **{f"cv_{s}": round(combo[s], 4) for s in COMBO_STRATEGIES},
                    "delta_best_vs_pooled": round(
                        combo["pooled"] - combo[best_name], 4
                    ),
                }
                combo_rows.append(row)

            # Per-feature alone vs pooled
            for fname in FEATURE_NAMES:
                records = [
                    (
                        p.event_times[from_ev],
                        p.event_times[to_ev],
                        labelers[fname](p),
                    )
                    for p in pair_ps
                ]
                # skip if only one label present
                if len({r[2] for r in records}) < 2:
                    continue
                res = cv_feature_vs_pooled(records)
                if not res:
                    continue
                feature_rows.append(
                    {
                        "band": label,
                        "band_lo": lo,
                        "band_hi": hi,
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n": len(pair_ps),
                        "feature": fname,
                        "cv_pooled": round(res["pooled"], 4),
                        "cv_feature": round(res["feature"], 4),
                        "delta_pooled_minus_feature": round(
                            res["pooled"] - res["feature"], 4
                        ),
                        "feature_beats_pooled": res["feature"] < res["pooled"] - EPS,
                    }
                )

    # Summarize features within this band
    by_feat: dict[str, list[dict]] = defaultdict(list)
    for r in feature_rows:
        by_feat[r["feature"]].append(r)
    for fname, rows in by_feat.items():
        beats = sum(1 for r in rows if r["feature_beats_pooled"])
        deltas = [r["delta_pooled_minus_feature"] for r in rows]
        summary_feature.append(
            {
                "band": label,
                "feature": fname,
                "n_pairs": len(rows),
                "beats_pooled": beats,
                "beat_rate": round(beats / len(rows), 3) if rows else 0,
                "mean_delta": round(statistics.mean(deltas), 4) if deltas else None,
                "median_delta": round(statistics.median(deltas), 4) if deltas else None,
            }
        )

    # Combo summary
    strategy_wins: dict[str, int] = defaultdict(int)
    for r in combo_rows:
        strategy_wins[r["best_strategy"]] += 1
    combo_summary = []
    for s in COMBO_STRATEGIES:
        if s == "pooled":
            continue
        deltas = [r[f"cv_pooled"] - r[f"cv_{s}"] for r in combo_rows]
        beats = sum(1 for d in deltas if d > EPS)
        combo_summary.append(
            {
                "band": label,
                "strategy": s,
                "n_pairs": len(combo_rows),
                "beats_pooled": beats,
                "beat_rate": round(beats / len(combo_rows), 3) if combo_rows else 0,
                "mean_delta": round(statistics.mean(deltas), 4) if deltas else None,
                "times_best": strategy_wins.get(s, 0),
            }
        )
    pooled_best = strategy_wins.get("pooled", 0)

    return {
        "band": label,
        "lo": lo,
        "hi": hi,
        "combo_rows": combo_rows,
        "feature_rows": feature_rows,
        "summary_feature": summary_feature,
        "combo_summary": combo_summary,
        "strategy_wins": dict(strategy_wins),
        "pooled_best_count": pooled_best,
        "n_combo_pairs": len(combo_rows),
        "n_athlete_seasons": sum(
            len([p for p in profiles.values() if in_band(p.result_was, lo, hi)])
            for profiles in profiles_by_gg.values()
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_reports(results: list[dict]) -> None:
    # Per-band reports
    for result in results:
        band = result["band"]
        lines = [
            f"Feature Importance within WA Point Band {band}",
            "=" * (44 + len(band)),
            "",
            "Question: Do previously important features (bal/spec, best_event,",
            "best_is_from, pair WA gap, from_stronger_wa, bal×best_event) make",
            f"the band {band} time models more accurate than the pooled band model?",
            "",
            "Method:",
            "  • Same population as Point_Bands_Time_Models (March 1+, no relays/steeple).",
            f"  • Inclusion: ≥1 result with WA in [{result['lo']}, {result['hi']}).",
            "  • For each event pair (n≥20): 5-fold CV routes predictions by feature",
            f"    labels; sub-cohorts need n≥{MIN_COHORT_N} to fit a dedicated model,",
            "    else fall back to the pooled band formula.",
            f"  • “Beats pooled” = CV median |error| improves by >{EPS}s.",
            f"  • Athlete-seasons in band: {result['n_athlete_seasons']}.",
            f"  • Evaluated pairs: {result['n_combo_pairs']}.",
            "",
            "Strategy leaderboard vs pooled band model",
            "-" * 40,
        ]
        ranked = sorted(
            result["combo_summary"],
            key=lambda r: (-(r["beat_rate"] or 0), -(r["mean_delta"] or 0)),
        )
        lines.append(
            f"  {'Strategy':<22} {'Beats':>8} {'Mean Δ':>10} {'#Best':>6}"
        )
        for s in ranked:
            lines.append(
                f"  {s['strategy']:<22} {s['beats_pooled']:>3}/{s['n_pairs']:<3} "
                f"{s['mean_delta']:>+9.3f}s {s['times_best']:>6}"
            )
        lines.append(
            f"  {'pooled (no split)':<22} {'—':>8} {'—':>10} "
            f"{result['pooled_best_count']:>6}"
        )
        lines.append("")
        lines.append("Feature-alone vs pooled (secondary table)")
        lines.append("-" * 40)
        feat_ranked = sorted(
            result["summary_feature"],
            key=lambda r: (-(r["beat_rate"] or 0), -(r["mean_delta"] or 0)),
        )
        for s in feat_ranked:
            desc = FEATURE_DESCRIPTIONS.get(s["feature"], "")
            lines.append(
                f"  {s['feature']:<22} {s['beats_pooled']:>3}/{s['n_pairs']:<3} "
                f"mean Δ {s['mean_delta']:+.3f}s  — {desc}"
            )

        lines.extend(["", "Per-pair best strategy", "-" * 40])
        for r in sorted(
            result["combo_rows"],
            key=lambda x: (
                x["event_group"],
                x["gender"],
                x["from_event"],
                x["to_event"],
            ),
        ):
            best_s = r["best_strategy"]
            best_cv = r[f"cv_{best_s}"]
            lines.append(
                f"  {r['gender']} {r['event_group']} {r['from_event']}->{r['to_event']} "
                f"n={r['n']}  pooled={r['cv_pooled']:.3f}s  "
                f"best={best_s} ({best_cv:.3f}s, Δ={r['delta_best_vs_pooled']:+.3f}s)"
            )

        # Verdict for this band
        helpful = [
            s
            for s in ranked
            if s["beat_rate"] >= 0.5 and (s["mean_delta"] or 0) > EPS
        ]
        lines.extend(["", "Band verdict", "-" * 12])
        if helpful:
            names = ", ".join(s["strategy"] for s in helpful)
            lines.append(
                f"  YES — {names} beat the pooled {band} model on ≥50% of pairs "
                f"with mean improvement >{EPS}s."
            )
        elif any((s["mean_delta"] or 0) > EPS for s in ranked):
            top = ranked[0]
            lines.append(
                f"  MIXED — best lift from {top['strategy']} "
                f"({top['beats_pooled']}/{top['n_pairs']}, mean Δ {top['mean_delta']:+.3f}s); "
                "not a clear majority win."
            )
        else:
            lines.append(
                f"  NO — no tested feature consistently beats the pooled {band} model "
                "(mean Δ ≤ 0 or rare wins)."
            )

        lines.extend(
            [
                "",
                "Source: Feature_Importance_Point_Band_Time_Models/"
                "analyze_feature_importance_point_bands.py",
            ]
        )
        (
            OUTPUT_ROOT / f"feature_importance_band_{result['lo']}_{result['hi']}_report.txt"
        ).write_text("\n".join(lines).rstrip() + "\n")

    # Combined findings
    findings = [
        "Feature Importance in Point-Band Time Models — Findings",
        "=======================================================",
        "",
        "Bands tested (width 200): 750–950, 800–1000, 850–1050.",
        "Baseline: pooled band time model (same as formula files).",
        "Features: bal/spec, best_event, best_is_from, pair_wa_gap_50,",
        "          from_stronger_wa, bal×best_event (+ secondary factor scan).",
        "",
    ]
    any_yes = False
    for result in results:
        band = result["band"]
        findings.append(f"Band {band} ({result['n_combo_pairs']} pairs, "
                        f"{result['n_athlete_seasons']} athlete-seasons):")
        ranked = sorted(
            result["combo_summary"],
            key=lambda r: (-(r["beat_rate"] or 0), -(r["mean_delta"] or 0)),
        )
        for s in ranked[:4]:
            findings.append(
                f"  {s['strategy']:<22} beats pooled {s['beats_pooled']}/{s['n_pairs']} "
                f"(mean Δ {s['mean_delta']:+.3f}s); best on {s['times_best']} pairs"
            )
        findings.append(
            f"  pooled best on {result['pooled_best_count']}/{result['n_combo_pairs']} pairs"
        )
        helpful = [
            s
            for s in ranked
            if s["beat_rate"] >= 0.5 and (s["mean_delta"] or 0) > EPS
        ]
        if helpful:
            any_yes = True
            findings.append(
                "  Verdict: YES — "
                + ", ".join(s["strategy"] for s in helpful)
                + " help vs pooled."
            )
        elif ranked and (ranked[0]["mean_delta"] or 0) > EPS:
            findings.append(
                f"  Verdict: MIXED — modest lift from {ranked[0]['strategy']} "
                f"({ranked[0]['beats_pooled']}/{ranked[0]['n_pairs']})."
            )
        else:
            findings.append("  Verdict: NO clear improvement over pooled band model.")
        findings.append("")

    findings.append("Overall:")
    if any_yes:
        findings.append(
            "  At least one band shows a feature that reliably beats the pooled"
        )
        findings.append("  band model. Prefer the winning route when labels are known.")
    else:
        findings.append(
            "  Within these high-WA bands, prior features rarely beat the pooled"
        )
        findings.append(
            "  band model by much; ability banding already captures much of the signal."
        )
    findings.extend(
        [
            "",
            "See feature_importance_band_*_report.txt for pair detail.",
            "Source: analyze_feature_importance_point_bands.py",
        ]
    )
    (OUTPUT_ROOT / "feature_importance_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )

    # Combined short report
    lines = [
        "Feature Importance × Point-Band Time Models — Combined Report",
        "=============================================================",
        "",
        "Does bal/spec, best_event, or related features improve accuracy inside",
        "the 750–950, 800–1000, and 850–1050 WA bands?",
        "",
    ]
    for result in results:
        lines.append(f"=== Band {result['band']} ===")
        ranked = sorted(
            result["combo_summary"],
            key=lambda r: (-(r["beat_rate"] or 0), -(r["mean_delta"] or 0)),
        )
        for s in ranked:
            lines.append(
                f"  {s['strategy']:<22} {s['beats_pooled']:>2}/{s['n_pairs']:<2} "
                f"mean Δ {s['mean_delta']:+.3f}s  times_best={s['times_best']}"
            )
        lines.append(
            f"  pooled best={result['pooled_best_count']}/{result['n_combo_pairs']}"
        )
        lines.append("")
    lines.append("Source: analyze_feature_importance_point_bands.py")
    (OUTPUT_ROOT / "feature_importance_combined_report.txt").write_text(
        "\n".join(lines).rstrip() + "\n"
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    profiles_by_gg: dict[tuple[str, str], dict[str, BandProfile]] = {}
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            profiles = load_profiles(folder, prefix, gender, events)
            for p in profiles.values():
                p.event_group = event_group
            profiles_by_gg[(gender, event_group)] = profiles
            print(f"Loaded {gender} {event_group}: {len(profiles)} athlete-seasons")

    all_combo: list[dict] = []
    all_feat: list[dict] = []
    all_sum_feat: list[dict] = []
    all_sum_combo: list[dict] = []
    results: list[dict] = []

    for lo, hi in TARGET_BANDS:
        print(f"Analyzing band {lo}-{hi}...")
        result = analyze_band(lo, hi, profiles_by_gg)
        results.append(result)
        all_combo.extend(result["combo_rows"])
        all_feat.extend(result["feature_rows"])
        all_sum_feat.extend(result["summary_feature"])
        all_sum_combo.extend(result["combo_summary"])
        best = sorted(
            result["combo_summary"],
            key=lambda r: (-(r["beat_rate"] or 0), -(r["mean_delta"] or 0)),
        )
        if best:
            b0 = best[0]
            print(
                f"  top strategy {b0['strategy']}: beats pooled "
                f"{b0['beats_pooled']}/{b0['n_pairs']} mean Δ {b0['mean_delta']:+.3f}s"
            )

    write_csv(OUTPUT_ROOT / "pair_strategy_cv_by_band.csv", all_combo)
    write_csv(OUTPUT_ROOT / "feature_alone_cv_by_band.csv", all_feat)
    write_csv(OUTPUT_ROOT / "feature_summary_by_band.csv", all_sum_feat)
    write_csv(OUTPUT_ROOT / "strategy_summary_by_band.csv", all_sum_combo)
    write_reports(results)
    print(f"Wrote feature-importance analysis to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
