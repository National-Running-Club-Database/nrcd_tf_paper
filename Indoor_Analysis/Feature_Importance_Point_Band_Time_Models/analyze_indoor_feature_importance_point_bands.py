"""Indoor feature-importance point-band time models (750–950, 800–1000).

Produces reports modeled on:
  time_models/Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/
  new_feature_important_time_models_band_800_1000.txt

Outputs under:
  indoor_analysis/Feature_Importance_Point_Band_Time_Models/
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
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDOOR_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
TIME_MODELS_ROOT = PROJECT_ROOT / "time_models"
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
SPEC_ROOT = TIME_MODELS_ROOT / "specialized_time_models"

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))

from relay_rq1_data import parse_performance  # noqa: E402
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
from analyze_specialization_time_models import specialization_label  # noqa: E402

SEASONS = ("2024", "2025", "2026")
STANDARD_RELAY_IDS = {21, 22, 23, 24, 25, 26, 29, 30, 31}
MIN_COHORT_N = 12
MIN_REPORT_N = 20
SPREAD_THRESHOLD = 50.0
EPS = 0.01

TARGET_BANDS = ((750, 950), (800, 1000))

SPRINT_EVENTS = {"60m": 2, "200m": 4, "400m": 6}
DISTANCE_EVENTS = {"800m": 9, "Mile": 13, "3000m": 14}
EVENT_ORDER = {
    "Sprints": ["60m", "200m", "400m"],
    "Distance": ["800m", "Mile", "3000m"],
}
GROUP_FOLDERS = {
    "Sprints": ("Indoor_Sprints", "Sprints", SPRINT_EVENTS),
    "Distance": ("Indoor_Distance", "Distance", DISTANCE_EVENTS),
}

COMBO_STRATEGIES = (
    "pooled",
    "bal_spec",
    "best_event",
    "best_is_from",
    "pair_wa_gap_50",
    "from_stronger_wa",
    "bal_x_best_event",
)

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

FIT_PRED: dict[str, tuple[Callable | None, Callable | None]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
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
    "best_is_to": "Athlete's best-WA event is the target event",
    "pair_wa_gap_50": "|WA(from)−WA(to)| ≥ 50",
    "pair_wa_gap_median": "|WA(from)−WA(to)| relative to band-pair median",
    "from_stronger_wa": "Source event has higher WA than target",
    "events_competed": "Distinct events raced: 2 vs 3+",
    "bal_x_best_event": "bal/spec × best_event joint route",
}

FEATURE_LABEL_FNS = {
    "bal_spec": "bal_spec",
    "best_event": "best_event",
    "best_is_from": "best_is_from",
    "pair_wa_gap_50": "pair_wa_gap_50",
    "from_stronger_wa": "from_stronger_wa",
}

PARAMETRIC_MODELS = {
    "linear_ols",
    "robust_trimmed_linear",
    "median_ratio",
    "log_linear",
    "ratio_linear",
    "quadratic",
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


def points_col(gender: str, metric: str = "wa") -> str:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents
    # walk up until scoring package is importable
    for p in Path(__file__).resolve().parents:
        if (p / "scoring" / "columns.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    from scoring.columns import points_col as _pc
    return _pc(gender, metric)


def in_band(result_was: list[float], lo: int, hi: int) -> bool:
    return any(lo <= wa < hi for wa in result_was)


def csv_path(folder: str, group: str, gender: str, year: str) -> Path:
    return INDOOR_ROOT / folder / f"Indoor_Relays_{group}_{gender}_{year}_Data.csv"


def load_profiles(
    folder: str,
    group: str,
    gender: str,
    event_name_to_id: dict[str, int],
) -> dict[str, BandProfile]:
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    allowed = set(event_name_to_id.values())
    pcol = points_col(gender)
    season_results: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)

    for year in SEASONS:
        path = csv_path(folder, group, gender, year)
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    event_id = int(float(row["running_event_id"]))
                except (TypeError, ValueError):
                    continue
                if event_id in STANDARD_RELAY_IDS or event_id not in allowed:
                    continue
                wa = float(row.get(pcol) or 0)
                if wa <= 0:
                    continue
                aid = (row.get("athlete_id") or "").strip()
                if not aid or aid.lower() == "nan":
                    continue
                try:
                    aid = str(int(float(aid)))
                except ValueError:
                    continue
                t = parse_performance(row.get("result_time", ""), event_id)
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

        def pooled_fn(x: float, _pp=pp, _ppred=ppred) -> float:
            return _ppred(x, _pp)

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

        def pooled_fn(x: float, _pp=pp, _ppred=ppred) -> float:
            return _ppred(x, _pp)

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

    if winner.name not in PARAMETRIC_MODELS and winner.name != "binned_ratio":
        parametric = [r for r in results if r.name in PARAMETRIC_MODELS]
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
        key = STRATEGY_CV_KEYS[strat]
        cv = float(cv_row[key])
        delta = pooled - cv
        if delta > EPS:
            out.append((strat, cv, delta))
    out.sort(key=lambda t: -t[2])
    return out


def cohort_formulas_for_strategy(
    profiles: list[BandProfile],
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
    cells: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in profiles:
        cells[labelers[lab_key](p)].append(
            (p.event_times[from_ev], p.event_times[to_ev])
        )
    for lab, pairs in sorted(cells.items()):
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def analyze_band(
    lo: int,
    hi: int,
    profiles_by_gg: dict[tuple[str, str], dict[str, BandProfile]],
) -> dict:
    label = f"{lo}-{hi}"
    combo_rows: list[dict] = []
    feature_rows: list[dict] = []

    for (gender, event_group), profiles in profiles_by_gg.items():
        band_profiles = [p for p in profiles.values() if in_band(p.result_was, lo, hi)]
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
                combo_rows.append(
                    {
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
                )

            for fname in FEATURE_NAMES:
                records = [
                    (
                        p.event_times[from_ev],
                        p.event_times[to_ev],
                        labelers[fname](p),
                    )
                    for p in pair_ps
                ]
                if len({r[2] for r in records}) < 2:
                    continue
                res = cv_feature_vs_pooled(records)
                if not res:
                    continue
                feature_rows.append(
                    {
                        "band": label,
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

    return {"combo_rows": combo_rows, "feature_rows": feature_rows, "band": label, "lo": lo, "hi": hi}


def write_band_report(result: dict, n_as: int, path: Path) -> None:
    label = result["band"]
    lo, hi = result["lo"], result["hi"]
    combo_rows = result["combo_rows"]
    feature_rows = result["feature_rows"]

    lines = [
        f"Indoor Feature Importance within WA Point Band {label}",
        "=" * (52 + len(label)),
        "",
        "Question: Do previously important features (bal/spec, best_event,",
        "best_is_from, pair WA gap, from_stronger_wa, bal×best_event) make",
        f"the indoor band {label} time models more accurate than the pooled band model?",
        "",
        "Method:",
        "  • Indoor CSVs 2024–2026 (all dates in files); no relays.",
        "  • Events: Sprints 60/200/400; Distance 800/Mile/3000.",
        f"  • Inclusion: ≥1 result with WA in [{lo}, {hi}).",
        "  • For each event pair (n≥20): 5-fold CV routes predictions by feature",
        "    labels; sub-cohorts need n≥12 to fit a dedicated model,",
        "    else fall back to the pooled band formula.",
        "  • “Beats pooled” = CV median |error| improves by >0.01s.",
        f"  • Athlete-seasons in band: {n_as}.",
        f"  • Evaluated pairs: {len(combo_rows)}.",
        "",
        "Feature definitions",
        "-------------------",
    ]
    for name in list(COMBO_STRATEGIES) + [
        f for f in FEATURE_NAMES if f not in COMBO_STRATEGIES
    ]:
        if name == "pooled":
            continue
        desc = FEATURE_DESCRIPTIONS.get(name, "")
        how = FEATURE_HOW_TO_KNOW.get(name, "")
        lines.append(f"  • {name}: {desc}")
        if how:
            lines.append(f"      How to use: {how}")
    lines.append("")

    # Strategy leaderboard
    lines.append("Strategy leaderboard vs pooled band model")
    lines.append("----------------------------------------")
    lines.append(f"  {'Strategy':<22} {'Beats':>6} {'Mean Δ':>10} {'#Best':>6}")
    strat_stats = []
    for strat in COMBO_STRATEGIES:
        if strat == "pooled":
            continue
        key = STRATEGY_CV_KEYS[strat]
        deltas = [float(r["cv_pooled"]) - float(r[key]) for r in combo_rows]
        beats = sum(1 for d in deltas if d > EPS)
        mean_d = statistics.mean(deltas) if deltas else 0.0
        n_best = sum(1 for r in combo_rows if r["best_strategy"] == strat)
        strat_stats.append((strat, beats, mean_d, n_best, len(combo_rows)))
    strat_stats.sort(key=lambda t: (-t[1], -t[2]))
    for strat, beats, mean_d, n_best, n_tot in strat_stats:
        lines.append(
            f"  {strat:<22} {beats:>2}/{n_tot:<3} {mean_d:>+9.3f}s {n_best:>6}"
        )
    n_pooled_best = sum(1 for r in combo_rows if r["best_strategy"] == "pooled")
    lines.append(f"  {'pooled (no split)':<22} {'—':>6} {'—':>10} {n_pooled_best:>6}")
    lines.append("")

    lines.append("Feature-alone vs pooled (secondary table)")
    lines.append("----------------------------------------")
    by_feat: dict[str, list[dict]] = defaultdict(list)
    for r in feature_rows:
        by_feat[r["feature"]].append(r)
    feat_summ = []
    for fname, rows in by_feat.items():
        beats = sum(1 for r in rows if r["feature_beats_pooled"])
        deltas = [r["delta_pooled_minus_feature"] for r in rows]
        feat_summ.append(
            (
                fname,
                beats,
                len(rows),
                statistics.mean(deltas) if deltas else 0.0,
            )
        )
    feat_summ.sort(key=lambda t: (-t[1], -t[3]))
    for fname, beats, n, mean_d in feat_summ:
        desc = FEATURE_DESCRIPTIONS.get(fname, "")
        lines.append(
            f"  {fname:<22} {beats:>2}/{n:<3}  mean Δ {mean_d:+.3f}s  — {desc}"
        )
    lines.append("")

    lines.append("Per-pair best strategy")
    lines.append("----------------------------------------")
    for r in sorted(
        combo_rows,
        key=lambda x: (x["event_group"], x["gender"], x["from_event"], x["to_event"]),
    ):
        lines.append(
            f"  {r['gender']} {r['event_group']} {r['from_event']}->{r['to_event']} "
            f"n={r['n']}  pooled={r['cv_pooled']:.3f}s  "
            f"best={r['best_strategy']} ({r['cv_' + r['best_strategy']]:.3f}s, "
            f"Δ={r['delta_best_vs_pooled']:+.3f}s)"
        )
    lines.append("")

    helpful = [
        s
        for s, beats, mean_d, _, n in strat_stats
        if beats >= max(1, n // 2) and mean_d > EPS
    ]
    lines.append("Band verdict")
    lines.append("------------")
    if helpful:
        lines.append(
            f"  YES — {', '.join(helpful)} beat the pooled {label} model on ≥50% of "
            "pairs with mean improvement >0.01s."
        )
    else:
        lines.append(
            "  MIXED — features help on some pairs but do not dominate across ≥50% "
            "with mean Δ >0.01s; still prefer pair-level recommended routes below."
        )
    lines.append("")
    lines.append(
        "Source: indoor_analysis/Feature_Importance_Point_Band_Time_Models/"
        "analyze_indoor_feature_importance_point_bands.py"
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_feature_important_file(
    lo: int,
    hi: int,
    profiles_by_gg: dict,
    cv_index: dict,
) -> Path:
    band = f"{lo}-{hi}"
    title = f"Indoor Feature-Important Time Models — WA Band {band} (Width 200)"
    lines = [
        title,
        "=" * len(title),
        "",
        "Method:",
        "  • Data: indoor_analysis CSVs 2024–2026 (all dates in files).",
        f"  • Population: athlete-seasons with ≥1 individual result in WA [{lo}, {hi}).",
        "  • Excludes relays. Sprints (60/200/400) & Distance (800/Mile/3000) only.",
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
            "",
            "Source:",
            "  indoor_analysis/Feature_Importance_Point_Band_Time_Models/"
            "analyze_indoor_feature_importance_point_bands.py",
            "  Feature importance CV: pair_strategy_cv_by_band.csv",
        ]
    )

    out = OUTPUT_ROOT / f"new_feature_important_time_models_band_{lo}_{hi}.txt"
    out.write_text("\n".join(lines).rstrip() + "\n")
    return out


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    profiles_by_gg: dict[tuple[str, str], dict[str, BandProfile]] = {}
    for event_group, (folder, group, events) in GROUP_FOLDERS.items():
        for gender in ("Men", "Women"):
            profiles = load_profiles(folder, group, gender, events)
            for p in profiles.values():
                p.event_group = event_group
            profiles_by_gg[(gender, event_group)] = profiles
            print(f"Loaded {gender} {event_group}: {len(profiles)}")

    all_combo: list[dict] = []
    all_feat: list[dict] = []
    cv_index: dict[tuple, dict] = {}

    for lo, hi in TARGET_BANDS:
        print(f"Analyzing band {lo}-{hi}...")
        result = analyze_band(lo, hi, profiles_by_gg)
        all_combo.extend(result["combo_rows"])
        all_feat.extend(result["feature_rows"])
        for r in result["combo_rows"]:
            key = (
                r["band"],
                r["gender"],
                r["event_group"],
                r["from_event"],
                r["to_event"],
            )
            cv_index[key] = r

        n_as = sum(
            1
            for profiles in profiles_by_gg.values()
            for p in profiles.values()
            if in_band(p.result_was, lo, hi)
        )
        write_band_report(
            result,
            n_as,
            OUTPUT_ROOT / f"feature_importance_band_{lo}_{hi}_report.txt",
        )
        path = write_feature_important_file(lo, hi, profiles_by_gg, cv_index)
        print(f"Wrote {path.name} ({len(result['combo_rows'])} pairs, {n_as} athlete-seasons)")

    write_csv(OUTPUT_ROOT / "pair_strategy_cv_by_band.csv", all_combo)
    write_csv(OUTPUT_ROOT / "feature_alone_cv_by_band.csv", all_feat)
    print(f"Wrote CSVs and reports to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
