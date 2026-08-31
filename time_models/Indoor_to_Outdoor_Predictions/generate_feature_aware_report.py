"""Feature-aware Indoor → Outdoor formula report.

Mirrors the style of:
  time_models/Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/
  new_feature_important_time_models_band_750_950.txt

For each indoor→outdoor pair and gender:
  1. Pooled formula (best of linear / median_ratio / log_linear / robust by CV)
  2. Which routing features beat pooled (bal_spec, best_event, best_is_from,
     from_stronger_wa, pair_wa_gap_50, bal_x_best_event)
  3. Cohort formulas showing how those features alter the prediction

Also writes WA-band reports (750–950, 800–1000, 850–1050) with band membership
based on the indoor from-event season-best World Athletics score (WA of the
fastest / best indoor mark that season).
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
BANDS_DIR = OUT / "by_indoor_wa_band"
BANDS_DIR.mkdir(parents=True, exist_ok=True)

from analyze_indoor_to_outdoor import (  # noqa: E402
    INDOOR_EVENTS,
    OUTDOOR_EVENTS,
    PAIRS,
    SEASONS,
    better,
    format_mark,
    indoor_path,
    is_field,
    load_csv,
    normalize,
    outdoor_paths,
    parse_performance,
)


def wa_col(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


CV_FOLDS = 5
CV_SEED = 42
MIN_PAIR_N = 20
MIN_COHORT_N = 12
SPREAD_THRESHOLD = 50.0
EPS = 0.01  # improvement vs pooled (seconds or meters)

# Width-200 bands; athlete placed by indoor from-event season-best WA
TARGET_BANDS = (
    (750, 950),
    (800, 1000),
    (850, 1050),
)

# Outdoor events used for profile features within each group
GROUP_OUTDOOR_IDS: dict[str, list[int]] = {
    "Sprints": [3, 4, 6],
    "Distance": [9, 11, 17, 20],
    "Jumps": [38, 39, 40],
    "Hurdles": [34, 35],
    "Throws": [41],
}

STRATEGIES = (
    "pooled",
    "bal_spec",
    "best_event",
    "best_is_from",
    "from_stronger_wa",
    "pair_wa_gap_50",
    "bal_x_best_event",
)

FEATURE_HOW_TO = {
    "pooled": "No extra labels — use the single pooled formula.",
    "bal_spec": (
        f"Compute wa_spread = max(event WA) − min(event WA) across outdoor season PBs "
        f"in the event group. specialized if ≥ {SPREAD_THRESHOLD:.0f}; else balanced."
    ),
    "best_event": (
        "Identify which outdoor event in the group has the athlete's highest season WA; "
        "route to the best_<event> formula."
    ),
    "best_is_from": (
        "Check whether the indoor from-event holds the athlete's highest WA among "
        "indoor-from + outdoor group PBs. Yes → best_is_from; no → best_not_from."
    ),
    "from_stronger_wa": (
        "Compare WA(indoor from) to WA(outdoor to). If from ≥ to, use from_stronger_wa; "
        "else to_stronger_wa. (Requires knowing outdoor to-event WA — validation routing.)"
    ),
    "pair_wa_gap_50": (
        "If |WA(indoor from) − WA(outdoor to)| ≥ 50, use large_gap; else small_gap."
    ),
    "bal_x_best_event": (
        "Need both bal/spec and outdoor best_event. Prefer matching cell formula when "
        "n is large enough; else fall back to best_event, then bal/spec, then pooled."
    ),
}


def linreg_pairs(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    den_r = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    r = num / den_r if den_r else 0.0
    return intercept, slope, r


def fit_linear(train: list[tuple[float, float]]) -> dict:
    intercept, slope, _ = linreg_pairs(train)
    return {"model": "linear_ols", "intercept": intercept, "slope": slope}


def pred_linear(x: float, p: dict) -> float:
    return p["intercept"] + p["slope"] * x


def fit_median_ratio(train: list[tuple[float, float]]) -> dict:
    ratios = sorted(y / x for x, y in train if x > 0)
    return {"model": "median_ratio", "ratio": statistics.median(ratios)}


def pred_median_ratio(x: float, p: dict) -> float:
    return x * p["ratio"]


def fit_log_linear(train: list[tuple[float, float]]) -> dict:
    log_pairs = [(math.log(x), math.log(y)) for x, y in train if x > 0 and y > 0]
    if len(log_pairs) < 5:
        return fit_linear(train)
    intercept, slope, _ = linreg_pairs(log_pairs)
    return {"model": "log_linear", "intercept": intercept, "slope": slope}


def pred_log_linear(x: float, p: dict) -> float:
    if p.get("model") != "log_linear":
        return pred_linear(x, p)
    return math.exp(p["intercept"] + p["slope"] * math.log(max(x, 1e-9)))


def fit_robust(train: list[tuple[float, float]]) -> dict:
    """Trim 10% extremes by y, then OLS."""
    if len(train) < 15:
        return fit_linear(train)
    ys = sorted(y for _, y in train)
    lo, hi = ys[len(ys) // 10], ys[-(len(ys) // 10 + 1)]
    trimmed = [(x, y) for x, y in train if lo <= y <= hi]
    if len(trimmed) < 10:
        trimmed = train
    p = fit_linear(trimmed)
    p["model"] = "robust_trimmed_linear"
    return p


FIT_PRED: dict[str, tuple[Callable, Callable]] = {
    "linear_ols": (fit_linear, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "robust_trimmed_linear": (fit_robust, pred_linear),
}


def human_formula(params: dict, from_ev: str, to_ev: str) -> str:
    m = params.get("model", "linear_ols")
    if m == "median_ratio":
        return f"{to_ev} = {params['ratio']:.4f} × {from_ev}"
    if m == "log_linear":
        return f"{to_ev} = exp({params['intercept']:.3f} + {params['slope']:.3f}×log({from_ev}))"
    intercept, slope = params["intercept"], params["slope"]
    if abs(intercept) < 1e-6:
        return f"{to_ev} = {slope:.4f} × {from_ev}"
    if intercept >= 0:
        return f"{to_ev} = {intercept:.4f} + {slope:.4f} × {from_ev}"
    return f"{to_ev} = {slope:.4f} × {from_ev} − {abs(intercept):.4f}"


def cv_median_abs(
    pairs: list[tuple[float, float]],
    fit: Callable,
    pred: Callable,
) -> float:
    if len(pairs) < CV_FOLDS * 2:
        p = fit(pairs)
        errs = [abs(y - pred(x, p)) for x, y in pairs]
        return statistics.median(errs)
    rng = random.Random(CV_SEED)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    fold = len(pairs) // CV_FOLDS
    actual, preds = [], []
    for f in range(CV_FOLDS):
        start = f * fold
        end = start + fold if f < CV_FOLDS - 1 else len(pairs)
        test_i = set(idx[start:end])
        train = [pairs[i] for i in range(len(pairs)) if i not in test_i]
        test = [pairs[i] for i in test_i]
        if len(train) < 8:
            continue
        p = fit(train)
        for x, y in test:
            actual.append(y)
            preds.append(pred(x, p))
    if not actual:
        return float("nan")
    return statistics.median(abs(a - b) for a, b in zip(actual, preds))


def choose_pooled_model(pairs: list[tuple[float, float]]) -> dict:
    best_name, best_cv, best_params = "linear_ols", float("inf"), fit_linear(pairs)
    for name, (fit, pred) in FIT_PRED.items():
        try:
            cv = cv_median_abs(pairs, fit, pred)
            params = fit(pairs)
            if cv < best_cv:
                best_name, best_cv, best_params = name, cv, params
        except Exception:
            continue
    best_params = dict(best_params)
    best_params["model"] = best_name
    best_params["cv_medae"] = best_cv
    return best_params


@dataclass
class AthletePair:
    athlete_id: int
    season_year: int
    gender: str
    event_group: str
    from_event: str
    to_event: str
    from_mark: float
    to_mark: float
    from_wa: float
    to_wa: float
    outdoor_wa: dict[str, float] = field(default_factory=dict)  # outdoor event label → WA
    outdoor_marks: dict[str, float] = field(default_factory=dict)


def load_expanded_outdoor(group: str, gender: str) -> pd.DataFrame:
    ids = set(GROUP_OUTDOOR_IDS.get(group, []))
    parts = []
    for year in SEASONS:
        for path in outdoor_paths(group, gender, year):
            raw = load_csv(path)
            part = normalize(raw, ids, year)
            if not part.empty:
                parts.append(part)
                break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_indoor_for_ids(group: str, gender: str, ids: set[int]) -> pd.DataFrame:
    parts = []
    for year in SEASONS:
        part = normalize(load_csv(indoor_path(group, gender, year)), ids, year)
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def best_mark_wa(
    df: pd.DataFrame, event_id: int, gender: str
) -> dict[tuple[int, int], tuple[float, float]]:
    """key -> (best_mark, wa_at_best)."""
    field = is_field(event_id)
    pc = wa_col(gender)
    out: dict[tuple[int, int], tuple[float, float]] = {}
    sub = df[df["running_event_id"] == event_id]
    for r in sub.itertuples():
        mark = parse_performance(str(r.result_time), event_id)
        if math.isinf(mark) or mark <= 0:
            continue
        wa = getattr(r, pc, None)
        try:
            wa_f = float(wa) if wa is not None and not (isinstance(wa, float) and math.isnan(wa)) else 0.0
        except (TypeError, ValueError):
            wa_f = 0.0
        key = (int(r.athlete_id), int(r.season_year))
        cur = out.get(key)
        if cur is None or better(mark, cur[0], field):
            out[key] = (mark, wa_f)
    return out


def build_pair_rows(
    gender: str,
    group: str,
    indoor_id: int,
    outdoor_id: int,
    indoor_df: pd.DataFrame,
    outdoor_df: pd.DataFrame,
) -> list[AthletePair]:
    from_name = INDOOR_EVENTS[indoor_id]
    to_name = OUTDOOR_EVENTS[outdoor_id]
    in_best = best_mark_wa(indoor_df, indoor_id, gender)
    out_best = best_mark_wa(outdoor_df, outdoor_id, gender)

    # All outdoor event bests for profile
    outdoor_profiles: dict[tuple[int, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    for eid in GROUP_OUTDOOR_IDS.get(group, [outdoor_id]):
        if eid not in OUTDOOR_EVENTS:
            continue
        for key, (mark, wa) in best_mark_wa(outdoor_df, eid, gender).items():
            outdoor_profiles[key][OUTDOOR_EVENTS[eid]] = (mark, wa)

    rows: list[AthletePair] = []
    for key in sorted(set(in_best) & set(out_best)):
        fm, fw = in_best[key]
        tm, tw = out_best[key]
        owa = {ev: wa for ev, (_, wa) in outdoor_profiles[key].items() if wa > 0}
        omarks = {ev: mark for ev, (mark, _) in outdoor_profiles[key].items()}
        rows.append(
            AthletePair(
                athlete_id=key[0],
                season_year=key[1],
                gender=gender,
                event_group=group,
                from_event=from_name,
                to_event=to_name,
                from_mark=fm,
                to_mark=tm,
                from_wa=fw,
                to_wa=tw,
                outdoor_wa=owa,
                outdoor_marks=omarks,
            )
        )
    return rows


def feature_labels(row: AthletePair) -> dict[str, str]:
    """Compute routing labels for one athlete-pair row."""
    labels: dict[str, str] = {}

    # bal_spec from outdoor group WA
    was = [w for w in row.outdoor_wa.values() if w > 0]
    if len(was) >= 2:
        spread = max(was) - min(was)
        labels["bal_spec"] = "specialized" if spread >= SPREAD_THRESHOLD else "balanced"
    elif len(was) == 1:
        labels["bal_spec"] = "balanced"
    else:
        labels["bal_spec"] = "unknown"

    # best outdoor event
    if row.outdoor_wa:
        be = max(row.outdoor_wa.items(), key=lambda kv: kv[1])[0]
        labels["best_event"] = f"best_{be.replace('Outdoor ', '')}"
    else:
        labels["best_event"] = "unknown"

    # best_is_from: indoor from WA vs outdoor group WAs
    candidates = {row.from_event: row.from_wa}
    candidates.update(row.outdoor_wa)
    candidates = {k: v for k, v in candidates.items() if v > 0}
    if candidates:
        best = max(candidates.items(), key=lambda kv: kv[1])[0]
        labels["best_is_from"] = (
            "best_is_from" if best == row.from_event else "best_not_from"
        )
    else:
        labels["best_is_from"] = "unknown"

    if row.from_wa > 0 and row.to_wa > 0:
        labels["from_stronger_wa"] = (
            "from_stronger_wa" if row.from_wa >= row.to_wa else "to_stronger_wa"
        )
        gap = abs(row.from_wa - row.to_wa)
        labels["pair_wa_gap_50"] = "large_gap" if gap >= 50 else "small_gap"
    else:
        labels["from_stronger_wa"] = "unknown"
        labels["pair_wa_gap_50"] = "unknown"

    labels["bal_x_best_event"] = f"{labels['bal_spec']} ∩ {labels['best_event']}"
    return labels


def strategy_cv(
    rows: list[AthletePair],
    strategy: str,
    pooled_fit: Callable,
    pooled_pred: Callable,
) -> float:
    """CV median abs error when routing by strategy (refit within train cohorts)."""
    pairs = [(r.from_mark, r.to_mark) for r in rows]
    if strategy == "pooled":
        return cv_median_abs(pairs, pooled_fit, pooled_pred)

    rng = random.Random(CV_SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    fold = max(1, len(rows) // CV_FOLDS)
    actual, preds = [], []

    for f in range(CV_FOLDS):
        start = f * fold
        end = start + fold if f < CV_FOLDS - 1 else len(rows)
        test_i = set(idx[start:end])
        train_rows = [rows[i] for i in range(len(rows)) if i not in test_i]
        test_rows = [rows[i] for i in test_i]
        if len(train_rows) < 10:
            continue

        # Fit pooled fallback on train
        train_pairs = [(r.from_mark, r.to_mark) for r in train_rows]
        pooled_params = pooled_fit(train_pairs)

        # Cohort params
        cohorts: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in train_rows:
            lab = feature_labels(r).get(strategy, "unknown")
            if lab == "unknown":
                continue
            cohorts[lab].append((r.from_mark, r.to_mark))
        cohort_params = {
            lab: pooled_fit(pts)
            for lab, pts in cohorts.items()
            if len(pts) >= MIN_COHORT_N
        }

        for r in test_rows:
            lab = feature_labels(r).get(strategy, "unknown")
            params = cohort_params.get(lab, pooled_params)
            actual.append(r.to_mark)
            preds.append(pooled_pred(r.from_mark, params))

    if not actual:
        return float("nan")
    return statistics.median(abs(a - b) for a, b in zip(actual, preds))


def cohort_formulas(
    rows: list[AthletePair],
    strategy: str,
    from_ev: str,
    to_ev: str,
) -> list[tuple[str, int, dict, float]]:
    """Return (label, n, params, full_sample_medae)."""
    buckets: dict[str, list[AthletePair]] = defaultdict(list)
    for r in rows:
        lab = feature_labels(r).get(strategy, "unknown")
        if lab != "unknown":
            buckets[lab].append(r)
    out = []
    for lab, rs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < MIN_COHORT_N:
            continue
        pairs = [(r.from_mark, r.to_mark) for r in rs]
        params = choose_pooled_model(pairs)
        fit, pred = FIT_PRED[params["model"]]
        # full-sample medae with chosen family fit on all
        p = fit(pairs)
        p["model"] = params["model"]
        medae = statistics.median(abs(y - pred(x, p)) for x, y in pairs)
        out.append((lab, len(rs), p, medae))
    return out


def ratio_stats(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    ratios = sorted(y / x for x, y in pairs if x > 0)
    if not ratios:
        return float("nan"), float("nan"), float("nan")
    n = len(ratios)

    def pct(p: float) -> float:
        return ratios[min(n - 1, max(0, int(p * (n - 1))))]

    return statistics.median(ratios), pct(0.25), pct(0.75)


def format_pair_section(
    gender: str,
    group: str,
    rows: list[AthletePair],
    band_note: str,
) -> list[str]:
    if len(rows) < MIN_PAIR_N:
        return []
    from_ev = rows[0].from_event
    to_ev = rows[0].to_event
    pairs = [(r.from_mark, r.to_mark) for r in rows]
    _, _, r = linreg_pairs(pairs)
    pooled = choose_pooled_model(pairs)
    fit, pred = FIT_PRED[pooled["model"]]
    # refit for formula display
    params = fit(pairs)
    params["model"] = pooled["model"]
    formula = human_formula(params, from_ev, to_ev)
    full_medae = statistics.median(abs(y - pred(x, params)) for x, y in pairs)
    cv_pooled = pooled["cv_medae"]
    r_med, r_p25, r_p75 = ratio_stats(pairs)
    from_med = statistics.median(x for x, _ in pairs)
    to_med = statistics.median(y for _, y in pairs)
    unit = "m" if any(k in to_ev for k in ("Jump", "Shot Put")) else "s"

    lines = [
        f"{from_ev} -> {to_ev}  (n={len(rows)} athlete-seasons, r={r:.3f})",
        f"  Pooled model: {params['model']}",
        f"  Formula: {formula}",
        f"  CV median |error| (pooled routing): {cv_pooled:.3f}{unit}",
        f"  Full-sample median |error|: {full_medae:.3f}{unit}",
        f"  Ratio {to_ev}/{from_ev}: median {r_med:.3f} (IQR {r_p25:.3f}–{r_p75:.3f})",
        f"  Median marks: {from_ev} {format_mark(from_med, unit=='m')}, "
        f"{to_ev} {format_mark(to_med, unit=='m')}",
        "",
    ]

    # Feature CV
    feature_scores: list[tuple[str, float, float]] = []
    for strat in STRATEGIES:
        if strat == "pooled":
            continue
        cv = strategy_cv(rows, strat, fit, pred)
        if math.isnan(cv):
            continue
        delta = cv_pooled - cv  # positive = improvement
        if delta > EPS:
            feature_scores.append((strat, cv, delta))
    feature_scores.sort(key=lambda t: -t[2])

    lines.append("  Important features to know")
    lines.append("  --------------------------")
    if not feature_scores:
        lines.append(
            "  • No feature routing beat pooled by >0.01 on CV; use the pooled formula."
        )
        lines.append("")
        return lines

    rec, rec_cv, rec_d = feature_scores[0]
    lines.append(
        f"  • Recommended route: {rec} (CV {rec_cv:.3f}{unit}, Δ +{rec_d:.3f}{unit} vs pooled)."
    )
    lines.append(f"  • How to determine it: {FEATURE_HOW_TO[rec]}")
    lines.append("  • Other features that also beat pooled:")
    for strat, cv, delta in feature_scores:
        mark = " ← recommended" if strat == rec else ""
        lines.append(f"      - {strat}: CV {cv:.3f}{unit} (Δ +{delta:.3f}{unit}){mark}")
    lines.append("")

    # Detail for recommended + up to 1 alternate
    for strat, cv, delta in feature_scores[:2]:
        lines.append(f"  How {strat} alters the formula:")
        lines.append(f"  • Baseline (pooled): {formula}")
        lines.append(f"  • {FEATURE_HOW_TO[strat]}")
        cohorts = cohort_formulas(rows, strat, from_ev, to_ev)
        if not cohorts:
            lines.append("  • (insufficient cohort n for split formulas)")
        for lab, n, p, medae in cohorts:
            lines.append(
                f"      [{lab}] n={n}: {human_formula(p, from_ev, to_ev)}  "
                f"(full-sample MedAE {medae:.3f}{unit})"
            )
        lines.append("")

    return lines


def filter_band_by_indoor_wa(
    rows: list[AthletePair], lo: int, hi: int
) -> list[AthletePair]:
    """Keep athletes whose indoor from-event season-best WA is in [lo, hi)."""
    return [r for r in rows if lo <= r.from_wa < hi]


def write_report(
    path: Path, title: str, sections: dict[str, list[str]], preamble: list[str]
) -> None:
    lines = [title, "=" * len(title), ""] + preamble + [""]
    for header, body in sections.items():
        if not body:
            continue
        lines.append(header)
        lines.append("-" * len(header))
        lines.append("")
        lines.extend(body)
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def build_sections(
    all_pair_data: list[tuple[str, str, str, list[AthletePair]]],
    band: tuple[int, int] | None,
) -> dict[str, list[str]]:
    """If band is set, filter each pair's rows by indoor from-event WA."""
    sections: dict[str, list[str]] = {}
    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance", "Jumps", "Hurdles", "Throws"):
            header = f"{group} — {gender}"
            body: list[str] = []
            for g, grp, _, rows in all_pair_data:
                if g != gender or grp != group:
                    continue
                use = (
                    filter_band_by_indoor_wa(rows, band[0], band[1])
                    if band is not None
                    else rows
                )
                body.extend(format_pair_section(gender, group, use, "band" if band else "all"))
            if body:
                sections[header] = body
    return sections


def write_band_model_csv(
    all_pair_data: list[tuple[str, str, str, list[AthletePair]]],
    lo: int,
    hi: int,
) -> None:
    """Pooled formula table for the band (quick lookup CSV)."""
    rows_out = []
    for gender, group, pair_label, rows in all_pair_data:
        use = filter_band_by_indoor_wa(rows, lo, hi)
        if len(use) < MIN_PAIR_N:
            continue
        pairs = [(r.from_mark, r.to_mark) for r in use]
        _, _, r = linreg_pairs(pairs)
        pooled = choose_pooled_model(pairs)
        fit, pred = FIT_PRED[pooled["model"]]
        params = fit(pairs)
        params["model"] = pooled["model"]
        from_ev, to_ev = use[0].from_event, use[0].to_event
        full_medae = statistics.median(abs(y - pred(x, params)) for x, y in pairs)
        rows_out.append(
            {
                "band": f"{lo}-{hi}",
                "band_basis": "indoor_from_event_season_best_wa",
                "gender": gender,
                "event_group": group,
                "from_event": from_ev,
                "to_event": to_ev,
                "n_athletes": len(use),
                "r": round(r, 4),
                "pooled_model": params["model"],
                "cv_medae": round(pooled["cv_medae"], 4),
                "full_medae": round(full_medae, 4),
                "formula": human_formula(params, from_ev, to_ev),
                "indoor_wa_median": round(
                    statistics.median(r.from_wa for r in use if r.from_wa > 0), 1
                ),
            }
        )
    path = BANDS_DIR / f"indoor_to_outdoor_models_band_{lo}_{hi}.csv"
    pd.DataFrame(rows_out).to_csv(path, index=False)
    print(f"Wrote {path}")


def run() -> None:
    frame_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def frames(group: str, gender: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = (group, gender)
        if key not in frame_cache:
            print(f"[Feature-aware] Loading {gender} {group}…")
            indoor_ids = {
                p[1] for p in PAIRS if p[0] == group and (p[3] is None or p[3] == gender)
            }
            inn = load_indoor_for_ids(group, gender, indoor_ids)
            out = load_expanded_outdoor(group, gender)
            frame_cache[key] = (inn, out)
        return frame_cache[key]

    all_pair_data: list[tuple[str, str, str, list[AthletePair]]] = []

    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance", "Jumps", "Hurdles", "Throws"):
            pair_defs = [
                p for p in PAIRS if p[0] == group and (p[3] is None or p[3] == gender)
            ]
            if not pair_defs:
                continue
            inn, out = frames(group, gender)
            for _, iid, oid, gfilt in pair_defs:
                if gfilt and gfilt != gender:
                    continue
                rows = build_pair_rows(gender, group, iid, oid, inn, out)
                if len(rows) < 5:
                    continue
                all_pair_data.append(
                    (
                        gender,
                        group,
                        f"{INDOOR_EVENTS[iid]} → {OUTDOOR_EVENTS[oid]}",
                        rows,
                    )
                )
                print(
                    f"  {gender} {INDOOR_EVENTS[iid]} → {OUTDOOR_EVENTS[oid]}: n={len(rows)}"
                )

    preamble_all = [
        "Method:",
        "  • Data: indoor + outdoor CSVs 2024–2026; same athlete × year.",
        "  • Pairs: Indoor→Outdoor event translations (sprints, distance, jumps,",
        "    hurdles, throws) as in Indoor_to_Outdoor_Predictions.",
        "  • Pooled formula: lowest-CV winner among linear_ols, median_ratio,",
        "    log_linear, robust_trimmed_linear (seed=42, folds=5).",
        "  • Feature formulas: re-fit within feature cohorts (n≥12); pair needs n≥20.",
        "  • Important features: strategies that beat pooled CV by >0.01 (s or m).",
        "",
        "Use:",
        "  1. Start from the pooled formula for the indoor→outdoor pair.",
        "  2. If important features are known, switch to the matching cohort formula",
        "     (prefer the pair's recommended strategy first).",
        "  3. Features bal_spec / best_event use outdoor season WA in the event group;",
        "     best_is_from uses indoor-from WA vs that outdoor profile.",
        "  4. from_stronger_wa / pair_wa_gap_50 use outdoor to-event WA (available when",
        "     validating; for pure forecasting prefer bal_spec / best_is_from).",
        "",
    ]

    write_report(
        OUT / "feature_aware_indoor_to_outdoor_findings.txt",
        "Feature-Aware Indoor → Outdoor Time Models",
        build_sections(all_pair_data, band=None),
        preamble_all,
    )

    for lo, hi in TARGET_BANDS:
        title = (
            f"Feature-Aware Indoor → Outdoor Time Models — "
            f"WA Band {lo}-{hi} (Width {hi - lo})"
        )
        preamble_band = [
            "Method:",
            "  • Same indoor→outdoor pairs and feature routing as",
            "    feature_aware_indoor_to_outdoor_findings.txt.",
            f"  • Band membership: indoor from-event season-best World Athletics score",
            f"    ∈ [{lo}, {hi}). The WA value is taken from the athlete's best",
            f"    (fastest / farthest) indoor mark in the from-event that season.",
            "  • Pooled + feature cohort formulas; CV seed=42, folds=5; cohort n≥12;",
            "    pair needs n≥20 after banding.",
            "",
            "Use:",
            f"  1. Compute the athlete's indoor from-event season PB and its WA score.",
            f"  2. If that WA is in [{lo}, {hi}), use this band's formulas.",
            "  3. Start from the pooled formula; switch to feature-cohort formulas when labeled.",
            "  4. For pure forecasting (outdoor unknown), prefer bal_spec / best_is_from",
            "     over from_stronger_wa / pair_wa_gap_50.",
            "",
        ]
        sections = build_sections(all_pair_data, band=(lo, hi))
        write_report(
            BANDS_DIR / f"feature_aware_indoor_to_outdoor_band_{lo}_{hi}.txt",
            title,
            sections,
            preamble_band,
        )
        # Also mirror at output/ root for the names users may look for
        write_report(
            OUT / f"feature_aware_indoor_to_outdoor_band_{lo}_{hi}.txt",
            title,
            sections,
            preamble_band,
        )
        write_band_model_csv(all_pair_data, lo, hi)

    md = [
        "# Feature-Aware Indoor → Outdoor Predictions",
        "",
        "Reports in the style of `feature_aware_indoor_to_outdoor_findings.txt`.",
        "",
        "## Unbanded",
        "",
        "- [`feature_aware_indoor_to_outdoor_findings.txt`](feature_aware_indoor_to_outdoor_findings.txt)",
        "",
        "## By indoor from-event WA band",
        "",
        "Band = WA of the athlete's **indoor season-best** (fastest/best mark) in the from-event.",
        "",
        "| Band | Report | Models CSV |",
        "|------|--------|------------|",
    ]
    for lo, hi in TARGET_BANDS:
        md.append(
            f"| {lo}–{hi} | "
            f"[`feature_aware_indoor_to_outdoor_band_{lo}_{hi}.txt`]"
            f"(feature_aware_indoor_to_outdoor_band_{lo}_{hi}.txt) | "
            f"[`by_indoor_wa_band/indoor_to_outdoor_models_band_{lo}_{hi}.csv`]"
            f"(by_indoor_wa_band/indoor_to_outdoor_models_band_{lo}_{hi}.csv) |"
        )
    md += [
        "",
        "```bash",
        "python time_models/Indoor_to_Outdoor_Predictions/main.py feature",
        "```",
        "",
    ]
    (OUT / "Feature_Aware_Summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT / 'Feature_Aware_Summary.md'}")


if __name__ == "__main__":
    run()
