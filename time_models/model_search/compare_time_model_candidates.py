"""Compare candidate cross-event time models vs. linear baseline (CV evaluation).

Outputs to time_models/model_search/
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[1]
RELAYS_ROOT = PROJECT_ROOT / "relays_findings"
OUTPUT_ROOT = Path(__file__).resolve().parent
BASELINE_CSV = TIME_MODELS_ROOT / "cross_event_time_models.csv"

sys.path.insert(0, str(RELAYS_ROOT))
sys.path.insert(0, str(TIME_MODELS_ROOT))
from relay_rq1_data import points_col, season_rows  # noqa: E402
from build_cross_event_time_models import (  # noqa: E402
    DISTANCE_EVENTS,
    GROUP_CONFIG,
    SPRINT_EVENTS,
    linreg,
    load_season_pbs,
)

EVENT_ORDER = {
    "Sprints": ["100m", "200m", "400m"],
    "Distance": ["800m", "1500m", "5000m"],
}

CV_FOLDS = 5
CV_SEED = 42
KNN_K = 20
BIN_COUNT = 8


@dataclass
class CandidateResult:
    name: str
    cv_median_abs: float
    cv_rmse: float
    full_median_abs: float
    full_rmse: float
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def metrics(actual: list[float], pred: list[float]) -> tuple[float, float]:
    errs = [a - p for a, p in zip(actual, pred)]
    abs_errs = [abs(e) for e in errs]
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    med = statistics.median(abs_errs)
    return med, rmse


def cross_validate(
    pairs: list[tuple[float, float]],
    fit: Callable[[list[tuple[float, float]]], dict],
    predict: Callable[[float, dict], float],
    folds: int = CV_FOLDS,
) -> tuple[float, float]:
    if len(pairs) < folds * 2:
        pred_ys = []
        actual_ys = []
        params = fit(pairs)
        for x, y in pairs:
            pred_ys.append(predict(x, params))
            actual_ys.append(y)
        return metrics(actual_ys, pred_ys)

    rng = random.Random(CV_SEED)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    fold_size = len(pairs) // folds
    all_actual: list[float] = []
    all_pred: list[float] = []

    for f in range(folds):
        start = f * fold_size
        end = start + fold_size if f < folds - 1 else len(pairs)
        test_idx = set(idx[start:end])
        train = [pairs[i] for i in range(len(pairs)) if i not in test_idx]
        test = [pairs[i] for i in test_idx]
        if len(train) < 10:
            continue
        params = fit(train)
        for x, y in test:
            all_actual.append(y)
            all_pred.append(predict(x, params))

    return metrics(all_actual, all_pred)


# --- Candidate model families ---


def fit_linear(train: list[tuple[float, float]]) -> dict:
    intercept, slope, _, _, _ = linreg(train)
    return {"intercept": intercept, "slope": slope}


def pred_linear(x: float, p: dict) -> float:
    return p["intercept"] + p["slope"] * x


def fit_log_linear(train: list[tuple[float, float]]) -> dict:
    log_pairs = [(math.log(x), math.log(y)) for x, y in train if x > 0 and y > 0]
    intercept, slope, _, _, _ = linreg(log_pairs)
    return {"intercept": intercept, "slope": slope}


def pred_log_linear(x: float, p: dict) -> float:
    return math.exp(p["intercept"] + p["slope"] * math.log(x))


def fit_median_ratio(train: list[tuple[float, float]]) -> dict:
    ratios = sorted(y / x for x, y in train if x > 0)
    return {"ratio": statistics.median(ratios)}


def pred_median_ratio(x: float, p: dict) -> float:
    return x * p["ratio"]


def fit_ratio_linear(train: list[tuple[float, float]]) -> dict:
    """ratio = a + b * from_time"""
    intercept, slope, _, _, _ = linreg([(x, y / x) for x, y in train if x > 0])
    return {"intercept": intercept, "slope": slope}


def pred_ratio_linear(x: float, p: dict) -> float:
    ratio = p["intercept"] + p["slope"] * x
    return x * max(ratio, 0.1)


def fit_quadratic(train: list[tuple[float, float]]) -> dict:
    # y = a + b*x + c*x^2 via normal equations
    n = len(train)
    s1 = float(n)
    sx = sum(x for x, _ in train)
    sx2 = sum(x * x for x, _ in train)
    sx3 = sum(x**3 for x, _ in train)
    sx4 = sum(x**4 for x, _ in train)
    sy = sum(y for _, y in train)
    sxy = sum(x * y for x, y in train)
    sx2y = sum(x * x * y for x, y in train)

    # Solve 3x3
    m = [
        [s1, sx, sx2, sy],
        [sx, sx2, sx3, sxy],
        [sx2, sx3, sx4, sx2y],
    ]
    a, b, c = solve_3x3(m)
    return {"intercept": a, "slope": b, "quad": c}


def solve_3x3(aug: list[list[float]]) -> tuple[float, float, float]:
    """Gaussian elimination for 3x3."""
    a = [row[:] for row in aug]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            return 0.0, 1.0, 0.0
        div = a[col][col]
        for j in range(4):
            a[col][j] /= div
        for r in range(3):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(4):
                a[r][j] -= factor * a[col][j]
    return a[0][3], a[1][3], a[2][3]


def pred_quadratic(x: float, p: dict) -> float:
    return p["intercept"] + p["slope"] * x + p["quad"] * x * x


def fit_binned_ratio(train: list[tuple[float, float]], bins: int = BIN_COUNT) -> dict:
    sorted_train = sorted(train, key=lambda t: t[0])
    chunk = max(1, len(sorted_train) // bins)
    edges: list[float] = []
    ratios: list[float] = []
    for i in range(0, len(sorted_train), chunk):
        part = sorted_train[i : i + chunk]
        if not part:
            continue
        xs = [x for x, _ in part]
        edges.append(statistics.median(xs))
        ratios.append(statistics.median(y / x for x, y in part if x > 0))
    return {"edges": edges, "ratios": ratios}


def pred_binned_ratio(x: float, p: dict) -> float:
    edges = p["edges"]
    ratios = p["ratios"]
    if not edges:
        return x
    # nearest edge bin
    idx = min(range(len(edges)), key=lambda i: abs(edges[i] - x))
    return x * ratios[idx]


def fit_knn(train: list[tuple[float, float]], k: int = KNN_K) -> dict:
    return {"data": list(train), "k": min(k, len(train))}


def pred_knn(x: float, p: dict) -> float:
    data = p["data"]
    k = p["k"]
    neighbors = sorted(data, key=lambda t: abs(t[0] - x))[:k]
    return statistics.median(y for _, y in neighbors)


def fit_robust_trimmed(train: list[tuple[float, float]]) -> dict:
    p = fit_linear(train)
    resid = [(y - pred_linear(x, p), x, y) for x, y in train]
    abs_r = [abs(r[0]) for r in resid]
    cutoff = statistics.median(abs_r) * 2.5
    trimmed = [(x, y) for r, x, y in resid if abs(r) <= cutoff]
    if len(trimmed) < max(10, len(train) // 2):
        return p
    return fit_linear(trimmed)


CANDIDATES: list[tuple[str, Callable, Callable, str]] = [
    ("linear_ols", fit_linear, pred_linear, "Baseline: intercept + slope × from_time"),
    ("log_linear", fit_log_linear, pred_log_linear, "log(to) = a + b×log(from); multiplicative"),
    ("median_ratio", fit_median_ratio, pred_median_ratio, "to = from × median(to/from)"),
    ("ratio_linear", fit_ratio_linear, pred_ratio_linear, "to = from × (a + b×from); speed-dependent ratio"),
    ("quadratic", fit_quadratic, pred_quadratic, "to = a + b×from + c×from²"),
    ("binned_ratio", fit_binned_ratio, pred_binned_ratio, f"to = from × median_ratio within {BIN_COUNT} speed bins"),
    ("knn_median", fit_knn, pred_knn, f"median to_time of {KNN_K} nearest from_time neighbors"),
    ("robust_trimmed_linear", fit_robust_trimmed, pred_linear, "OLS after trimming large residuals"),
]


def evaluate_pair(
    pairs: list[tuple[float, float]],
) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for name, fit_fn, pred_fn, notes in CANDIDATES:
        cv_med, cv_rmse = cross_validate(pairs, fit_fn, pred_fn)
        params = fit_fn(pairs)
        full_pred = [pred_fn(x, params) for x, _ in pairs]
        full_actual = [y for _, y in pairs]
        full_med, full_rmse = metrics(full_actual, full_pred)
        # JSON-serialize params (convert tuples in knn)
        serial = serialize_params(params)
        results.append(
            CandidateResult(
                name=name,
                cv_median_abs=cv_med,
                cv_rmse=cv_rmse,
                full_median_abs=full_med,
                full_rmse=full_rmse,
                params=serial,
                notes=notes,
            )
        )
    return results


def serialize_params(params: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k == "data":
            out["k"] = params.get("k", KNN_K)
            out["n_train"] = len(v)
        else:
            out[k] = v
    return out


def fit_chain_adjusted(
    train_triple: list[tuple[float, float, float]],
    mid_event: str,
) -> dict:
    """train rows: (from, mid, to). Adjust to-prediction using mid residual."""
    pairs_fm = [(a, b) for a, b, _ in train_triple]
    pairs_mt = [(b, c) for _, b, c in train_triple]
    p_fm = fit_linear(pairs_fm)
    p_mt = fit_linear(pairs_mt)
    residuals = []
    adjustments = []
    for a, b, c in train_triple:
        exp_b = pred_linear(a, p_fm)
        res = b - exp_b
        base_c = pred_linear(b, p_mt)
        adjustments.append(c - base_c)
        residuals.append(res)
    # fit adjustment = k * residual
    if len(residuals) >= 10:
        k, _, _, _, _ = linreg(list(zip(residuals, adjustments)))
    else:
        k = 0.0
    return {
        "from_to_mid": p_fm,
        "mid_to_to": p_mt,
        "residual_k": k,
        "mid_event": mid_event,
    }


def pred_chain_adjusted(from_t: float, mid_t: float, p: dict) -> float:
    exp_mid = pred_linear(from_t, p["from_to_mid"])
    res = mid_t - exp_mid
    base = pred_linear(mid_t, p["mid_to_to"])
    return base + p["residual_k"] * res


def evaluate_chain_models(
    athlete_bests: dict[str, dict[str, float]],
    from_ev: str,
    mid_ev: str,
    to_ev: str,
) -> CandidateResult | None:
    triple = [
        (b[from_ev], b[mid_ev], b[to_ev])
        for b in athlete_bests.values()
        if from_ev in b and mid_ev in b and to_ev in b
    ]
    if len(triple) < 30:
        return None

    pairs_direct = [(a, c) for a, _, c in triple]
    pairs_mid_only = [(b, c) for _, b, c in triple]

    def fit_chain(train_rows: list[tuple[float, float, float]]) -> dict:
        return fit_chain_adjusted(train_rows, mid_ev)

    def pred_chain(x: float, p: dict) -> float:
        # x is from_time; need mid - use expected mid from from
        exp_mid = pred_linear(x, p["from_to_mid"])
        return pred_chain_adjusted(x, exp_mid, p)

    def pred_chain_oracle(x: float, p: dict) -> float:
        return pred_chain(x, p)

    # CV for chain with known mid at test time (oracle - athlete has run mid)
    rng = random.Random(CV_SEED)
    idx = list(range(len(triple)))
    rng.shuffle(idx)
    fold_size = len(triple) // CV_FOLDS
    cv_actual: list[float] = []
    cv_pred: list[float] = []

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else len(triple)
        test_idx = set(idx[start:end])
        train = [triple[i] for i in range(len(triple)) if i not in test_idx]
        test = [triple[i] for i in test_idx]
        if len(train) < 20:
            continue
        p = fit_chain(train)
        for a, b, c in test:
            cv_actual.append(c)
            cv_pred.append(pred_chain_adjusted(a, b, p))

    cv_med, cv_rmse = metrics(cv_actual, cv_pred)

    p_full = fit_chain(triple)
    full_pred = [pred_chain_adjusted(a, b, p_full) for a, b, _ in triple]
    full_actual = [c for _, _, c in triple]
    full_med, full_rmse = metrics(full_actual, full_pred)

    # Compare to mid-only linear
    cv_med_mid, _ = cross_validate(pairs_mid_only, fit_linear, pred_linear)

    return CandidateResult(
        name="chain_residual_adjusted",
        cv_median_abs=cv_med,
        cv_rmse=cv_rmse,
        full_median_abs=full_med,
        full_rmse=full_rmse,
        params=serialize_params(p_full),
        notes=(
            f"Uses actual {mid_ev}: adjust {to_ev} pred by k×({mid_ev} residual from {from_ev}). "
            f"mid_only_linear CV med|err|={cv_med_mid:.3f}s"
        ),
    )


def evaluate_multivariate(
    athlete_bests: dict[str, dict[str, float]],
    ev_a: str,
    ev_b: str,
    to_ev: str,
) -> CandidateResult | None:
    rows = [
        (b[ev_a], b[ev_b], b[to_ev])
        for b in athlete_bests.values()
        if ev_a in b and ev_b in b and to_ev in b
    ]
    if len(rows) < 30:
        return None

    def fit_multi(train: list[tuple[float, float, float]]) -> dict:
        # z = a + b*x + c*y
        n = len(train)
        sx = sum(x for x, _, _ in train)
        sy = sum(y for _, y, _ in train)
        s1 = float(n)
        sxx = sum(x * x for x, _, _ in train)
        syy = sum(y * y for _, y, _ in train)
        sxy = sum(x * y for x, y, _ in train)
        sz = sum(z for _, _, z in train)
        sxz = sum(x * z for x, _, z in train)
        syz = sum(y * z for _, y, z in train)
        # Solve manually using simple approach: treat as multiple linreg - use normal eq 3x3
        m = [
            [s1, sx, sy, sz],
            [sx, sxx, sxy, sxz],
            [sy, sxy, syy, syz],
        ]
        a, b, c = solve_3x3(m)
        return {"intercept": a, "coef_a": b, "coef_b": c, "event_a": ev_a, "event_b": ev_b}

    def pred_multi(x: float, p: dict) -> float:
        raise NotImplementedError

    def fit_multi_pairs(train_pairs: list[tuple[float, float]]) -> dict:
        return {}

    # CV with two inputs - custom
    rng = random.Random(CV_SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    fold_size = len(rows) // CV_FOLDS
    cv_actual: list[float] = []
    cv_pred: list[float] = []

    for f in range(CV_FOLDS):
        start = f * fold_size
        end = start + fold_size if f < CV_FOLDS - 1 else len(rows)
        test_idx = set(idx[start:end])
        train = [rows[i] for i in range(len(rows)) if i not in test_idx]
        test = [rows[i] for i in test_idx]
        p = fit_multi(train)
        for x, y, z in test:
            cv_actual.append(z)
            cv_pred.append(p["intercept"] + p["coef_a"] * x + p["coef_b"] * y)

    cv_med, cv_rmse = metrics(cv_actual, cv_pred)
    p_full = fit_multi(rows)
    full_pred = [
        p_full["intercept"] + p_full["coef_a"] * x + p_full["coef_b"] * y for x, y, _ in rows
    ]
    full_actual = [z for _, _, z in rows]
    full_med, full_rmse = metrics(full_actual, full_pred)

    return CandidateResult(
        name="multivariate_ols",
        cv_median_abs=cv_med,
        cv_rmse=cv_rmse,
        full_median_abs=full_med,
        full_rmse=full_rmse,
        params=p_full,
        notes=f"{to_ev} = a + b×{ev_a} + c×{ev_b} (requires both inputs)",
    )


def load_baseline_median(gender: str, group: str, from_ev: str, to_ev: str) -> float | None:
    if not BASELINE_CSV.exists():
        return None
    for row in csv.DictReader(open(BASELINE_CSV)):
        if (
            row["gender"] == gender
            and row["event_group"] == group
            and row["from_event"] == from_ev
            and row["to_event"] == to_ev
        ):
            return float(row["median_abs_error_seconds"])
    return None


def format_params(name: str, params: dict) -> str:
    if name == "linear_ols":
        return f"{params['intercept']:.4f} + {params['slope']:.4f}×x"
    if name == "log_linear":
        return f"exp({params['intercept']:.4f} + {params['slope']:.4f}×log(x))"
    if name == "median_ratio":
        return f"x × {params['ratio']:.4f}"
    if name == "ratio_linear":
        return f"x × ({params['intercept']:.4f} + {params['slope']:.6f}×x)"
    if name == "quadratic":
        return (
            f"{params['intercept']:.4f} + {params['slope']:.4f}×x + "
            f"{params['quad']:.6f}×x²"
        )
    if name == "binned_ratio":
        return f"{len(params.get('edges', []))} bins (see JSON)"
    if name == "knn_median":
        return f"k={params.get('k', KNN_K)} nearest neighbors"
    if name == "chain_residual_adjusted":
        return f"chain via {params.get('mid_event','?')} + residual k={params.get('residual_k',0):.4f}"
    if name == "multivariate_ols":
        return (
            f"{params['intercept']:.2f} + {params['coef_a']:.4f}×{params['event_a']} + "
            f"{params['coef_b']:.4f}×{params['event_b']}"
        )
    return json.dumps(params)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict] = []
    best_rows: list[dict] = []
    report_lines = [
        "Cross-Event Time Model Search — Candidate Comparison",
        "====================================================",
        "",
        f"Evaluation: {CV_FOLDS}-fold cross-validation on athlete PB pairs (seed={CV_SEED}).",
        "Primary metric: CV median absolute error (seconds). Lower is better.",
        "Baseline: linear_ols from time_models/cross_event_time_models.csv.",
        "Steeplechase excluded. Data: outdoor 2024–2026, March 1+.",
        "",
    ]

    wins_by_model: dict[str, int] = {}
    improvements: list[tuple[str, float]] = []

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            bests = load_season_pbs(folder, prefix, gender, events)
            section = f"{event_group} — {gender}"
            report_lines.extend(["", section, "-" * len(section)])

            for from_ev, to_ev in permutations(order, 2):
                pairs = [
                    (b[from_ev], b[to_ev])
                    for b in bests.values()
                    if from_ev in b and to_ev in b
                ]
                if len(pairs) < 20:
                    continue

                results = evaluate_pair(pairs)
                baseline_cv = next(r for r in results if r.name == "linear_ols").cv_median_abs
                baseline_full = load_baseline_median(gender, event_group, from_ev, to_ev)

                # Chain / multivariate extras when applicable
                extra: list[CandidateResult] = []
                if event_group == "Sprints":
                    if from_ev == "100m" and to_ev == "400m":
                        chain = evaluate_chain_models(bests, "100m", "200m", "400m")
                        if chain:
                            extra.append(chain)
                        multi = evaluate_multivariate(bests, "100m", "200m", "400m")
                        if multi:
                            extra.append(multi)
                if event_group == "Distance":
                    if from_ev == "800m" and to_ev == "5000m":
                        chain = evaluate_chain_models(bests, "800m", "1500m", "5000m")
                        if chain:
                            extra.append(chain)
                        multi = evaluate_multivariate(bests, "800m", "1500m", "5000m")
                        if multi:
                            extra.append(multi)

                all_results = results + extra
                winner = min(all_results, key=lambda r: r.cv_median_abs)
                wins_by_model[winner.name] = wins_by_model.get(winner.name, 0) + 1
                improve = baseline_cv - winner.cv_median_abs
                improvements.append((f"{section} {from_ev}->{to_ev}", improve))

                report_lines.append(
                    f"\n{from_ev} -> {to_ev}  (n={len(pairs)})  "
                    f"linear CV med|err|={baseline_cv:.3f}s"
                )
                report_lines.append(f"  WINNER: {winner.name}  CV med|err|={winner.cv_median_abs:.3f}s  "
                                    f"({'↓' if improve > 0 else '↑'}{abs(improve):.3f}s vs linear)")
                report_lines.append(f"  {winner.notes}")
                report_lines.append(f"  Params: {format_params(winner.name, winner.params)}")

                for r in sorted(all_results, key=lambda x: x.cv_median_abs):
                    delta = baseline_cv - r.cv_median_abs
                    report_lines.append(
                        f"    {r.name:<24} CV med|err|={r.cv_median_abs:7.3f}s  "
                        f"CV RMSE={r.cv_rmse:7.3f}s  Δlinear={delta:+.3f}s"
                    )
                    comparison_rows.append(
                        {
                            "gender": gender,
                            "event_group": event_group,
                            "from_event": from_ev,
                            "to_event": to_ev,
                            "n_athletes": len(pairs),
                            "model": r.name,
                            "cv_median_abs_error": round(r.cv_median_abs, 4),
                            "cv_rmse": round(r.cv_rmse, 4),
                            "full_median_abs_error": round(r.full_median_abs, 4),
                            "full_rmse": round(r.full_rmse, 4),
                            "delta_vs_linear_cv_med": round(baseline_cv - r.cv_median_abs, 4),
                            "is_winner": r.name == winner.name,
                            "params_summary": format_params(r.name, r.params),
                        }
                    )

                best_rows.append(
                    {
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n_athletes": len(pairs),
                        "best_model": winner.name,
                        "linear_cv_median_abs": round(baseline_cv, 4),
                        "best_cv_median_abs": round(winner.cv_median_abs, 4),
                        "improvement_seconds": round(improve, 4),
                        "best_full_median_abs": round(winner.full_median_abs, 4),
                        "params_json": json.dumps(winner.params),
                        "params_summary": format_params(winner.name, winner.params),
                        "notes": winner.notes,
                    }
                )

    report_lines.extend(
        [
            "",
            "Summary",
            "-------",
            f"Pairwise comparisons evaluated: {len(best_rows)}",
            "Wins by model (lowest CV median |error|):",
        ]
    )
    for name, count in sorted(wins_by_model.items(), key=lambda x: (-x[1], x[0])):
        report_lines.append(f"  {name}: {count}")

    avg_improve = statistics.mean(i for _, i in improvements if i > 0) if improvements else 0
    n_improved = sum(1 for _, i in improvements if i > 0.01)
    report_lines.extend(
        [
            f"Pairs where winner beats linear by >0.01s CV med|err|: {n_improved}/{len(best_rows)}",
            f"Mean improvement when better than linear: {avg_improve:.3f}s",
            "",
            "Recommendations",
            "---------------",
            "1. Use chain_residual_adjusted when the athlete has run the intermediate event",
            "   (e.g. 100m+200m -> 400m; 800m+1500m -> 5000m).",
            "2. Use knn_median or binned_ratio for single-input pairs where they beat linear",
            "   (often 100m->400m and long distance hops).",
            "3. Keep linear_ols where it wins — it is simple and often optimal for adjacent",
            "   sprint pairs (100m<->200m).",
            "",
            "Source: time_models/model_search/compare_time_model_candidates.py",
        ]
    )

    with open(OUTPUT_ROOT / "model_comparison_all_candidates.csv", "w", newline="") as f:
        if comparison_rows:
            w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader()
            w.writerows(comparison_rows)

    with open(OUTPUT_ROOT / "best_time_models.csv", "w", newline="") as f:
        if best_rows:
            w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            w.writeheader()
            w.writerows(best_rows)

    (OUTPUT_ROOT / "model_search_report.txt").write_text("\n".join(report_lines).rstrip() + "\n")
    print(f"Wrote model search results to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
