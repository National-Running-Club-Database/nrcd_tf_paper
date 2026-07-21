"""Write time-model formula files for selected WA point bands (width 200).

Output format matches time_models_higher_races_formulas.txt.
"""

from __future__ import annotations

import json
import statistics
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(OUTPUT_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    EVENT_ORDER,
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
from analyze_point_bands_time_models import (  # noqa: E402
    MIN_PAIR_N,
    in_band,
    load_athlete_season_records,
    run_search_for_bests,
)

# (lo, hi) inclusive-lo exclusive-hi; user-facing labels use these endpoints
TARGET_BANDS = (
    (850, 1050),
    (800, 1000),
    (750, 950),
)

PRED = {
    "linear_ols": (fit_linear, pred_linear),
    "log_linear": (fit_log_linear, pred_log_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "binned_ratio": (None, pred_binned_ratio),
}


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
        return (
            f"{to_ev} = {from_ev} × ratio_bin({from_ev})  "
            f"[nearest speed bin; see binned table below]"
        )
    if model == "multivariate_ols":
        return (
            f"{to_ev} = {params['intercept']:.2f} + {params['coef_a']:.3f}×{params['event_a']} "
            f"+ {params['coef_b']:.3f}×{params['event_b']}"
        )
    if model == "chain_residual_adjusted":
        mid = params.get("mid_event", "?")
        p_fm = params["from_to_mid"]
        p_mt = params["mid_to_to"]
        return (
            f"Step 1: {mid}_est = {p_fm['intercept']:.2f} + {p_fm['slope']:.3f}×{from_ev}\n"
            f"  Step 2 (with actual {mid}): {to_ev} = {p_mt['intercept']:.2f} + "
            f"{p_mt['slope']:.3f}×{mid}"
        )
    return format_params(model, params)


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
        "knn_median": "knn_median",
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


def pair_detail(
    bests: dict[str, dict[str, float]],
    from_ev: str,
    to_ev: str,
    row: dict,
) -> dict[str, Any]:
    pairs = [
        (b[from_ev], b[to_ev]) for b in bests.values() if from_ev in b and to_ev in b
    ]
    model = row["best_model"]
    params = json.loads(row["params_json"])
    _, _, r, _, _ = linreg(pairs)
    r_med, r_p25, r_p75 = ratio_quartiles(pairs)
    full_med = None
    if model in PRED and PRED[model][1] is not None:
        pred_fn = PRED[model][1]
        preds = [pred_fn(x, params) for x, _ in pairs]
        actual = [y for _, y in pairs]
        full_med, _ = metrics(actual, preds)
    return {
        "n": len(pairs),
        "r": r,
        "formula": human_formula(model, params, from_ev, to_ev),
        "model": model,
        "cv_med": float(row["best_cv_median_abs"]),
        "full_med": full_med,
        "r_med": r_med,
        "r_p25": r_p25,
        "r_p75": r_p75,
        "from_med": statistics.median(p[0] for p in pairs),
        "to_med": statistics.median(p[1] for p in pairs),
        "params": params,
        "binned_lines": binned_table(params, from_ev, to_ev),
    }


def write_band_formulas(
    *,
    lo: int,
    hi: int,
    profiles_by_gg: dict[tuple[str, str], dict[str, dict]],
    path: Path,
) -> list[dict]:
    label = f"{lo}-{hi}"
    best_rows: list[dict] = []
    bests_by_gg: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    n_as = 0
    for (gender, event_group), profiles in profiles_by_gg.items():
        included = {
            k: p["event_times"]
            for k, p in profiles.items()
            if in_band(p["result_was"], lo, hi)
        }
        bests_by_gg[(gender, event_group)] = included
        n_as += len(included)
        best_rows.extend(run_search_for_bests(included, gender, event_group))

    mean_cv = (
        statistics.mean(r["best_cv_median_abs"] for r in best_rows) if best_rows else None
    )
    title = f"WA Point Band {label} (Width 200)"
    lines = [
        f"Cross-Event Time Models — {title} (Sprints & Distance)",
        "=" * (len(title) + 52),
        "",
        "Method:",
        "  • Data: outdoor CSVs, 2024–2026, results on or after March 1.",
        f"  • Population: athlete-seasons with ≥1 individual (non-relay, non-steeple)",
        f"    result in World Athletics points [{lo}, {hi}).",
        "  • Athlete-season PB: fastest valid mark per event within that season (WA > 0).",
        "  • Model selection: 5-fold cross-validation across 8+ candidate model families;",
        "    the formula below is the lowest-CV-median-|error| winner for each pair.",
        "  • Relays and 3000m Steeplechase excluded. Sprints and Distance only.",
        f"  • Minimum n = {MIN_PAIR_N} athlete-season pairs required to report a model.",
        f"  • CV settings: seed={CV_SEED}, folds={CV_FOLDS}.",
        "",
        "Use:",
        f"  • Apply when the athlete has at least one season result in WA band {label}.",
        "  • Predict time in the unknown event from known PB(s), then convert predicted",
        "    time to World Athletics points in the TARGET event table.",
        "  • Treat CV median |error| as the typical ± uncertainty band (cross-validated).",
        "  • Do not compare WA across events directly.",
        "",
        "Coverage note:",
        f"  {len(best_rows)} of 24 event pairs met n ≥ {MIN_PAIR_N} in band {label}"
        f" ({n_as} athlete-seasons).",
    ]
    if mean_cv is not None:
        lines.append(f"  Mean CV median |error| across reported pairs: {mean_cv:.3f}s.")
    lines.append("")

    reported_keys: set[tuple[str, str, str, str]] = set()
    summary_rows: list[tuple[str, str, str, str, dict]] = []

    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            section_rows = [
                r
                for r in best_rows
                if r["gender"] == gender and r["event_group"] == group
            ]
            section = f"{group} — {gender}"
            lines.extend(["", section, "-" * len(section), ""])
            if not section_rows:
                lines.append(
                    f"No pairs met the minimum sample size in WA band {label}."
                )
                lines.append("")
                continue
            bests = bests_by_gg[(gender, group)]
            for row in sorted(
                section_rows, key=lambda r: (r["from_event"], r["to_event"])
            ):
                from_ev, to_ev = row["from_event"], row["to_event"]
                reported_keys.add((gender, group, from_ev, to_ev))
                stats = pair_detail(bests, from_ev, to_ev, row)
                summary_rows.append((gender, group, from_ev, to_ev, stats))
                lines.append(
                    f"{from_ev} -> {to_ev}  (n={stats['n']} athlete-seasons, r={stats['r']:.3f})"
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

    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            missing = []
            for from_ev, to_ev in permutations(EVENT_ORDER[group], 2):
                if (gender, group, from_ev, to_ev) not in reported_keys:
                    missing.append(f"{from_ev} -> {to_ev}")
            if missing:
                lines.extend(
                    [
                        f"Not reported for {group} — {gender} "
                        f"(n < {MIN_PAIR_N} athlete-season pairs):",
                        "  " + ", ".join(missing),
                        "",
                    ]
                )

    lines.extend(["Pair summary tables", "-------------------", ""])
    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            sub = [s for s in summary_rows if s[1] == group and s[0] == gender]
            if not sub:
                continue
            lines.append(f"{group} — {gender} (band {label})")
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

    lines.extend(
        [
            "When to use these vs other formula sets",
            "---------------------------------------",
            "  • Use these when filtering by WA point band membership (width 200).",
            "  • See also time_models_higher_races_formulas.txt (6 races),",
            "    time_models_5_races_formulas.txt (5 races), and",
            "    cross_event_time_models_report.txt (full sample).",
            "",
            "Source: Point_Bands_Time_Models/generate_point_band_formulas.py",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")
    return best_rows


def main() -> None:
    profiles_by_gg: dict[tuple[str, str], dict[str, dict]] = {}
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            profiles = load_athlete_season_records(folder, prefix, gender, events)
            profiles_by_gg[(gender, event_group)] = profiles
            print(f"Loaded {gender} {event_group}: {len(profiles)} athlete-seasons")

    for lo, hi in TARGET_BANDS:
        out = OUTPUT_ROOT / f"time_models_band_{lo}_{hi}_formulas.txt"
        rows = write_band_formulas(
            lo=lo, hi=hi, profiles_by_gg=profiles_by_gg, path=out
        )
        mean_cv = (
            statistics.mean(r["best_cv_median_abs"] for r in rows) if rows else float("nan")
        )
        print(
            f"Wrote {out.name}: {len(rows)} pairs, mean CV {mean_cv:.3f}s"
        )


if __name__ == "__main__":
    main()
