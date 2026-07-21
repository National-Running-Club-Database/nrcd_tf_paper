"""Feature-important point-band time models including 3000m Steeplechase.

Only builds models if including steeplechase adds athlete-seasons to the WA bands
750–950, 800–1000, and/or 850–1050 (vs the prior no-steeple point-band definition).

Distance data: New_Steeplechase_Data/Distance_Relays_Findings (corrected steeple WA).
Sprints: Relays_Findings/Sprinters_Relays_Findings.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEEPLE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
TIME_MODELS = PROJECT_ROOT / "time_models"
RELAYS = PROJECT_ROOT / "Relays_Findings"
BAND_FI = TIME_MODELS / "Point_Bands_Time_Models" / "Feature_Importance_Point_Band_Time_Models"
MODEL_SEARCH = TIME_MODELS / "model_search"
SPEC_ROOT = TIME_MODELS / "specialized_time_models"
DIST_DATA = STEEPLE_ROOT / "Distance_Relays_Findings"

sys.path.insert(0, str(RELAYS))
sys.path.insert(0, str(TIME_MODELS))
sys.path.insert(0, str(MODEL_SEARCH))
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(TIME_MODELS / "Point_Bands_Time_Models"))
sys.path.insert(0, str(BAND_FI))

from relay_rq1_data import (  # noqa: E402
    STANDARD_RELAY_IDS,
    is_in_season,
    points_col,
    season_rows,
)
from build_cross_event_time_models import linreg, parse_performance, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    CV_FOLDS,
    CV_SEED,
    evaluate_chain_models,
    evaluate_multivariate,
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
from analyze_feature_importance_point_bands import (  # noqa: E402
    BandProfile,
    COMBO_STRATEGIES,
    EPS,
    MIN_COHORT_N,
    MIN_REPORT_N,
    SPREAD_THRESHOLD,
    TARGET_BANDS,
    cv_combo_strategies,
    feature_labelers,
)

STEEPLE_ID = 20
SPRINT_EVENTS = {"100m": 3, "200m": 4, "400m": 6}
DISTANCE_EVENTS_NO_STEEPLE = {"800m": 9, "1500m": 11, "5000m": 17}
DISTANCE_EVENTS = {**DISTANCE_EVENTS_NO_STEEPLE, "3000m Steeplechase": 20}

EVENT_ORDER = {
    "Sprints": ["100m", "200m", "400m"],
    "Distance": ["800m", "1500m", "3000m Steeplechase", "5000m"],
}

FIT_PRED: dict[str, tuple[Callable | None, Callable | None]] = {
    "linear_ols": (fit_linear, pred_linear),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "log_linear": (fit_log_linear, pred_log_linear),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "binned_ratio": (None, pred_binned_ratio),
}

PARAMETRIC = {
    "linear_ols",
    "robust_trimmed_linear",
    "median_ratio",
    "log_linear",
    "ratio_linear",
    "quadratic",
}


def in_band(result_was: list[float], lo: int, hi: int) -> bool:
    return any(lo <= wa < hi for wa in result_was)


def load_season_rows(event_group: str, gender: str, year: str) -> list[dict]:
    if event_group == "Distance":
        path = DIST_DATA / f"Relays_Distance_{gender}_Outdoor_{year}_Data.csv"
        if not path.exists():
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    return season_rows("Sprinters_Relays_Findings", "Sprinters", gender, year)


def load_profiles(
    gender: str,
    event_group: str,
    events: dict[str, int],
) -> dict[str, BandProfile]:
    id_to_name = {eid: name for name, eid in events.items()}
    allowed = set(events.values())
    pcol = points_col(gender)
    season_results: dict[str, dict[str, tuple[str, float, float]]] = defaultdict(dict)

    for year in ("2024", "2025", "2026"):
        for row in load_season_rows(event_group, gender, year):
            date_str = (row.get("start_date") or "").strip()
            if date_str and not is_in_season(date_str):
                continue
            try:
                event_id = int(row["running_event_id"])
            except (TypeError, ValueError):
                continue
            if event_id in STANDARD_RELAY_IDS:
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
            rid = (row.get("result_id") or "").strip()
            if not rid:
                continue
            key = f"{aid}|{year}"
            season_results[key][rid] = (id_to_name[event_id], t, wa)

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
            event_group=event_group,
            event_times=event_times,
            event_wa=event_wa,
            result_was=result_was,
            wa_spread=max(wa_vals) - min(wa_vals),
            best_event=max(event_wa, key=lambda e: event_wa[e]),
            events_competed=len(event_times),
            max_wa=max(wa_vals),
        )
    return out


def band_athlete_keys(
    profiles_by_gg: dict[tuple[str, str], dict[str, BandProfile]],
    lo: int,
    hi: int,
) -> set[str]:
    keys: set[str] = set()
    for profiles in profiles_by_gg.values():
        for k, p in profiles.items():
            if in_band(p.result_was, lo, hi):
                keys.add(k)
    return keys


def run_search_for_bests(
    bests: dict[str, dict[str, float]],
    gender: str,
    event_group: str,
) -> list[dict]:
    order = EVENT_ORDER[event_group]
    best_rows: list[dict] = []
    for from_ev, to_ev in permutations(order, 2):
        pairs = [
            (b[from_ev], b[to_ev])
            for b in bests.values()
            if from_ev in b and to_ev in b
        ]
        if len(pairs) < MIN_REPORT_N:
            continue
        results = evaluate_pair(pairs)
        baseline_cv = next(r for r in results if r.name == "linear_ols").cv_median_abs
        extra = []
        if event_group == "Sprints" and from_ev == "100m" and to_ev == "400m":
            c = evaluate_chain_models(bests, "100m", "200m", "400m")
            if c:
                extra.append(c)
            m = evaluate_multivariate(bests, "100m", "200m", "400m")
            if m:
                extra.append(m)
        if event_group == "Distance" and from_ev == "800m" and to_ev == "5000m":
            c = evaluate_chain_models(bests, "800m", "1500m", "5000m")
            if c:
                extra.append(c)
            m = evaluate_multivariate(bests, "800m", "1500m", "5000m")
            if m:
                extra.append(m)
        all_r = results + extra
        winner = min(all_r, key=lambda r: r.cv_median_abs)
        best_rows.append(
            {
                "gender": gender,
                "event_group": event_group,
                "from_event": from_ev,
                "to_event": to_ev,
                "n_athlete_seasons": len(pairs),
                "best_model": winner.name,
                "linear_cv_median_abs": round(baseline_cv, 4),
                "best_cv_median_abs": round(winner.cv_median_abs, 4),
                "params_json": json.dumps(winner.params),
                "params_summary": format_params(winner.name, winner.params),
            }
        )
    return best_rows


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
        return f"{to_ev} = {from_ev} × ratio_bin({from_ev})"
    if model == "knn_median":
        return f"{to_ev} ≈ median of k={params.get('k', '?')} nearest {from_ev} neighbors"
    if model == "multivariate_ols":
        return (
            f"{to_ev} = {params['intercept']:.2f} + {params['coef_a']:.3f}×{params['event_a']} "
            f"+ {params['coef_b']:.3f}×{params['event_b']}"
        )
    return format_params(model, params)


def fit_full(pairs: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len(pairs) < MIN_COHORT_N:
        return None
    results = evaluate_pair(pairs)
    winner = min(results, key=lambda r: r.cv_median_abs)
    if winner.name not in PARAMETRIC and winner.name != "binned_ratio":
        parametric = [r for r in results if r.name in PARAMETRIC]
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
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_band_formulas(
    lo: int,
    hi: int,
    profiles_by_gg: dict,
    combo_by_pair: dict,
) -> None:
    band = f"{lo}-{hi}"
    lines = [
        f"Feature-Important Time Models — WA Band {band} (with Steeplechase)",
        "=" * (62 + len(band)),
        "",
        "Method:",
        "  • Outdoor 2024–2026, March 1+.",
        f"  • Population: athlete-seasons with ≥1 individual result WA in [{lo}, {hi}).",
        "  • Distance group includes 800m, 1500m, 3000m Steeplechase, 5000m",
        "    (corrected steeplechase WA from New_Steeplechase_Data).",
        "  • Relays excluded. Feature routing: bal/spec, best_event, best_is_from,",
        "    pair_wa_gap_50, from_stronger_wa, bal×best_event.",
        f"  • Min pair n={MIN_REPORT_N}; cohort fit n≥{MIN_COHORT_N}.",
        "",
    ]

    for event_group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            profiles = [
                p
                for p in profiles_by_gg[(gender, event_group)].values()
                if in_band(p.result_was, lo, hi)
            ]
            section = f"{event_group} — {gender}"
            lines.extend(["", section, "-" * len(section), ""])
            bests = {p.key: p.event_times for p in profiles}
            rows = run_search_for_bests(bests, gender, event_group)
            if not rows:
                lines.append(f"No pairs met n≥{MIN_REPORT_N}.")
                lines.append("")
                continue
            for row in sorted(rows, key=lambda r: (r["from_event"], r["to_event"])):
                from_ev, to_ev = row["from_event"], row["to_event"]
                pairs = [
                    (b[from_ev], b[to_ev])
                    for b in bests.values()
                    if from_ev in b and to_ev in b
                ]
                params = json.loads(row["params_json"])
                formula = human_formula(row["best_model"], params, from_ev, to_ev)
                _, _, r, _, _ = linreg(pairs)
                r_med, r_p25, r_p75 = ratio_quartiles(pairs)
                key = (band, gender, event_group, from_ev, to_ev)
                combo = combo_by_pair.get(key)
                lines.append(
                    f"{from_ev} -> {to_ev}  (n={len(pairs)} athlete-seasons, r={r:.3f})"
                )
                lines.append(f"  Pooled band model: {row['best_model']}")
                lines.append(f"  Formula: {formula.split(chr(10))[0]}")
                lines.append(
                    f"  CV median |error|: {row['best_cv_median_abs']:.3f}s"
                )
                lines.append(
                    f"  Ratio {to_ev}/{from_ev}: median {r_med:.3f} "
                    f"(IQR {r_p25:.3f}–{r_p75:.3f})"
                )
                lines.append(
                    f"  Median times: {from_ev} {statistics.median(p[0] for p in pairs):.2f}s, "
                    f"{to_ev} {statistics.median(p[1] for p in pairs):.2f}s"
                )
                if combo:
                    best_s = combo["best_strategy"]
                    lines.append(
                        f"  Important features: recommended {best_s} "
                        f"(pooled CV {combo['cv_pooled']:.3f}s → "
                        f"{combo[f'cv_{best_s}']:.3f}s, Δ {combo['delta_best_vs_pooled']:+.3f}s)."
                    )
                    helpful = []
                    for s in COMBO_STRATEGIES:
                        if s == "pooled":
                            continue
                        d = combo["cv_pooled"] - combo[f"cv_{s}"]
                        if d > EPS:
                            helpful.append(f"{s} (Δ {d:+.3f}s)")
                    if helpful:
                        lines.append("  Features that beat pooled: " + "; ".join(helpful))
                    else:
                        lines.append(
                            "  No feature split beat pooled by >0.01s — use pooled formula."
                        )
                lines.append("")

    lines.append(
        "Source: New_Steeplechase_Data/Feature_Importance_Point_Band_Time_Models/"
        "analyze_feature_importance_with_steeple.py"
    )
    (
        OUTPUT_ROOT / f"new_feature_important_time_models_band_{lo}_{hi}_with_steeple.txt"
    ).write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Baseline (no steeple) vs with steeple
    base_by_gg: dict[tuple[str, str], dict[str, BandProfile]] = {}
    with_by_gg: dict[tuple[str, str], dict[str, BandProfile]] = {}

    for gender in ("Men", "Women"):
        base_by_gg[(gender, "Sprints")] = load_profiles(gender, "Sprints", SPRINT_EVENTS)
        with_by_gg[(gender, "Sprints")] = base_by_gg[(gender, "Sprints")]
        base_by_gg[(gender, "Distance")] = load_profiles(
            gender, "Distance", DISTANCE_EVENTS_NO_STEEPLE
        )
        with_by_gg[(gender, "Distance")] = load_profiles(
            gender, "Distance", DISTANCE_EVENTS
        )
        print(
            f"{gender}: Sprints={len(with_by_gg[(gender,'Sprints')])}  "
            f"Distance no-steeple={len(base_by_gg[(gender,'Distance')])}  "
            f"Distance+steeple={len(with_by_gg[(gender,'Distance')])}"
        )

    membership_rows = []
    any_new = False
    for lo, hi in TARGET_BANDS:
        a = band_athlete_keys(base_by_gg, lo, hi)
        b = band_athlete_keys(with_by_gg, lo, hi)
        new = b - a
        membership_rows.append(
            {
                "band": f"{lo}-{hi}",
                "n_without_steeple": len(a),
                "n_with_steeple": len(b),
                "n_new_athlete_seasons": len(new),
                "new_keys_sample": ",".join(sorted(new)[:20]),
            }
        )
        print(
            f"Band {lo}-{hi}: without={len(a)} with={len(b)} new={len(new)}"
        )
        if new:
            any_new = True

    write_csv(OUTPUT_ROOT / "band_membership_with_vs_without_steeple.csv", membership_rows)

    findings = [
        "Steeplechase-Inclusive Point-Band Models — Gate Check",
        "=====================================================",
        "",
        "Question: Does including 3000m Steeplechase add athlete-seasons to the",
        "WA bands 750–950, 800–1000, 850–1050?",
        "",
    ]
    for r in membership_rows:
        findings.append(
            f"  {r['band']}: {r['n_without_steeple']} → {r['n_with_steeple']} "
            f"(+{r['n_new_athlete_seasons']} new athlete-seasons)"
        )

    if not any_new:
        findings.extend(
            [
                "",
                "Verdict: NO new athletes in any band — skipping new time models",
                "(would be redundant with existing Point_Bands models).",
                "",
                "Source: analyze_feature_importance_with_steeple.py",
            ]
        )
        (OUTPUT_ROOT / "steeple_band_gate_findings.txt").write_text(
            "\n".join(findings).rstrip() + "\n"
        )
        print("No new athletes — skipping model build.")
        return

    findings.extend(
        [
            "",
            "Verdict: YES — at least one band gains athletes. Building",
            "feature-important time models with steeplechase included.",
            "",
            "Source: analyze_feature_importance_with_steeple.py",
        ]
    )
    (OUTPUT_ROOT / "steeple_band_gate_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )

    # Feature importance + formulas for each band
    all_combo: list[dict] = []
    combo_by_pair: dict = {}
    all_best: list[dict] = []

    for lo, hi in TARGET_BANDS:
        band = f"{lo}-{hi}"
        print(f"Analyzing band {band} with steeple...")
        for (gender, event_group), profiles in with_by_gg.items():
            band_ps = [
                p for p in profiles.values() if in_band(p.result_was, lo, hi)
            ]
            bests = {p.key: p.event_times for p in band_ps}
            for row in run_search_for_bests(bests, gender, event_group):
                all_best.append({**row, "band": band, "band_lo": lo, "band_hi": hi})

            order = EVENT_ORDER[event_group]
            for from_ev, to_ev in permutations(order, 2):
                pair_ps = [
                    p
                    for p in band_ps
                    if from_ev in p.event_times and to_ev in p.event_times
                ]
                if len(pair_ps) < MIN_REPORT_N:
                    continue
                labelers = feature_labelers(pair_ps, from_ev, to_ev)
                records = [
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
                combo = cv_combo_strategies(records)
                if not combo:
                    continue
                best_name = min(COMBO_STRATEGIES, key=lambda s: combo[s])
                row = {
                    "band": band,
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
                all_combo.append(row)
                combo_by_pair[(band, gender, event_group, from_ev, to_ev)] = row

        write_band_formulas(lo, hi, with_by_gg, combo_by_pair)

    write_csv(OUTPUT_ROOT / "pair_strategy_cv_by_band_with_steeple.csv", all_combo)
    write_csv(OUTPUT_ROOT / "best_time_models_by_band_with_steeple.csv", all_best)

    # Summary report
    report = [
        "Feature Importance Point-Band Time Models — With Steeplechase",
        "===============================================================",
        "",
        "Bands: 750–950, 800–1000, 850–1050.",
        "Distance events: 800m, 1500m, 3000m Steeplechase, 5000m.",
        "Steeplechase WA from New_Steeplechase_Data.",
        "",
        "Band membership vs prior (no steeple):",
    ]
    for r in membership_rows:
        report.append(
            f"  {r['band']}: +{r['n_new_athlete_seasons']} athlete-seasons "
            f"({r['n_without_steeple']} → {r['n_with_steeple']})"
        )
    report.append("")
    for lo, hi in TARGET_BANDS:
        band = f"{lo}-{hi}"
        rows = [r for r in all_combo if r["band"] == band]
        report.append(f"=== Band {band} ({len(rows)} pairs with feature CV) ===")
        wins: dict[str, int] = defaultdict(int)
        for r in rows:
            wins[r["best_strategy"]] += 1
        for s, n in sorted(wins.items(), key=lambda x: -x[1]):
            report.append(f"  best_strategy={s}: {n} pairs")
        report.append(
            f"  Formula file: new_feature_important_time_models_band_{lo}_{hi}_with_steeple.txt"
        )
        report.append("")
    report.append("Source: analyze_feature_importance_with_steeple.py")
    (OUTPUT_ROOT / "feature_importance_with_steeple_report.txt").write_text(
        "\n".join(report).rstrip() + "\n"
    )
    print(f"Wrote steeple-inclusive models to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
