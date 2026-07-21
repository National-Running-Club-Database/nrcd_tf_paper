"""Generate cross-event formula reports for high race-volume athlete-seasons."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from itertools import permutations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIME_MODELS_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
RACE_COUNT_DIR = MODEL_SEARCH_ROOT / "race_count"
OUTPUT_DIR = TIME_MODELS_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "relays_findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))

from build_cross_event_time_models import GROUP_CONFIG, linreg, ratio_quartiles  # noqa: E402
from compare_time_model_candidates import (  # noqa: E402
    pred_linear,
    pred_log_linear,
    pred_median_ratio,
    pred_ratio_linear,
    pred_quadratic,
    pred_binned_ratio,
    fit_linear,
    fit_log_linear,
    fit_median_ratio,
    fit_ratio_linear,
    fit_quadratic,
    fit_robust_trimmed,
    metrics,
    evaluate_chain_models,
    evaluate_multivariate,
    evaluate_pair,
    format_params,
    EVENT_ORDER,
    CV_FOLDS,
    CV_SEED,
)
from compare_time_models_by_race_count import (  # noqa: E402
    load_athlete_season_pbs as _load_exact,
    run_model_search,
    MIN_PAIR_N,
)

RaceFilter = int | frozenset[int]

PRED = {
    "linear_ols": (fit_linear, pred_linear),
    "log_linear": (fit_log_linear, pred_log_linear),
    "median_ratio": (fit_median_ratio, pred_median_ratio),
    "ratio_linear": (fit_ratio_linear, pred_ratio_linear),
    "quadratic": (fit_quadratic, pred_quadratic),
    "robust_trimmed_linear": (fit_robust_trimmed, pred_linear),
    "binned_ratio": (None, pred_binned_ratio),
    "knn_median": (None, None),
    "chain_residual_adjusted": (None, None),
    "multivariate_ols": (None, None),
}


def load_athlete_season_pbs(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
    race_filter: RaceFilter | None,
) -> dict[str, dict[str, float]]:
    if isinstance(race_filter, int) or race_filter is None:
        return _load_exact(folder, prefix, gender, event_name_to_id, race_filter)

    allowed = set(race_filter)
    all_bests: dict[str, dict[str, float]] = {}
    for rc in allowed:
        subset = _load_exact(folder, prefix, gender, event_name_to_id, rc)
        all_bests.update(subset)
    return all_bests


def run_model_search_filter(
    race_filter: RaceFilter | None,
    label: str,
) -> list[dict]:
    from itertools import permutations as perm

    best_rows: list[dict] = []
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        order = EVENT_ORDER[event_group]
        for gender in ("Men", "Women"):
            bests = load_athlete_season_pbs(folder, prefix, gender, events, race_filter)
            for from_ev, to_ev in perm(order, 2):
                pairs = [
                    (b[from_ev], b[to_ev])
                    for b in bests.values()
                    if from_ev in b and to_ev in b
                ]
                if len(pairs) < MIN_PAIR_N:
                    continue

                results = evaluate_pair(pairs)
                baseline_cv = next(r for r in results if r.name == "linear_ols").cv_median_abs
                extra = []
                if event_group == "Sprints" and from_ev == "100m" and to_ev == "400m":
                    chain = evaluate_chain_models(bests, "100m", "200m", "400m")
                    if chain:
                        extra.append(chain)
                    multi = evaluate_multivariate(bests, "100m", "200m", "400m")
                    if multi:
                        extra.append(multi)
                if event_group == "Distance" and from_ev == "800m" and to_ev == "5000m":
                    chain = evaluate_chain_models(bests, "800m", "1500m", "5000m")
                    if chain:
                        extra.append(chain)
                    multi = evaluate_multivariate(bests, "800m", "1500m", "5000m")
                    if multi:
                        extra.append(multi)

                all_results = results + extra
                winner = min(all_results, key=lambda r: r.cv_median_abs)
                improve = baseline_cv - winner.cv_median_abs
                best_rows.append(
                    {
                        "race_count_filter": label,
                        "gender": gender,
                        "event_group": event_group,
                        "from_event": from_ev,
                        "to_event": to_ev,
                        "n_athlete_seasons": len(pairs),
                        "best_model": winner.name,
                        "linear_cv_median_abs": round(baseline_cv, 4),
                        "best_cv_median_abs": round(winner.cv_median_abs, 4),
                        "improvement_seconds": round(improve, 4),
                        "params_json": json.dumps(winner.params),
                        "params_summary": format_params(winner.name, winner.params),
                        "notes": winner.notes,
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
        return f"{to_ev} = {from_ev} × ratio_bin({from_ev})  [nearest speed bin; see binned table below]"
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


def pair_stats(
    race_filter: RaceFilter | None,
    gender: str,
    event_group: str,
    from_ev: str,
    to_ev: str,
    row: dict,
) -> dict[str, Any]:
    folder, prefix, events = GROUP_CONFIG[event_group]
    bests = load_athlete_season_pbs(folder, prefix, gender, events, race_filter)
    pairs = [(b[from_ev], b[to_ev]) for b in bests.values() if from_ev in b and to_ev in b]
    model = row["best_model"]
    params = json.loads(row["params_json"])
    _, _, r, _, _ = linreg(pairs)
    r_med, r_p25, r_p75 = ratio_quartiles(pairs)
    from_times = [p[0] for p in pairs]
    to_times = [p[1] for p in pairs]

    full_med = None
    full_rmse = None
    if model in PRED and PRED[model][1] is not None:
        pred_fn = PRED[model][1]
        preds = [pred_fn(x, params) for x, _ in pairs]
        actual = [y for _, y in pairs]
        full_med, full_rmse = metrics(actual, preds)

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
        "from_med": statistics.median(from_times),
        "to_med": statistics.median(to_times),
        "params": params,
        "binned_lines": binned_table(params, from_ev, to_ev),
    }


def load_all_baseline_rows() -> dict[tuple[str, str, str, str], float]:
    path = RACE_COUNT_DIR / "best_time_models_by_race_count.csv"
    out: dict[tuple[str, str, str, str], float] = {}
    for row in csv.DictReader(open(path)):
        if row["race_count_filter"] != "all":
            continue
        key = (row["gender"], row["event_group"], row["from_event"], row["to_event"])
        out[key] = float(row["best_cv_median_abs"])
    return out


def write_report(
    *,
    label: str,
    title_suffix: str,
    population_line: str,
    use_line: str,
    race_filter: RaceFilter | None,
    best_rows: list[dict],
    output_path: Path,
    mean_cv_note: str | None = None,
) -> None:
    baseline = load_all_baseline_rows()
    lines = [
        f"Cross-Event Time Models — {title_suffix} (Sprints & Distance)",
        "=" * (len(title_suffix) + 52),
        "",
        "Method:",
        "  • Data: relay-inclusive outdoor CSVs, 2024–2026, results on or after March 1.",
        f"  • Population: {population_line}",
        "  • Athlete-season PB: fastest valid mark per event within that season (WA > 0).",
        "  • Model selection: 5-fold cross-validation across 8+ candidate model families;",
        "    the formula below is the lowest-CV-median-|error| winner for each pair.",
        "  • 3000m Steeplechase excluded from distance.",
        f"  • Minimum n = {MIN_PAIR_N} athlete-season pairs required to report a model.",
        "",
        "Use:",
        f"  • {use_line}",
        "  • Predict time in the unknown event from known PB(s), then convert predicted",
        "    time to World Athletics points in the TARGET event table.",
        "  • Treat CV median |error| as the typical ± uncertainty band (cross-validated).",
        "  • Do not compare WA across events directly.",
        "",
    ]
    if mean_cv_note:
        lines.extend(["Coverage note:", f"  {mean_cv_note}", ""])

    reported_keys: set[tuple[str, str, str, str]] = set()
    summary_rows: list[tuple[str, str, str, str, dict]] = []

    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            section_rows = [
                r
                for r in best_rows
                if r["gender"] == gender and r["event_group"] == group
            ]
            if not section_rows:
                continue
            section = f"{group} — {gender}"
            lines.extend(["", section, "-" * len(section), ""])
            section_rows.sort(key=lambda r: (r["from_event"], r["to_event"]))
            for row in section_rows:
                from_ev, to_ev = row["from_event"], row["to_event"]
                key = (gender, group, from_ev, to_ev)
                reported_keys.add(key)
                stats = pair_stats(race_filter, gender, group, from_ev, to_ev, row)
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

    # Missing pairs note per section
    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            order = EVENT_ORDER[group]
            missing = []
            for from_ev, to_ev in permutations(order, 2):
                if (gender, group, from_ev, to_ev) not in reported_keys:
                    missing.append(f"{from_ev} -> {to_ev}")
            if missing:
                lines.extend([
                    f"Not reported for {group} — {gender} (n < {MIN_PAIR_N} athlete-season pairs):",
                    "  " + ", ".join(missing),
                    "",
                ])

    lines.extend(["Pair summary tables", "-------------------", ""])
    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            sub = [s for s in summary_rows if s[1] == group and s[0] == gender]
            if not sub:
                continue
            lines.append(f"{group} — {gender} ({title_suffix.lower()})")
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

    # Accuracy vs all-season
    improvements = []
    for gender, group, from_ev, to_ev, stats in summary_rows:
        base = baseline.get((gender, group, from_ev, to_ev))
        if base is not None:
            improvements.append((from_ev, to_ev, gender, group, base, stats["cv_med"]))

    if improvements:
        lines.extend([
            "Accuracy vs all athlete-season models",
            "-------------------------------------",
        ])
        improvements.sort(key=lambda x: x[4] - x[5], reverse=True)
        for from_ev, to_ev, gender, group, base, cv in improvements[:8]:
            delta = cv - base
            lines.append(
                f"  {gender} {group} {from_ev}->{to_ev}: all-season {base:.3f}s -> "
                f"{label} {cv:.3f}s ({delta:+.3f}s)"
            )
        lines.append("")

    lines.extend([
        "When to use these vs other formula sets",
        "---------------------------------------",
        "  • See also time_models_higher_races_formulas.txt (6 races),",
        "    time_models_5_races_formulas.txt (5 races), and",
        "    cross_event_time_models_report.txt (full sample).",
        "",
        "Source: time_models/generate_higher_races_formula_reports.py",
        f"        time_models/model_search/race_count/best_time_models_by_race_count.csv",
    ])
    output_path.write_text("\n".join(lines).rstrip() + "\n")


def append_best_rows(rows: list[dict], csv_path: Path) -> None:
    existing = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    existing_keys = {
        (r["race_count_filter"], r["gender"], r["event_group"], r["from_event"], r["to_event"])
        for r in existing
    }
    merged = [r for r in existing if r["race_count_filter"] not in {"5_or_6"}]
    for row in rows:
        key = (row["race_count_filter"], row["gender"], row["event_group"], row["from_event"], row["to_event"])
        if key not in existing_keys or row["race_count_filter"] == "5_or_6":
            merged.append(row)
    # dedupe keeping last
    deduped: dict[tuple, dict] = {}
    for r in merged:
        k = (r["race_count_filter"], r["gender"], r["event_group"], r["from_event"], r["to_event"])
        deduped[k] = r
    final = list(deduped.values())
    if final:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(final[0].keys()))
            w.writeheader()
            w.writerows(final)


def main() -> None:
    csv_path = RACE_COUNT_DIR / "best_time_models_by_race_count.csv"
    all_rows = list(csv.DictReader(open(csv_path)))

    rows_5 = [r for r in all_rows if r["race_count_filter"] == "5"]
    rows_5_or_6 = [r for r in all_rows if r["race_count_filter"] == "5_or_6"]
    if not rows_5_or_6:
        print("Running model search for 5_or_6 races...")
        rows_5_or_6 = run_model_search_filter(frozenset({5, 6}), "5_or_6")
        append_best_rows(rows_5_or_6, csv_path)

    write_report(
        label="5 races",
        title_suffix="5 Races Per Season",
        population_line=(
            "athlete-seasons with exactly 5 valid results in the event group's "
            "discipline CSV (same counting as the race-volume analysis)."
        ),
        use_line=(
            "Apply these formulas when the athlete has run exactly 5 races in the "
            "event group during the season. Mean CV median |error| across evaluated "
            "pairs is 4.90s vs 6.77s for all athlete-seasons."
        ),
        race_filter=5,
        best_rows=rows_5,
        output_path=OUTPUT_DIR / "time_models_5_races_formulas.txt",
        mean_cv_note=(
            "16 of 24 event pairs met n ≥ 20 at exactly 5 races. Women's distance "
            "pairs qualify; women's long sprint hops (100m↔400m) generally do not."
        ),
    )
    print(f"Wrote {OUTPUT_DIR / 'time_models_5_races_formulas.txt'}")

    mean_cv = statistics.mean(float(r["best_cv_median_abs"]) for r in rows_5_or_6)
    write_report(
        label="5 or 6 races",
        title_suffix="5 or 6 Races Per Season",
        population_line=(
            "athlete-seasons with exactly 5 OR 6 valid results in the event group's "
            "discipline CSV (combined high-volume cohort)."
        ),
        use_line=(
            "Apply these formulas when the athlete has run 5 or 6 races in the event "
            f"group during the season (combined cohort). Mean CV median |error| across "
            f"evaluated pairs is {mean_cv:.2f}s vs 6.77s for all athlete-seasons."
        ),
        race_filter=frozenset({5, 6}),
        best_rows=rows_5_or_6,
        output_path=OUTPUT_DIR / "time_models_5_or_6_races_formulas.txt",
        mean_cv_note=(
            f"{len(rows_5_or_6)} event pairs met n ≥ 20 in the combined 5-or-6-race "
            "cohort. Use when race count is 5 or 6 and you want the largest available "
            "sample for that high-volume group."
        ),
    )
    print(f"Wrote {OUTPUT_DIR / 'time_models_5_or_6_races_formulas.txt'}")


if __name__ == "__main__":
    main()
