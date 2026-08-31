"""Audit data-leakage risks in the collegiate time-model test pipelines.

Focus
-----
Identity isolation (club fit vs NCAA/NAIA scrape) is expected to be clean.
This script quantifies **feature / protocol leakage**: routing labels and
season profiles that incorporate the **target** event (or full-season marks)
when scoring predictions of that event.

Usage
-----
  python analyze_test_data_leakage.py
  python analyze_test_data_leakage.py --dataset All_Divisions
  python analyze_test_data_leakage.py --dataset D1_Only
  python analyze_test_data_leakage.py --dataset both

Writes:
  output/<dataset>/data_leakage_report.txt
  output/<dataset>/data_leakage_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output"

# Strategies whose cohort label uses WA(to) / the target mark directly
TARGET_WA_STRATEGIES = frozenset(
    {
        "from_stronger_wa",
        "pair_wa_gap_50",
        "pair_wa_gap_median",
        "best_is_to",
    }
)

# Strategies that use a full-season event profile (includes target unless leave-out)
PROFILE_STRATEGIES = frozenset(
    {
        "bal_spec",
        "best_event",
        "best_is_from",
        "bal_x_best_event",
    }
)

LEAKAGE_FINDINGS = """
DATA LEAKAGE AUDIT — Time-model test pipelines
==============================================

Executive verdict
-----------------
• Train/test IDENTITY isolation (club NRCD fit vs collegiate scrape) looks clean.
• FEATURE / PROTOCOL leakage is present: some routing labels and season profiles
  incorporate the TARGET event's WA / PB when predicting that event.

1. Confirmed issues
-------------------
1A. Target-WA routing (CRITICAL)
    Strategies from_stronger_wa and pair_wa_gap_* compare WA(from) to WA(to).
    WA(to) is a transform of the time being predicted.
    Code: validate.py :: cohort_label(); club FI scripts :: feature_labelers().

1B. Full-season profile includes target (CRITICAL if framed as forecast)
    best_event, bal_spec / WA_Spread, bal_x_best_event, best_is_from use the
    full event-group season PB profile (preprocess.py :: add_features).
    Even best_is_from needs a group argmax that includes WA(to).

1C. Chronological uses full-season features (MODERATE–CRITICAL)
    (x, y) dates are early→later, but Bal_Spec / Best_Event (and band membership)
    can use marks from the whole season, including after date_from.

1D. Band membership may include the target mark (MODERATE)
    athlete_in_band(): ≥1 result WA in [lo, hi) — may be the target itself.

1E. Strategy selection used the same leaked labels in club CV (MODERATE)
    "Recommended route" in band model reports was chosen with those labels.

2. Framing
----------
• Season-PB as joint equivalency ("given both season bests…") → less severe.
• Season-PB as forecasting unknown B from known A → 1A/1B are leakage.

3. Clear non-issues
-------------------
• Club CSVs use athlete_id/team_id; test scrapes use college names — no shared fit.
• Validators parse frozen formula text; they do not refit on test CSVs.
• Pooled formulas predict from from-event time only (once frozen).
• RQ1 / scoring / volume analyses are not this predictive holdout design.

4. Remediations
---------------
• Drop or quarantine target-WA strategies for predictive claims.
• Leave-target-out profiles (and time-aware profiles for chronological).
• Prospective band = from-event WA (or marks ≤ prediction time) only.
• Report pooled-only metrics separately from "oracle" feature routing.
""".strip()


def _dataset_paths(name: str) -> dict[str, Path]:
    if name == "All_Divisions":
        base = OUT_ROOT / "All_Divisions"
        return {
            "base": base,
            "predictions": base / "model_validation" / "predictions.csv",
            "results": base / "men_test_dataset_results.csv",
            "report": base / "data_leakage_report.txt",
            "summary_csv": base / "data_leakage_summary.csv",
        }
    if name == "D1_Only":
        base = OUT_ROOT
        return {
            "base": base,
            "predictions": base / "model_validation" / "D1_Only_predictions.csv",
            "results": base / "D1_Only_men_test_dataset_results.csv",
            "report": base / "D1_Only_data_leakage_report.txt",
            "summary_csv": base / "D1_Only_data_leakage_summary.csv",
        }
    raise ValueError(f"Unknown dataset: {name}")


def analyze_predictions(pred: pd.DataFrame) -> dict[str, float | int | str]:
    """Quantify leakage-related flags on season-PB feature-route rows."""
    f = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    p = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "pooled")].copy()
    if f.empty:
        return {"n_feature": 0, "n_pooled": len(p)}

    strat = f["model_recommended_strategy"].fillna("").astype(str)
    target_wa = strat.isin(TARGET_WA_STRATEGIES)
    profile = strat.isin(PROFILE_STRATEGIES)
    # Applied feature cohort (not pooled fallback) with target-WA strategy
    applied_target = target_wa & f["route_detail"].astype(str).str.startswith("feature:")
    best_is_to = (
        f["best_event"].notna() & (f["best_event"].astype(str) == f["to_event"].astype(str))
    )

    out: dict[str, float | int | str] = {
        "n_feature": int(len(f)),
        "n_pooled": int(len(p)),
        "n_athlete_seasons": int(
            f.groupby(["college", "athlete", "season_year"]).ngroups
        ),
        "pct_recommended_target_wa_strategy": float(target_wa.mean()),
        "n_recommended_target_wa_strategy": int(target_wa.sum()),
        "pct_applied_target_wa_feature_cohort": float(applied_target.mean()),
        "n_applied_target_wa_feature_cohort": int(applied_target.sum()),
        "pct_recommended_profile_strategy": float(profile.mean()),
        "n_recommended_profile_strategy": int(profile.sum()),
        "pct_best_event_equals_to_event": float(best_is_to.mean()),
        "n_best_event_equals_to_event": int(best_is_to.sum()),
        "feature_within_tol_rate": float(f["within_tolerance"].mean()),
        "pooled_within_tol_rate": float(p["within_tolerance"].mean()) if len(p) else float("nan"),
        "feature_medape": float(f["abs_pct_error"].median()),
        "pooled_medape": float(p["abs_pct_error"].median()) if len(p) else float("nan"),
    }

    # Stratified: feature rows that avoid target-WA recommended strategies
    clean = f[~target_wa]
    if len(clean):
        out["n_feature_excl_target_wa_strategy"] = int(len(clean))
        out["medape_excl_target_wa_strategy"] = float(clean["abs_pct_error"].median())
        out["within_tol_excl_target_wa_strategy"] = float(clean["within_tolerance"].mean())
    else:
        out["n_feature_excl_target_wa_strategy"] = 0
        out["medape_excl_target_wa_strategy"] = float("nan")
        out["within_tol_excl_target_wa_strategy"] = float("nan")

    return out


def strategy_breakdown(pred: pd.DataFrame) -> pd.DataFrame:
    f = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    if f.empty:
        return pd.DataFrame()
    g = (
        f.groupby(f["model_recommended_strategy"].fillna("(none)"), dropna=False)
        .agg(
            n=("abs_pct_error", "size"),
            medape=("abs_pct_error", "median"),
            within_tol=("within_tolerance", "mean"),
        )
        .reset_index()
        .rename(columns={"model_recommended_strategy": "strategy"})
    )
    g["uses_target_wa"] = g["strategy"].isin(TARGET_WA_STRATEGIES)
    g["uses_full_season_profile"] = g["strategy"].isin(PROFILE_STRATEGIES)
    g["share"] = g["n"] / g["n"].sum()
    return g.sort_values("n", ascending=False)


def band_target_overlap(results: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float | int]:
    """Share of feature predictions where band eligibility could involve to-event WA."""
    if results.empty or "World_Athletics_Score_Men" not in results.columns:
        return {}
    f = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    if f.empty:
        return {}

    # Map athlete-season → event WA from season PB (best WA per event)
    track = results.copy()
    if "Gender" in track.columns:
        track = track[track["Gender"].astype(str).str.lower().isin(["men", "m", "male"])]
    pb = (
        track.sort_values("Result_Value")
        .groupby(["College", "Athlete", "Season_Year", "Event"], as_index=False)
        .first()
    )

    n_check = 0
    n_to_in_band = 0
    n_only_to_puts_in_band = 0
    for _, r in f.iterrows():
        lo, hi = map(int, str(r["band"]).split("-"))
        key = (r["college"], r["athlete"], int(r["season_year"]))
        ath = pb[
            (pb["College"] == key[0])
            & (pb["Athlete"] == key[1])
            & (pb["Season_Year"] == key[2])
        ]
        if ath.empty:
            continue
        was = ath["World_Athletics_Score_Men"].dropna()
        if was.empty:
            continue
        n_check += 1
        to_rows = ath[ath["Event"] == r["to_event"]]
        to_wa = (
            float(to_rows["World_Athletics_Score_Men"].iloc[0])
            if len(to_rows) and pd.notna(to_rows["World_Athletics_Score_Men"].iloc[0])
            else None
        )
        in_band_any = ((was >= lo) & (was < hi)).any()
        if to_wa is not None and lo <= to_wa < hi:
            n_to_in_band += 1
        # Would athlete still be in band without to-event?
        other = ath[ath["Event"] != r["to_event"]]["World_Athletics_Score_Men"].dropna()
        in_band_without_to = ((other >= lo) & (other < hi)).any() if len(other) else False
        if in_band_any and not in_band_without_to:
            n_only_to_puts_in_band += 1

    if n_check == 0:
        return {}
    return {
        "n_band_checks": n_check,
        "pct_to_event_wa_in_band": n_to_in_band / n_check,
        "n_to_event_wa_in_band": n_to_in_band,
        "pct_band_only_via_to_event": n_only_to_puts_in_band / n_check,
        "n_band_only_via_to_event": n_only_to_puts_in_band,
    }


def format_report(
    dataset: str,
    metrics: dict,
    breakdown: pd.DataFrame,
    band_stats: dict,
) -> str:
    lines = [
        LEAKAGE_FINDINGS,
        "",
        f"QUANTITATIVE SCAN — {dataset}",
        "=" * (20 + len(dataset)),
        "",
        f"Feature-route season-PB rows: {metrics.get('n_feature', 0)}",
        f"Pooled-route season-PB rows:  {metrics.get('n_pooled', 0)}",
        f"Athlete-seasons (feature):    {metrics.get('n_athlete_seasons', 0)}",
        "",
        "Target-WA recommended strategies (from_stronger_wa / pair_wa_gap_*):",
        f"  {metrics.get('n_recommended_target_wa_strategy', 0)} / "
        f"{metrics.get('n_feature', 0)} = "
        f"{metrics.get('pct_recommended_target_wa_strategy', float('nan')):.1%}",
        "  of which applied as feature cohort (not pooled fallback): "
        f"{metrics.get('n_applied_target_wa_feature_cohort', 0)} "
        f"({metrics.get('pct_applied_target_wa_feature_cohort', float('nan')):.1%})",
        "",
        "Full-season profile recommended strategies "
        "(bal_spec / best_event / best_is_from / bal_x_best_event):",
        f"  {metrics.get('n_recommended_profile_strategy', 0)} / "
        f"{metrics.get('n_feature', 0)} = "
        f"{metrics.get('pct_recommended_profile_strategy', float('nan')):.1%}",
        "",
        f"best_event == to_event: "
        f"{metrics.get('n_best_event_equals_to_event', 0)} / "
        f"{metrics.get('n_feature', 0)} = "
        f"{metrics.get('pct_best_event_equals_to_event', float('nan')):.1%}",
        "",
        "Headline accuracy (for context; not a leakage test):",
        f"  feature MedAPE={metrics.get('feature_medape', float('nan')):.2f}%  "
        f"within_tol={metrics.get('feature_within_tol_rate', float('nan')):.1%}",
        f"  pooled  MedAPE={metrics.get('pooled_medape', float('nan')):.2f}%  "
        f"within_tol={metrics.get('pooled_within_tol_rate', float('nan')):.1%}",
        f"  feature excl. target-WA strategies: n="
        f"{metrics.get('n_feature_excl_target_wa_strategy', 0)}  "
        f"MedAPE={metrics.get('medape_excl_target_wa_strategy', float('nan')):.2f}%  "
        f"within_tol={metrics.get('within_tol_excl_target_wa_strategy', float('nan')):.1%}",
        "",
    ]
    if band_stats:
        lines += [
            "Band membership vs target event WA:",
            f"  to-event WA itself in assigned band: "
            f"{band_stats.get('n_to_event_wa_in_band', 0)} / "
            f"{band_stats.get('n_band_checks', 0)} = "
            f"{band_stats.get('pct_to_event_wa_in_band', float('nan')):.1%}",
            f"  band membership ONLY via to-event: "
            f"{band_stats.get('n_band_only_via_to_event', 0)} / "
            f"{band_stats.get('n_band_checks', 0)} = "
            f"{band_stats.get('pct_band_only_via_to_event', float('nan')):.1%}",
            "",
        ]
    if not breakdown.empty:
        lines.append("Recommended strategy breakdown (feature route):")
        lines.append(breakdown.to_string(index=False))
        lines.append("")
    lines.append(
        "Identity note: this scan does not re-join NRCD athlete_ids to NCAA names; "
        "club vs scrape schemas remain separate by construction."
    )
    return "\n".join(lines) + "\n"


def run_dataset(name: str) -> Path | None:
    paths = _dataset_paths(name)
    pred_path = paths["predictions"]
    if not pred_path.exists():
        print(f"[{name}] Missing predictions: {pred_path} — skip")
        return None

    pred = pd.read_csv(pred_path)
    metrics = analyze_predictions(pred)
    breakdown = strategy_breakdown(pred)

    band_stats: dict = {}
    results_path = paths["results"]
    if results_path.exists():
        results = pd.read_csv(results_path)
        band_stats = band_target_overlap(results, pred)

    report = format_report(name, metrics, breakdown, band_stats)
    paths["report"].write_text(report)
    print(f"[{name}] Wrote {paths['report']}")

    # Flat summary row + strategy table
    summary_rows = [{**{"dataset": name, "section": "overall"}, **metrics, **band_stats}]
    summary = pd.DataFrame(summary_rows)
    if not breakdown.empty:
        b2 = breakdown.copy()
        b2.insert(0, "dataset", name)
        b2.insert(1, "section", "by_strategy")
        # write two blocks: overall metrics as one CSV; strategies alongside
        out = pd.concat([summary, b2], ignore_index=True, sort=False)
    else:
        out = summary
    out.to_csv(paths["summary_csv"], index=False)
    print(f"[{name}] Wrote {paths['summary_csv']}")
    print(report[:1800])
    return paths["report"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quantify feature/protocol leakage in time-model test validation."
    )
    parser.add_argument(
        "--dataset",
        choices=("All_Divisions", "D1_Only", "both"),
        default="both",
        help="Which validation outputs to scan (default: both).",
    )
    args = parser.parse_args(argv)
    names = (
        ["All_Divisions", "D1_Only"]
        if args.dataset == "both"
        else [args.dataset]
    )
    for name in names:
        run_dataset(name)


if __name__ == "__main__":
    main()
