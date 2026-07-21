"""Causal-style analysis: does more racing improve the same athlete's seasonal WA jump?

Methods (per research plan):
  1. Within-athlete comparisons across seasons (athlete × event-group fixed effects)
  2. Exclude 1-result seasons (mechanical zero point jump)
  3. Control for first-day opening WA
  4. Within-season dose–response (marginal meet index vs cumulative jump)
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
POINT_JUMP_ROOT = PROJECT_ROOT / "Number_Of_Events_Question"
PYLIBS = PROJECT_ROOT / "Non_Relays_Findings" / "Sprints_Events_Counting" / ".pylibs"
if PYLIBS.exists():
    sys.path.insert(0, str(PYLIBS))

sys.path.insert(0, str(POINT_JUMP_ROOT))
from analyze_point_jump_by_competition_count import (  # noqa: E402
    AthleteResult,
    compute_point_jumps,
    load_combined_dataset,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIN_RESULTS = 2
MIN_PANEL_SEASONS = 2
MIN_DOSE_N = 10
BOOTSTRAP_DRAWS = 500
RNG = np.random.default_rng(42)

EVENT_GROUPS = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]


def load_point_jump_rows() -> list[dict]:
    path = POINT_JUMP_ROOT / "athlete_point_jumps_by_season.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def filter_multimeet(rows: list[dict]) -> list[dict]:
    return [r for r in rows if int(r["result_count"]) >= MIN_RESULTS]


def panel_unit_id(row: dict) -> str:
    return f"{row['athlete_id']}|{row['gender']}|{row['event_group']}"


def ols_with_se(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and homoskedastic standard errors."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    n, k = x.shape
    df = max(n - k, 1)
    sigma2 = float(resid @ resid) / df
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    return beta, se


def within_transform(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    demeaned = values.copy().astype(float)
    for gid in np.unique(groups):
        mask = groups == gid
        demeaned[mask] -= demeaned[mask].mean()
    return demeaned


def fe_regression(
    rows: list[dict],
    *,
    include_first_wa: bool,
    include_season_fe: bool,
) -> dict:
    """Athlete×event-group fixed effects via within transformation."""
    y = np.array([float(r["point_jump"]) for r in rows])
    x_count = np.array([float(r["result_count"]) for r in rows])
    groups = np.array([panel_unit_id(r) for r in rows])

    y_w = within_transform(y, groups)
    xw_parts = [within_transform(x_count, groups)]
    names = ["result_count"]

    if include_first_wa:
        x_open = np.array([float(r["first_wa"]) for r in rows])
        xw_parts.append(within_transform(x_open, groups))
        names.append("first_wa")

    if include_season_fe:
        for season in sorted({r["season"] for r in rows}):
            col = np.array([1.0 if r["season"] == season else 0.0 for r in rows])
            if season == sorted({r["season"] for r in rows})[0]:
                continue  # drop one season
            xw_parts.append(within_transform(col, groups))
            names.append(f"season_{season}")

    x_w = np.column_stack(xw_parts)
    x_w = np.column_stack([np.ones(len(rows)), x_w])
    names = ["const"] + names

    beta, se = ols_with_se(x_w, y_w)
    n_units = len(np.unique(groups))
    return {
        "names": names,
        "beta": beta,
        "se": se,
        "n_obs": len(rows),
        "n_units": n_units,
        "r2_within": 1.0 - float(((y_w - x_w @ beta) ** 2).sum() / (y_w**2).sum()),
    }


def bootstrap_fe_beta(rows: list[dict], draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float, float]:
    """Cluster bootstrap on panel units; return mean beta, 2.5%, 97.5% for result_count."""
    units: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        units[panel_unit_id(r)].append(r)
    unit_keys = list(units.keys())
    betas = []
    for _ in range(draws):
        sampled = RNG.choice(unit_keys, size=len(unit_keys), replace=True)
        sample_rows = [row for key in sampled for row in units[key]]
        if len({panel_unit_id(r) for r in sample_rows}) < 2:
            continue
        res = fe_regression(sample_rows, include_first_wa=True, include_season_fe=True)
        betas.append(float(res["beta"][1]))
    if not betas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(betas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_within_athlete_deltas(rows: list[dict]) -> list[dict]:
    """For units with 2+ seasons, compute season-pair deltas."""
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[panel_unit_id(r)].append(r)

    deltas = []
    for unit, seasons in by_unit.items():
        if len(seasons) < MIN_PANEL_SEASONS:
            continue
        seasons = sorted(seasons, key=lambda r: r["season"])
        for i in range(len(seasons)):
            for j in range(i + 1, len(seasons)):
                a, b = seasons[i], seasons[j]
                deltas.append(
                    {
                        "unit": unit,
                        "gender": a["gender"],
                        "event_group": a["event_group"],
                        "season_a": a["season"],
                        "season_b": b["season"],
                        "delta_result_count": int(b["result_count"]) - int(a["result_count"]),
                        "delta_point_jump": float(b["point_jump"]) - float(a["point_jump"]),
                        "delta_first_wa": float(b["first_wa"]) - float(a["first_wa"]),
                    }
                )
    return deltas


def delta_regression(deltas: list[dict]) -> dict:
    """Regress delta_point_jump on delta_result_count, controlling delta_first_wa."""
    if len(deltas) < 10:
        return {"n": len(deltas)}
    y = np.array([d["delta_point_jump"] for d in deltas])
    x = np.column_stack(
        [
            np.ones(len(deltas)),
            [d["delta_result_count"] for d in deltas],
            [d["delta_first_wa"] for d in deltas],
        ]
    )
    beta, se = ols_with_se(x, y)
    return {
        "n": len(deltas),
        "beta_result_count": float(beta[1]),
        "se_result_count": float(se[1]),
        "beta_first_wa": float(beta[2]),
        "se_first_wa": float(se[2]),
    }


def build_within_season_trajectories(athlete_results: list[AthleteResult]) -> list[dict]:
    """Cumulative jump after meet k (k>=1), using first-day WA as baseline."""
    grouped: dict[tuple[str, str, str, str], list[AthleteResult]] = defaultdict(list)
    for r in athlete_results:
        for group in EVENT_GROUPS:
            from analyze_point_jump_by_competition_count import event_allowed_in_group

            if event_allowed_in_group(r.running_event_id, group, r.gender):
                key = (r.athlete_id, r.gender, r.season, group)
                grouped[key].append(r)

    rows = []
    for (athlete_id, gender, season, group), results in grouped.items():
        by_result: dict[str, AthleteResult] = {}
        for r in results:
            by_result[r.result_id] = r
        unique = list(by_result.values())
        if len(unique) < MIN_RESULTS:
            continue
        unique.sort(key=lambda r: (r.start_date, r.meet_id, r.result_id))
        first_date = unique[0].start_date
        first_day = [r for r in unique if r.start_date == first_date]
        baseline = max(r.wa_points for r in first_day)

        running_max = baseline
        meet_idx = 0
        for r in unique:
            meet_idx += 1
            running_max = max(running_max, r.wa_points)
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "gender": gender,
                    "season": season,
                    "event_group": group,
                    "meet_index": meet_idx,
                    "cumulative_jump": round(running_max - baseline, 1),
                }
            )
    return rows


def aggregate_dose_response(trajectories: list[dict]) -> list[dict]:
    bins: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in trajectories:
        key = (row["gender"], row["event_group"], row["meet_index"])
        bins[key].append(row["cumulative_jump"])

    out = []
    for (gender, group, meet_index), values in sorted(bins.items()):
        if len(values) < MIN_DOSE_N:
            continue
        out.append(
            {
                "gender": gender,
                "event_group": group,
                "meet_index": meet_index,
                "athlete_meet_obs": len(values),
                "avg_cumulative_jump": round(float(np.mean(values)), 2),
            }
        )
    return out


def run_fe_by_group(rows: list[dict]) -> list[dict]:
    results = []
    for gender in ("Men", "Women"):
        for group in EVENT_GROUPS:
            sub = [r for r in rows if r["gender"] == gender and r["event_group"] == group]
            units = {panel_unit_id(r) for r in sub}
            if len(sub) < 20 or len(units) < 10:
                continue
            multi = [u for u in units if sum(1 for r in sub if panel_unit_id(r) == u) >= MIN_PANEL_SEASONS]
            sub_multi = [r for r in sub if panel_unit_id(r) in multi]
            if len(sub_multi) < 15:
                continue
            res = fe_regression(sub_multi, include_first_wa=True, include_season_fe=True)
            boot_mean, boot_lo, boot_hi = bootstrap_fe_beta(sub_multi)
            results.append(
                {
                    "gender": gender,
                    "event_group": group,
                    "model": "athlete_event_group_FE",
                    "beta_result_count": round(float(res["beta"][1]), 3),
                    "se_result_count": round(float(res["se"][1]), 3),
                    "beta_first_wa": round(float(res["beta"][2]), 3),
                    "bootstrap_ci_low": round(boot_lo, 3),
                    "bootstrap_ci_high": round(boot_hi, 3),
                    "n_obs": res["n_obs"],
                    "n_units": res["n_units"],
                    "r2_within": round(res["r2_within"], 4),
                }
            )
    return results


def plot_dose_response(aggregated: list[dict]) -> None:
    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for gender in ("Men", "Women"):
        for group in EVENT_GROUPS:
            rows = [r for r in aggregated if r["gender"] == gender and r["event_group"] == group]
            if not rows:
                continue
            rows.sort(key=lambda r: r["meet_index"])
            x = [r["meet_index"] for r in rows]
            y = [r["avg_cumulative_jump"] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, marker="o", color="#2E86AB", linewidth=2)
            ax.set_xlabel("Meet index in season (chronological)")
            ax.set_ylabel("Avg cumulative jump from first-day WA")
            ax.set_title(
                f"{gender} {group} — Within-season dose response\n"
                f"(≥{MIN_DOSE_N} athlete-meet observations per point; seasons with ≥{MIN_RESULTS} results)"
            )
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / f"dose_response_{gender.lower()}_{group.lower()}.png", dpi=150)
            plt.close()


def write_report(
    fe_global: dict,
    fe_by_group: list[dict],
    delta_stats: dict,
    pooled: dict,
    dose: list[dict],
    n_raw: int,
    n_filtered: int,
) -> None:
    lines = [
        "Causal Analysis — Competition Volume and Seasonal WA Improvement",
        "==============================================================",
        "",
        "Question: If we added more races for the same athlete, would they improve more?",
        "",
        "Methods applied:",
        f"  1. Within-athlete panel: athlete×event-group fixed effects across seasons",
        f"  2. Excluded seasons with only 1 result (mechanical point jump = 0)",
        f"  3. Controlled for first-day opening WA (within FE and in delta models)",
        "  4. Within-season dose–response: cumulative jump by chronological meet index",
        "",
        f"Raw athlete-season-event_group rows: {n_raw:,}",
        f"After excluding 1-result seasons: {n_filtered:,}",
        "",
        "Interpretation: Positive β for result_count means that when the SAME athlete",
        "races more in a season (relative to their own other seasons), their point jump",
        "is larger — after controlling opening WA and season. This is stronger than",
        "the descriptive plots but still not a randomized experiment.",
        "",
        "## Model A — Global athlete×event-group fixed effects",
        f"Observations: {fe_global['n_obs']:,} | Panel units: {fe_global['n_units']:,}",
        f"β(result_count): {fe_global['beta'][1]:.3f}  (SE {fe_global['se'][1]:.3f})",
        f"β(first_wa):     {fe_global['beta'][2]:.3f}  (SE {fe_global['se'][2]:.3f})",
        f"Within R²: {fe_global['r2_within']:.4f}",
        f"Cluster bootstrap 95% CI for β(result_count): "
        f"[{fe_global['boot_lo']:.3f}, {fe_global['boot_hi']:.3f}]",
        "",
        "Reading: Each additional result in a season is associated with",
        f"{fe_global['beta'][1]:.2f} more WA points of seasonal jump, holding athlete",
        "baseline (fixed effect) and opening mark constant within unit.",
        "",
        "## Model B — Pooled OLS (no athlete FE; opening WA + season controls)",
        f"β(result_count): {pooled['beta_result_count']:.3f}  (SE {pooled['se_result_count']:.3f})",
        f"n = {pooled['n']:,}",
        "",
        "## Model C — Paired within-athlete season deltas",
        f"Season pairs (same athlete×group, ≥{MIN_RESULTS} results each season): {delta_stats.get('n', 0):,}",
    ]
    if delta_stats.get("n", 0) >= 10:
        lines.extend(
            [
                f"β(Δ result_count → Δ point_jump): {delta_stats['beta_result_count']:.3f} "
                f"(SE {delta_stats['se_result_count']:.3f})",
                f"Control: Δ first_wa coefficient {delta_stats['beta_first_wa']:.3f}",
                "",
                "When an athlete adds more races in season B than season A, seasonal jump",
                f"changes by ~{delta_stats['beta_result_count']:.1f} WA points per additional race.",
            ]
        )
    lines.append("")
    lines.append("## Model D — By event group (athlete×group FE, bootstrap CI)")
    for row in fe_by_group:
        sig = "positive" if row["bootstrap_ci_low"] > 0 else "mixed/uncertain"
        lines.append(
            f"  {row['gender']} {row['event_group']}: β={row['beta_result_count']:.3f} "
            f"[{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}] "
            f"n={row['n_obs']} units={row['n_units']} ({sig})"
        )
    lines.extend(
        [
            "",
            "## Within-season dose–response (summary)",
            "Average cumulative jump from first-day WA by meet index (see plots/).",
        ]
    )
    for gender in ("Men", "Women"):
        for group in EVENT_GROUPS:
            sub = [r for r in dose if r["gender"] == gender and r["event_group"] == group]
            if len(sub) < 2:
                continue
            sub.sort(key=lambda r: r["meet_index"])
            gain = sub[-1]["avg_cumulative_jump"] - sub[0]["avg_cumulative_jump"]
            lines.append(
                f"  {gender} {group}: meet {sub[0]['meet_index']} → {sub[-1]['meet_index']} "
                f"avg cumulative jump +{gain:.1f} WA"
            )
    lines.extend(
        [
            "",
            "## Causal credibility assessment",
            "  Supports 'more races → more improvement' IF:",
            "    • β(result_count) > 0 in FE models (especially with CI excluding 0)",
            "    • Dose–response curves rise with meet index",
            "  Remaining threats:",
            "    • Coaches assign more races to athletes already trending up within season",
            "    • Season context (team, health, weather) not fully observed",
            "    • Relay assignments drive volume for non-primary events",
            "",
            "Verdict: See coefficients above — FE estimate is the best available answer",
            "in this dataset, framed as within-athlete associative evidence.",
        ]
    )
    (OUTPUT_ROOT / "causal_analysis_report.txt").write_text("\n".join(lines).rstrip() + "\n")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading point jump data...")
    raw = load_point_jump_rows()
    filtered = filter_multimeet(raw)
    print(f"  Filtered to ≥{MIN_RESULTS} results: {len(filtered):,} rows")

    panel_units = {panel_unit_id(r) for r in filtered}
    multi_season_units = {
        u for u in panel_units if sum(1 for r in filtered if panel_unit_id(r) == u) >= MIN_PANEL_SEASONS
    }
    panel_rows = [r for r in filtered if panel_unit_id(r) in multi_season_units]
    print(f"  Panel (≥{MIN_PANEL_SEASONS} seasons per athlete×group): {len(panel_rows):,} rows, {len(multi_season_units):,} units")

    print("Model A — athlete×event-group fixed effects...")
    fe_global = fe_regression(panel_rows, include_first_wa=True, include_season_fe=True)
    boot_mean, boot_lo, boot_hi = bootstrap_fe_beta(panel_rows)
    fe_global["boot_lo"] = boot_lo
    fe_global["boot_hi"] = boot_hi

    print("Model B — pooled OLS with controls...")
    y = np.array([float(r["point_jump"]) for r in filtered])
    seasons = sorted({r["season"] for r in filtered})
    x_cols = [
        np.ones(len(filtered)),
        np.array([float(r["result_count"]) for r in filtered]),
        np.array([float(r["first_wa"]) for r in filtered]),
    ]
    names = ["const", "result_count", "first_wa"]
    for season in seasons[1:]:
        x_cols.append(np.array([1.0 if r["season"] == season else 0.0 for r in filtered]))
        names.append(f"season_{season}")
    for group in EVENT_GROUPS[1:]:
        x_cols.append(np.array([1.0 if r["event_group"] == group else 0.0 for r in filtered]))
        names.append(f"group_{group}")
    x = np.column_stack(x_cols)
    beta, se = ols_with_se(x, y)
    pooled = {
        "n": len(filtered),
        "beta_result_count": float(beta[1]),
        "se_result_count": float(se[1]),
    }

    print("Model C — paired season deltas...")
    deltas = paired_within_athlete_deltas(filtered)
    delta_stats = delta_regression(deltas)
    save_csv(deltas, OUTPUT_ROOT / "within_athlete_season_deltas.csv")

    print("Model D — by event group...")
    fe_by_group = run_fe_by_group(filtered)
    save_csv(fe_by_group, OUTPUT_ROOT / "fe_regression_by_event_group.csv")

    print("Within-season dose–response...")
    _, athlete_results = load_combined_dataset()
    trajectories = build_within_season_trajectories(athlete_results)
    dose = aggregate_dose_response(trajectories)
    save_csv(dose, OUTPUT_ROOT / "within_season_dose_response.csv")
    plot_dose_response(dose)

    save_csv(
        [
            {
                "model": "global_athlete_event_group_FE",
                "beta_result_count": round(float(fe_global["beta"][1]), 4),
                "se_result_count": round(float(fe_global["se"][1]), 4),
                "beta_first_wa": round(float(fe_global["beta"][2]), 4),
                "bootstrap_ci_low": round(boot_lo, 4),
                "bootstrap_ci_high": round(boot_hi, 4),
                "n_obs": fe_global["n_obs"],
                "n_units": fe_global["n_units"],
            }
        ],
        OUTPUT_ROOT / "fe_regression_global.csv",
    )

    write_report(fe_global, fe_by_group, delta_stats, pooled, dose, len(raw), len(filtered))
    print(f"Done. Outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
