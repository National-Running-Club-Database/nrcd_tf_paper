"""Inferential statistics layer for the causal competition-volume analysis.

Adds explicit hypothesis tests and bootstrap confidence intervals on top of
the athlete×event-group fixed-effects (FE) models in
`analyze_causal_competition_volume.py`:

  1. Paired Wilcoxon signed-rank test: within the same athlete×event-group
     panel unit, is the WA point jump larger in the season where the athlete
     raced MORE than in the season where they raced fewer times?
  2. Spearman rank correlation between the within-athlete season-to-season
     change in result count (delta_result_count) and the change in point
     jump (delta_point_jump) — a distribution-free complement to the FE
     regression's beta coefficient.
  3. Cluster (athlete-unit) bootstrap 95% CIs for every test statistic above,
     plus a cluster-bootstrap CI band around the within-season dose-response
     curves.

Design / causal-inference framing
----------------------------------
All tests here operate on WITHIN-ATHLETE (or within athlete×event-group)
comparisons, consistent with the fixed-effects design in
`analyze_causal_competition_volume.py`. They should be read as evidence for
a within-athlete ASSOCIATION / dose-response pattern between competition
volume and seasonal improvement, controlling for the athlete's own baseline.
They are NOT randomized-experiment evidence: coaches may assign more races
to athletes who are already trending up, and season-level confounders
(health, training load, weather, team context) are not observed. Do not
upgrade the language of these results beyond "within-athlete association"
or "dose-response under fixed effects" when reporting findings.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

# Writable cache for headless / sandboxed runs (avoid unwritable ~/.matplotlib)
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT_DIR = ROOT / "research"
PLOT_DIR = ROOT / "plots"
DELTAS_CSV = ROOT / "within_athlete_season_deltas.csv"
FE_BY_GROUP_CSV = ROOT / "fe_regression_by_event_group.csv"

RNG = np.random.default_rng(2026)
N_BOOT = 2000
N_BOOT_DOSE = 400  # nested bootstrap over units x meet-index bins; kept smaller for speed
ALPHA = 0.05
MIN_GROUP_N = 10  # minimum paired observations to report a by-group test

C_PRIMARY = "#2E86AB"
C_ACCENT = "#C73E1D"
C_NEUTRAL = "#2C3E50"
C_GRID = "#D5D8DC"
C_BAND = "#2E86AB"


# --------------------------------------------------------------------------
# Bootstrap helpers
# --------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = N_BOOT, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar statistic (IID resampling)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat_fn(values))
    if len(values) == 1:
        return point, point, point
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[RNG.integers(0, n, size=n)]
        boots[i] = float(stat_fn(sample))
    return point, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def cluster_bootstrap_ci(
    rows: list[dict], unit_key: str, stat_fn, n_boot: int = N_BOOT, alpha: float = ALPHA
) -> tuple[float, float, float]:
    """Percentile bootstrap CI resampling PANEL UNITS (not rows) with replacement.

    Correct for panels where a unit (athlete x event group) can contribute
    more than one season-pair row; resampling rows directly would understate
    variance because rows from the same athlete are not independent.
    """
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_unit[r[unit_key]].append(r)
    unit_keys = list(by_unit.keys())
    if not unit_keys:
        return float("nan"), float("nan"), float("nan")
    point = float(stat_fn(rows))
    if len(unit_keys) == 1:
        return point, point, point
    n_units = len(unit_keys)
    boots = []
    for _ in range(n_boot):
        sampled = RNG.choice(unit_keys, size=n_units, replace=True)
        sample_rows = [row for key in sampled for row in by_unit[key]]
        val = stat_fn(sample_rows)
        if val is not None and np.isfinite(val):
            boots.append(float(val))
    if not boots:
        return point, float("nan"), float("nan")
    arr = np.array(boots)
    return point, float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


# --------------------------------------------------------------------------
# 1. Paired Wilcoxon: higher-volume season vs lower-volume season point jump
# --------------------------------------------------------------------------


def _paired_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """For each season-pair row with a nonzero delta_result_count, return the
    (more-races-season point jump, fewer-races-season point jump) arrays."""
    more, fewer = [], []
    for r in rows:
        d = int(r["delta_result_count"])
        if d == 0:
            continue  # tie in race count carries no volume contrast
        if d > 0:
            more.append(float(r["point_jump_b"]))
            fewer.append(float(r["point_jump_a"]))
        else:
            more.append(float(r["point_jump_a"]))
            fewer.append(float(r["point_jump_b"]))
    return np.array(more), np.array(fewer)


def wilcoxon_volume_test(rows: list[dict]) -> dict:
    """H0: median(point_jump_more_races - point_jump_fewer_races) = 0
    H1: median difference > 0 (more racing associated with a bigger within-athlete jump)."""
    more, fewer = _paired_arrays(rows)
    n = len(more)
    out = {
        "n_pairs": n,
        "n_units": len({r["unit"] for r in rows}),
        "median_more_races_jump": float("nan"),
        "median_fewer_races_jump": float("nan"),
        "median_diff": float("nan"),
        "median_diff_ci_lo": float("nan"),
        "median_diff_ci_hi": float("nan"),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p_greater": float("nan"),
        "significant_005": False,
    }
    if n < 5:
        return out
    diff = more - fewer
    out["median_more_races_jump"] = float(np.median(more))
    out["median_fewer_races_jump"] = float(np.median(fewer))
    out["median_diff"] = float(np.median(diff))
    try:
        stat_, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="greater")
        out["wilcoxon_stat"] = float(stat_)
        out["wilcoxon_p_greater"] = float(p)
        out["significant_005"] = bool(p < ALPHA)
    except ValueError:
        pass

    def stat_fn(sample_rows: list[dict]) -> float | None:
        m, f = _paired_arrays(sample_rows)
        if len(m) == 0:
            return None
        return float(np.median(m - f))

    _, lo, hi = cluster_bootstrap_ci(rows, "unit", stat_fn)
    out["median_diff_ci_lo"] = lo
    out["median_diff_ci_hi"] = hi
    return out


# --------------------------------------------------------------------------
# 2. Spearman correlation on within-athlete deltas (dose-response, rank-based)
# --------------------------------------------------------------------------


def spearman_delta_test(rows: list[dict]) -> dict:
    """H0: rho(delta_result_count, delta_point_jump) = 0 across season pairs."""
    x = np.array([float(r["delta_result_count"]) for r in rows])
    y = np.array([float(r["delta_point_jump"]) for r in rows])
    n = len(x)
    out = {
        "n_pairs": n,
        "n_units": len({r["unit"] for r in rows}),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "rho_ci_lo": float("nan"),
        "rho_ci_hi": float("nan"),
        "significant_005": False,
    }
    if n < 5 or np.std(x) == 0 or np.std(y) == 0:
        return out
    rho, p = stats.spearmanr(x, y)
    out["spearman_rho"] = float(rho)
    out["spearman_p"] = float(p)
    out["significant_005"] = bool(p < ALPHA)

    def stat_fn(sample_rows: list[dict]) -> float | None:
        xs = np.array([float(r["delta_result_count"]) for r in sample_rows])
        ys = np.array([float(r["delta_point_jump"]) for r in sample_rows])
        if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
            return None
        r_, _ = stats.spearmanr(xs, ys)
        return float(r_)

    _, lo, hi = cluster_bootstrap_ci(rows, "unit", stat_fn)
    out["rho_ci_lo"] = lo
    out["rho_ci_hi"] = hi
    return out


def tests_by_group(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    wilcoxon_rows, spearman_rows = [], []
    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance", "Hurdles", "Jumps", "Throws"):
            sub = [r for r in rows if r["gender"] == gender and r["event_group"] == group]
            if len(sub) < MIN_GROUP_N:
                continue
            w = wilcoxon_volume_test(sub)
            w.update({"gender": gender, "event_group": group})
            wilcoxon_rows.append(w)
            s = spearman_delta_test(sub)
            s.update({"gender": gender, "event_group": group})
            spearman_rows.append(s)
    return wilcoxon_rows, spearman_rows


# --------------------------------------------------------------------------
# 3. Dose-response cluster-bootstrap CI band
# --------------------------------------------------------------------------


def cluster_bootstrap_dose_response(
    trajectories: list[dict], min_obs: int = 10, n_boot: int = N_BOOT_DOSE
) -> list[dict]:
    """Cluster-bootstrap 95% CI for avg cumulative jump per (gender, group, meet_index).

    Resamples athlete-season-event_group UNITS (not individual rows) so that
    the same athlete's repeated meet-index observations are resampled together.
    """
    units: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    unit_group: dict[tuple, tuple[str, str]] = {}
    for row in trajectories:
        key = (row["athlete_id"], row["gender"], row["season"], row["event_group"])
        units[key].append((row["meet_index"], row["cumulative_jump"]))
        unit_group[key] = (row["gender"], row["event_group"])

    group_units: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for key, meta in unit_group.items():
        group_units[meta].append(key)

    out_rows = []
    for (gender, group), unit_keys in group_units.items():
        n_units = len(unit_keys)
        # Point estimate + n per meet_index from the real (non-bootstrapped) sample
        real_bins: dict[int, list[float]] = defaultdict(list)
        for key in unit_keys:
            for meet_index, val in units[key]:
                real_bins[meet_index].append(val)
        valid_bins = {mi for mi, vals in real_bins.items() if len(vals) >= min_obs}
        if not valid_bins:
            continue

        draws: dict[int, list[float]] = defaultdict(list)
        for _ in range(n_boot):
            sample_idx = RNG.integers(0, n_units, size=n_units)
            bucket: dict[int, list[float]] = defaultdict(list)
            for idx in sample_idx:
                for meet_index, val in units[unit_keys[idx]]:
                    bucket[meet_index].append(val)
            for meet_index in valid_bins:
                vals = bucket.get(meet_index)
                if vals:
                    draws[meet_index].append(float(np.mean(vals)))

        for meet_index in sorted(valid_bins):
            means = np.array(draws.get(meet_index, []))
            avg = float(np.mean(real_bins[meet_index]))
            if len(means) >= 20:
                lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
            else:
                lo, hi = float("nan"), float("nan")
            out_rows.append(
                {
                    "gender": gender,
                    "event_group": group,
                    "meet_index": meet_index,
                    "n_athlete_meet_obs": len(real_bins[meet_index]),
                    "n_units_total": n_units,
                    "avg_cumulative_jump": round(avg, 2),
                    "bootstrap_ci_low": round(lo, 2),
                    "bootstrap_ci_high": round(hi, 2),
                }
            )
    return out_rows


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=C_NEUTRAL, pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, color=C_GRID, zorder=0)


def plot_dose_response_with_ci(dose_ci: list[dict]) -> None:
    """Regenerate the per-gender/event-group dose-response figures with a
    shaded 95% cluster-bootstrap CI band, explicit units, and sample size."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance", "Hurdles", "Jumps", "Throws"):
            rows = [r for r in dose_ci if r["gender"] == gender and r["event_group"] == group]
            if not rows:
                continue
            rows.sort(key=lambda r: r["meet_index"])
            x = [r["meet_index"] for r in rows]
            y = [r["avg_cumulative_jump"] for r in rows]
            lo = [r["bootstrap_ci_low"] for r in rows]
            hi = [r["bootstrap_ci_high"] for r in rows]
            n_units = rows[0]["n_units_total"]

            fig, ax = plt.subplots(figsize=(8, 5))
            has_ci = all(np.isfinite(lo)) and all(np.isfinite(hi))
            if has_ci:
                ax.fill_between(x, lo, hi, color=C_BAND, alpha=0.18, zorder=1, label="95% cluster-bootstrap CI")
            ax.plot(x, y, marker="o", color=C_PRIMARY, linewidth=2, zorder=3, label="Mean cumulative jump")
            for xi, r in zip(x, rows):
                ax.annotate(
                    f"n={r['n_athlete_meet_obs']}",
                    (xi, r["avg_cumulative_jump"]),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=7.5,
                    color=C_NEUTRAL,
                )
            _style_axes(
                ax,
                f"{gender} {group} — Within-season dose response\n"
                f"Mean cumulative WA-point jump from first-day mark ({n_units} athlete×season×group panel units)",
                "Meet index in season (chronological order)",
                "Cumulative WA-point jump from first-day WA score (points)",
            )
            ax.legend(fontsize=8, frameon=False, loc="upper left")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"dose_response_{gender.lower()}_{group.lower()}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_fe_forest(fe_by_group: list[dict]) -> None:
    """Forest plot: FE beta(result_count) with 95% cluster-bootstrap CI, by gender x event group."""
    if not fe_by_group:
        return
    rows = sorted(fe_by_group, key=lambda r: (r["gender"], -float(r["beta_result_count"])))
    labels = [f"{r['gender']} {r['event_group']} (n={r['n_obs']}, units={r['n_units']})" for r in rows]
    betas = [float(r["beta_result_count"]) for r in rows]
    lo = [float(r["bootstrap_ci_low"]) for r in rows]
    hi = [float(r["bootstrap_ci_high"]) for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.5 * len(rows) + 1.5)))
    colors = [C_PRIMARY if lo_i > 0 else (C_ACCENT if hi_i < 0 else "#95A5A6") for lo_i, hi_i in zip(lo, hi)]
    ax.errorbar(
        betas,
        y,
        xerr=[np.array(betas) - np.array(lo), np.array(hi) - np.array(betas)],
        fmt="none",
        ecolor=C_NEUTRAL,
        elinewidth=1.2,
        capsize=3,
        zorder=2,
    )
    ax.scatter(betas, y, color=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0, color=C_NEUTRAL, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    _style_axes(
        ax,
        "Athlete×event-group FE: effect of +1 result on seasonal WA-point jump\n"
        "95% cluster bootstrap CI (blue=CI>0, red=CI<0, grey=CI includes 0)",
        "β(result_count): WA points of seasonal jump per additional result",
        "",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fe_beta_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_scatter(rows: list[dict], spearman_overall: dict) -> None:
    """Scatter of delta_result_count vs delta_point_jump with an OLS trend line
    and the Spearman rho/p annotated (dose-response on within-athlete deltas)."""
    x = np.array([float(r["delta_result_count"]) for r in rows])
    y = np.array([float(r["delta_point_jump"]) for r in rows])
    if len(x) < 5:
        return
    jitter = RNG.uniform(-0.15, 0.15, size=len(x))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x + jitter, y, s=16, alpha=0.35, color=C_PRIMARY, edgecolors="none", zorder=2)

    # Simple robust trend line via OLS on (x, y) for visual reference only
    if np.std(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color=C_ACCENT, linewidth=2, zorder=3, label=f"OLS trend (slope={b1:.2f})")

    rho = spearman_overall.get("spearman_rho", float("nan"))
    p = spearman_overall.get("spearman_p", float("nan"))
    n = spearman_overall.get("n_pairs", len(x))
    ax.text(
        0.02,
        0.97,
        f"Spearman ρ = {rho:.3f}, p = {p:.2g}, n = {n:,} season pairs",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        color=C_NEUTRAL,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=C_GRID),
    )
    ax.axhline(0, color=C_GRID, linewidth=0.8, zorder=1)
    ax.axvline(0, color=C_GRID, linewidth=0.8, zorder=1)
    _style_axes(
        ax,
        "Within-athlete season-to-season change: race volume vs WA-point jump\n"
        "Each point = one athlete×event-group pair of seasons (x jittered for visibility)",
        "Δ result count (season B − season A)",
        "Δ point jump (WA points, season B − season A)",
    )
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "delta_dose_response_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Report + I/O
# --------------------------------------------------------------------------


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}"


def write_inferential_report(
    wilcoxon_overall: dict,
    spearman_overall: dict,
    wilcoxon_by_group: list[dict],
    spearman_by_group: list[dict],
    dose_ci: list[dict],
    fe_by_group: list[dict],
) -> None:
    lines = [
        "Causal Analysis — Inferential Statistics Report",
        "=================================================",
        "",
        "Scope: hypothesis tests and bootstrap confidence intervals layered on top",
        "of the athlete x event-group fixed-effects (FE) models in",
        "causal_analysis_report.txt. All tests below use WITHIN-ATHLETE comparisons",
        "(same athlete x event group, across two of their own seasons).",
        "",
        "Design / causal framing (read before citing these numbers)",
        "------------------------------------------------------------",
        "  - These are within-athlete ASSOCIATIONS and DOSE-RESPONSE patterns under a",
        "    fixed-effects design that controls for athlete baseline (fixed effect) and",
        "    opening-season WA score.",
        "  - They are NOT evidence from a randomized experiment. Coaches may assign",
        "    additional competitions to athletes who are already trending up within a",
        "    season; unobserved season-level factors (health, training load, weather,",
        "    team context) are not controlled for.",
        "  - Do not describe results as proof that racing more CAUSES improvement for a",
        "    given athlete beyond what this FE design supports; use language such as",
        "    'within-athlete association' or 'dose-response under fixed effects'.",
        "",
        "Bootstrap: percentile method, seed=2026, cluster-resampled by athlete x",
        "event-group panel UNIT (not by row) so repeated observations from the same",
        "athlete do not artificially shrink the interval.",
        f"  General statistics: {N_BOOT} resamples.",
        f"  Dose-response curves: {N_BOOT_DOSE} resamples (nested over nine meet-index bins x two genders x five groups).",
        "",
        "## 1. Paired Wilcoxon signed-rank test — higher-volume vs lower-volume season",
        "-------------------------------------------------------------------------------",
        "H0: median(point jump in the season with MORE results - point jump in the",
        "    season with FEWER results) = 0, for the same athlete x event group.",
        "H1 (one-sided): median difference > 0 (racing more in a season associates",
        "    with a bigger seasonal WA-point jump for that same athlete).",
        "",
        f"Overall: n={wilcoxon_overall['n_pairs']:,} season pairs "
        f"({wilcoxon_overall['n_units']:,} athlete x event-group units)",
        f"  Median jump, higher-volume season: {wilcoxon_overall['median_more_races_jump']:.1f} WA points",
        f"  Median jump, lower-volume season:  {wilcoxon_overall['median_fewer_races_jump']:.1f} WA points",
        f"  Median within-pair difference: {wilcoxon_overall['median_diff']:.1f} WA points "
        f"[95% CI {wilcoxon_overall['median_diff_ci_lo']:.1f}, {wilcoxon_overall['median_diff_ci_hi']:.1f}]",
        f"  Wilcoxon signed-rank statistic: {wilcoxon_overall['wilcoxon_stat']:.1f}, "
        f"p (one-sided, greater) = {_fmt_p(wilcoxon_overall['wilcoxon_p_greater'])}",
        f"  Significant at alpha=0.05: {wilcoxon_overall['significant_005']}",
        "",
        "By gender x event group (n >= 10 season pairs):",
    ]
    for r in wilcoxon_by_group:
        lines.append(
            f"  {r['gender']:6s} {r['event_group']:9s} n={r['n_pairs']:4d} units={r['n_units']:4d}  "
            f"median diff={r['median_diff']:+6.1f} [{r['median_diff_ci_lo']:+6.1f}, {r['median_diff_ci_hi']:+6.1f}]  "
            f"p={_fmt_p(r['wilcoxon_p_greater'])}  sig={r['significant_005']}"
        )
    lines.extend(
        [
            "",
            "## 2. Spearman correlation on within-athlete deltas (dose-response)",
            "---------------------------------------------------------------------",
            "H0: rho(delta result_count, delta point_jump) = 0 across season pairs.",
            "H1 (two-sided): rho != 0.",
            "",
            f"Overall: n={spearman_overall['n_pairs']:,} season pairs "
            f"({spearman_overall['n_units']:,} athlete x event-group units)",
            f"  Spearman rho = {spearman_overall['spearman_rho']:.3f} "
            f"[95% CI {spearman_overall['rho_ci_lo']:.3f}, {spearman_overall['rho_ci_hi']:.3f}]",
            f"  p (two-sided) = {_fmt_p(spearman_overall['spearman_p'])}, "
            f"significant at alpha=0.05: {spearman_overall['significant_005']}",
            "",
            "By gender x event group (n >= 10 season pairs):",
        ]
    )
    for r in spearman_by_group:
        lines.append(
            f"  {r['gender']:6s} {r['event_group']:9s} n={r['n_pairs']:4d} units={r['n_units']:4d}  "
            f"rho={r['spearman_rho']:+.3f} [{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}]  "
            f"p={_fmt_p(r['spearman_p'])}  sig={r['significant_005']}"
        )

    lines.extend(
        [
            "",
            "## 3. FE regression coefficients recap (from analyze_causal_competition_volume.py)",
            "--------------------------------------------------------------------------------------",
            "beta(result_count): WA points of seasonal jump per +1 result, holding athlete FE,",
            "opening WA, and season constant. 95% CI is the cluster bootstrap from Model D.",
            "",
        ]
    )
    for r in fe_by_group:
        ci_excludes_zero = float(r["bootstrap_ci_low"]) > 0 or float(r["bootstrap_ci_high"]) < 0
        lines.append(
            f"  {r['gender']:6s} {r['event_group']:9s} beta={float(r['beta_result_count']):+6.2f}  "
            f"[{float(r['bootstrap_ci_low']):+6.2f}, {float(r['bootstrap_ci_high']):+6.2f}]  "
            f"n={r['n_obs']:4} units={r['n_units']:4}  "
            f"{'CI excludes 0' if ci_excludes_zero else 'CI includes 0'}"
        )

    lines.extend(
        [
            "",
            "## 4. Within-season dose-response — cluster-bootstrap 95% CI",
            "----------------------------------------------------------------",
            "Mean cumulative WA-point jump from first-day mark by chronological meet index,",
            "with a 95% CI from resampling athlete x season x event-group panel units.",
            "See dose_response_bootstrap_ci.csv and plots/dose_response_*.png.",
            "",
        ]
    )
    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance", "Hurdles", "Jumps", "Throws"):
            sub = sorted(
                [r for r in dose_ci if r["gender"] == gender and r["event_group"] == group],
                key=lambda r: r["meet_index"],
            )
            if len(sub) < 2:
                continue
            first, last = sub[0], sub[-1]
            lines.append(
                f"  {gender:6s} {group:9s} meet {first['meet_index']:>2} -> {last['meet_index']:>2}: "
                f"{first['avg_cumulative_jump']:6.1f} [{first['bootstrap_ci_low']:6.1f}, {first['bootstrap_ci_high']:6.1f}] "
                f"-> {last['avg_cumulative_jump']:6.1f} [{last['bootstrap_ci_low']:6.1f}, {last['bootstrap_ci_high']:6.1f}] WA points"
            )

    lines.extend(
        [
            "",
            "## Bottom line",
            "--------------",
            "The paired Wilcoxon test and the delta-level Spearman correlation both test",
            "the SAME within-athlete dose-response question as the FE regression (Model A/D)",
            "but make no linearity assumption. Where all three agree (positive FE beta with",
            "a CI excluding 0, a significant one-sided Wilcoxon test, and a significant",
            "positive Spearman rho with a CI excluding 0), the within-athlete association",
            "between competition volume and seasonal WA improvement is well supported by",
            "this dataset -- as an associative/dose-response finding under fixed effects,",
            "not as a randomized-experiment causal claim.",
            "",
        ]
    )
    (OUT_DIR / "inferential_report.txt").write_text("\n".join(lines).rstrip() + "\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _load_deltas_csv() -> list[dict]:
    with open(DELTAS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("delta_result_count", "result_count_a", "result_count_b"):
            if k in r:
                r[k] = int(r[k])
        for k in ("delta_point_jump", "delta_first_wa", "point_jump_a", "point_jump_b"):
            if k in r:
                r[k] = float(r[k])
    return rows


def _load_fe_by_group_csv() -> list[dict]:
    with open(FE_BY_GROUP_CSV, newline="") as f:
        return list(csv.DictReader(f))


def run_research_stats(
    deltas: list[dict] | None = None,
    trajectories: list[dict] | None = None,
    fe_by_group: list[dict] | None = None,
) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if deltas is None:
        deltas = _load_deltas_csv()
    if fe_by_group is None and FE_BY_GROUP_CSV.exists():
        fe_by_group = _load_fe_by_group_csv()
    fe_by_group = fe_by_group or []

    if not any("point_jump_a" in r for r in deltas):
        raise SystemExit(
            "within_athlete_season_deltas.csv is missing point_jump_a/b columns; "
            "re-run analyze_causal_competition_volume.py to regenerate it with the "
            "extended schema before running research_stats standalone."
        )

    if trajectories is None:
        # Lazy import + recompute only when running this module standalone.
        from analyze_causal_competition_volume import build_within_season_trajectories

        _, athlete_results = load_combined_dataset_lazy()
        trajectories = build_within_season_trajectories(athlete_results)

    print("Running paired Wilcoxon + Spearman tests on within-athlete deltas...")
    wilcoxon_overall = wilcoxon_volume_test(deltas)
    spearman_overall = spearman_delta_test(deltas)
    wilcoxon_by_group, spearman_by_group = tests_by_group(deltas)

    print("Cluster-bootstrapping the within-season dose-response curves...")
    dose_ci = cluster_bootstrap_dose_response(trajectories)

    paths = {
        "wilcoxon_overall": OUT_DIR / "wilcoxon_volume_test_overall.csv",
        "wilcoxon_by_group": OUT_DIR / "wilcoxon_volume_test_by_group.csv",
        "spearman_overall": OUT_DIR / "spearman_delta_test_overall.csv",
        "spearman_by_group": OUT_DIR / "spearman_delta_test_by_group.csv",
        "dose_response_ci": OUT_DIR / "dose_response_bootstrap_ci.csv",
        "fe_by_group": OUT_DIR / "fe_regression_by_event_group.csv",
        "report": OUT_DIR / "inferential_report.txt",
    }
    save_csv([wilcoxon_overall], paths["wilcoxon_overall"])
    save_csv(wilcoxon_by_group, paths["wilcoxon_by_group"])
    save_csv([spearman_overall], paths["spearman_overall"])
    save_csv(spearman_by_group, paths["spearman_by_group"])
    save_csv(dose_ci, paths["dose_response_ci"])
    save_csv(fe_by_group, paths["fe_by_group"])  # copy alongside the other research/ CSVs

    write_inferential_report(
        wilcoxon_overall, spearman_overall, wilcoxon_by_group, spearman_by_group, dose_ci, fe_by_group
    )

    print("Regenerating research figures...")
    plot_dose_response_with_ci(dose_ci)
    plot_fe_forest(fe_by_group)
    plot_delta_scatter(deltas, spearman_overall)

    for label, p in paths.items():
        if p.exists():
            print(f"Wrote {p}")
    print(f"\nOverall Wilcoxon p={_fmt_p(wilcoxon_overall['wilcoxon_p_greater'])}, "
          f"Spearman rho={spearman_overall['spearman_rho']:.3f} (p={_fmt_p(spearman_overall['spearman_p'])})")
    return paths


def load_combined_dataset_lazy():
    from analyze_point_jump_by_competition_count import load_combined_dataset

    return load_combined_dataset()


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
