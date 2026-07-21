"""Research-grade statistical tests and figures for time-model validation.

Primary cross-event metric is absolute percent error (seconds are not
commensurate across 100m–5000m). Absolute-second metrics are reported
within event pairs / groups only.

Inferential procedures
----------------------
  • Bootstrap 95% CIs for MedAE and median |%| error
  • Wilcoxon signed-rank test of signed prediction bias (H0: median = 0)
  • Paired Wilcoxon: feature route vs pooled on |error| and |%| error
  • Spearman ρ (pred vs actual) with two-sided p-value
  • Clopper–Pearson 95% CI for within-tolerance rate
"""

from __future__ import annotations

import os
from pathlib import Path

# Writable cache for headless / sandboxed runs
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "output" / "model_validation" / "predictions.csv"
OUT_DIR = ROOT / "output" / "model_validation" / "research"
PLOT_DIR = OUT_DIR / "plots"

RNG = np.random.default_rng(2026)
N_BOOT = 2000
ALPHA = 0.05

# Research-safe palette (print-friendly, no purple gradient slop)
C_FEATURE = "#1B4F72"
C_POOLED = "#7B241C"
C_NEUTRAL = "#2C3E50"
C_GRID = "#D5D8DC"


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    rng: np.random.Generator = RNG,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a scalar statistic via percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat_fn(values))
    if len(values) == 1:
        return (point, point, point)
    boots = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = float(stat_fn(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Exact binomial CI for a rate."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    rate = k / n
    lo = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return rate, lo, hi


def wilcoxon_bias(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    out = {
        "n": int(len(errors)),
        "median_error": float(np.median(errors)) if len(errors) else float("nan"),
        "mean_error": float(np.mean(errors)) if len(errors) else float("nan"),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p": float("nan"),
        "bias_significant_005": False,
    }
    if len(errors) < 5:
        return out
    # Drop exact zeros for Wilcoxon (scipy handles via zero_method)
    try:
        stat, p = stats.wilcoxon(errors, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_stat"] = float(stat)
        out["wilcoxon_p"] = float(p)
        out["bias_significant_005"] = bool(p < ALPHA)
    except ValueError:
        pass
    return out


def spearman_pred_actual(pred: pd.DataFrame) -> dict:
    x = pred["to_time_pred"].to_numpy(dtype=float)
    y = pred["to_time_actual"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out = {"n": int(len(x)), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    if len(x) < 3:
        return out
    rho, p = stats.spearmanr(x, y)
    out["spearman_rho"] = float(rho)
    out["spearman_p"] = float(p)
    return out


def slice_metrics(g: pd.DataFrame) -> dict:
    abs_err = g["abs_error_sec"].to_numpy(dtype=float)
    abs_pct = g["abs_pct_error"].to_numpy(dtype=float)
    signed = g["error_sec"].to_numpy(dtype=float)

    medae, medae_lo, medae_hi = bootstrap_ci(abs_err, np.median)
    mape_med, mape_lo, mape_hi = bootstrap_ci(abs_pct, np.median)
    mae, mae_lo, mae_hi = bootstrap_ci(abs_err, np.mean)

    k = int(g["within_tolerance"].sum())
    n = int(len(g))
    tol_rate, tol_lo, tol_hi = clopper_pearson(k, n)

    bias = wilcoxon_bias(signed)
    corr = spearman_pred_actual(g)

    return {
        "n": n,
        "medae": medae,
        "medae_ci_lo": medae_lo,
        "medae_ci_hi": medae_hi,
        "mae": mae,
        "mae_ci_lo": mae_lo,
        "mae_ci_hi": mae_hi,
        "median_abs_pct_error": mape_med,
        "median_abs_pct_error_ci_lo": mape_lo,
        "median_abs_pct_error_ci_hi": mape_hi,
        "within_tol_rate": tol_rate,
        "within_tol_ci_lo": tol_lo,
        "within_tol_ci_hi": tol_hi,
        "within_tol_k": k,
        **{f"bias_{k}": v for k, v in bias.items() if k != "n"},
        **corr,
    }


def overall_inferential(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, route_kind), g in pred.groupby(["mode", "route_kind"]):
        st = slice_metrics(g)
        st.update({"mode": mode, "route_kind": route_kind, "unit": "mixed_events_seconds_caution"})
        rows.append(st)
    # Event-group stratified (absolute seconds more interpretable)
    for (mode, route_kind, eg), g in pred.groupby(["mode", "route_kind", "event_group"]):
        st = slice_metrics(g)
        st.update(
            {
                "mode": mode,
                "route_kind": route_kind,
                "event_group": eg,
                "unit": "seconds_within_group",
            }
        )
        rows.append(st)
    return pd.DataFrame(rows)


def by_pair_inferential(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(["mode", "band", "from_event", "to_event", "route_kind"]):
        st = slice_metrics(g)
        st.update(
            {
                "mode": keys[0],
                "band": keys[1],
                "from_event": keys[2],
                "to_event": keys[3],
                "route_kind": keys[4],
                "event_group": g["event_group"].iloc[0],
                "pair": f"{keys[2]}→{keys[3]}",
            }
        )
        rows.append(st)
    return pd.DataFrame(rows).sort_values(
        ["mode", "band", "n", "median_abs_pct_error"],
        ascending=[True, True, False, True],
    )


def feature_vs_pooled_tests(pred: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of feature vs pooled on the same athlete-pair rows."""
    key_cols = [
        "mode",
        "band",
        "event_group",
        "from_event",
        "to_event",
        "college",
        "athlete",
        "season_year",
    ]
    value_cols = ["abs_error_sec", "abs_pct_error", "error_sec"]
    feat = pred[pred["route_kind"] == "feature"][key_cols + value_cols].rename(
        columns={
            "abs_error_sec": "abs_err_feature",
            "abs_pct_error": "abs_pct_feature",
            "error_sec": "err_feature",
        }
    )
    pool = pred[pred["route_kind"] == "pooled"][key_cols + value_cols].rename(
        columns={
            "abs_error_sec": "abs_err_pooled",
            "abs_pct_error": "abs_pct_pooled",
            "error_sec": "err_pooled",
        }
    )
    m = feat.merge(pool, on=key_cols, how="inner")
    rows = []

    def _one(label: str, sub: pd.DataFrame) -> dict:
        d_abs = (sub["abs_err_feature"] - sub["abs_err_pooled"]).to_numpy(dtype=float)
        d_pct = (sub["abs_pct_feature"] - sub["abs_pct_pooled"]).to_numpy(dtype=float)
        out = {
            "slice": label,
            "n_paired": int(len(sub)),
            "median_abs_err_feature": float(sub["abs_err_feature"].median()),
            "median_abs_err_pooled": float(sub["abs_err_pooled"].median()),
            "median_delta_abs_err_feat_minus_pool": float(np.median(d_abs)),
            "median_abs_pct_feature": float(sub["abs_pct_feature"].median()),
            "median_abs_pct_pooled": float(sub["abs_pct_pooled"].median()),
            "median_delta_abs_pct_feat_minus_pool": float(np.median(d_pct)),
            "wilcoxon_abs_err_stat": float("nan"),
            "wilcoxon_abs_err_p": float("nan"),
            "wilcoxon_abs_pct_stat": float("nan"),
            "wilcoxon_abs_pct_p": float("nan"),
            "feature_better_abs_err": False,
            "feature_better_abs_pct": False,
        }
        if len(sub) >= 5:
            try:
                s, p = stats.wilcoxon(d_abs, zero_method="wilcox", alternative="two-sided")
                out["wilcoxon_abs_err_stat"] = float(s)
                out["wilcoxon_abs_err_p"] = float(p)
                out["feature_better_abs_err"] = bool(
                    p < ALPHA
                    and out["median_abs_err_feature"] < out["median_abs_err_pooled"]
                )
            except ValueError:
                pass
            try:
                s, p = stats.wilcoxon(d_pct, zero_method="wilcox", alternative="two-sided")
                out["wilcoxon_abs_pct_stat"] = float(s)
                out["wilcoxon_abs_pct_p"] = float(p)
                out["feature_better_abs_pct"] = bool(
                    p < ALPHA
                    and out["median_abs_pct_feature"] < out["median_abs_pct_pooled"]
                )
            except ValueError:
                pass
        return out

    rows.append(_one("all", m))
    for mode, sub in m.groupby("mode"):
        rows.append(_one(f"mode={mode}", sub))
    for eg, sub in m.groupby("event_group"):
        rows.append(_one(f"event_group={eg}", sub))
    for band, sub in m.groupby("band"):
        rows.append(_one(f"band={band}", sub))
    return pd.DataFrame(rows)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=C_NEUTRAL, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6, zorder=0)


def plot_pred_vs_actual(pred: pd.DataFrame, path: Path) -> None:
    g = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    if g.empty:
        return
    groups = sorted(g["event_group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.0), squeeze=False)
    for ax, eg in zip(axes[0], groups):
        sub = g[g["event_group"] == eg]
        ax.scatter(
            sub["to_time_actual"],
            sub["to_time_pred"],
            s=28,
            alpha=0.75,
            c=C_FEATURE,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        lo = min(sub["to_time_actual"].min(), sub["to_time_pred"].min())
        hi = max(sub["to_time_actual"].max(), sub["to_time_pred"].max())
        ax.plot([lo, hi], [lo, hi], color=C_NEUTRAL, linewidth=1.0, linestyle="--", zorder=2)
        _style_axes(ax, f"{eg} (n={len(sub)})", "Actual time (s)", "Predicted time (s)")
    fig.suptitle(
        "Season-PB · feature route: predicted vs actual",
        fontsize=12,
        color=C_NEUTRAL,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman(pred: pd.DataFrame, path: Path) -> None:
    g = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    if g.empty:
        return
    # Use % difference so events are comparable
    mean_t = (g["to_time_pred"] + g["to_time_actual"]) / 2.0
    pct_diff = 100.0 * (g["to_time_pred"] - g["to_time_actual"]) / g["to_time_actual"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = g["event_group"].map(
        {"Sprints": C_FEATURE, "Distance": C_POOLED, "Hurdles": "#117A65"}
    ).fillna(C_NEUTRAL)
    ax.scatter(mean_t, pct_diff, c=colors, s=28, alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)
    md = float(pct_diff.mean())
    sd = float(pct_diff.std(ddof=1)) if len(pct_diff) > 1 else 0.0
    ax.axhline(md, color=C_NEUTRAL, linewidth=1.2, label=f"mean bias {md:.2f}%")
    ax.axhline(md + 1.96 * sd, color=C_POOLED, linewidth=1.0, linestyle="--", label=f"±1.96 SD ({md + 1.96 * sd:.2f}%)")
    ax.axhline(md - 1.96 * sd, color=C_POOLED, linewidth=1.0, linestyle="--")
    _style_axes(
        ax,
        "Bland–Altman (percent scale) · season-PB · feature",
        "Mean of predicted and actual (s)",
        "Percent error (pred − actual) / actual × 100",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pair_mape(by_pair: pd.DataFrame, path: Path) -> None:
    g = by_pair[(by_pair["mode"] == "season_pb") & (by_pair["route_kind"] == "feature")].copy()
    g = g[g["n"] >= 3].sort_values("median_abs_pct_error")
    if g.empty:
        return
    labels = [
        f"{r.band} {r.from_event}→{r.to_event} (n={int(r.n)})" for r in g.itertuples()
    ]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.35 * len(g) + 1.5)))
    ax.barh(y, g["median_abs_pct_error"], color=C_FEATURE, height=0.7, zorder=3)
    xerr_lo = g["median_abs_pct_error"] - g["median_abs_pct_error_ci_lo"]
    xerr_hi = g["median_abs_pct_error_ci_hi"] - g["median_abs_pct_error"]
    ax.errorbar(
        g["median_abs_pct_error"],
        y,
        xerr=[xerr_lo, xerr_hi],
        fmt="none",
        ecolor=C_NEUTRAL,
        elinewidth=1.0,
        capsize=2.5,
        zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    _style_axes(
        ax,
        "Median |%| error by event pair (season-PB · feature) with 95% bootstrap CI",
        "Median absolute percent error (%)",
        "",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_vs_pooled(pred: pd.DataFrame, path: Path) -> None:
    key_cols = [
        "mode",
        "band",
        "from_event",
        "to_event",
        "college",
        "athlete",
        "season_year",
        "event_group",
    ]
    feat = pred[pred["route_kind"] == "feature"]
    pool = pred[pred["route_kind"] == "pooled"]
    m = feat.merge(
        pool,
        on=key_cols,
        suffixes=("_f", "_p"),
    )
    m = m[m["mode"] == "season_pb"]
    if m.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, eg in zip(axes, sorted(m["event_group"].unique())):
        sub = m[m["event_group"] == eg]
        ax.scatter(
            sub["abs_pct_error_p"],
            sub["abs_pct_error_f"],
            s=28,
            alpha=0.75,
            c=C_FEATURE,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        lo = 0.0
        hi = float(max(sub["abs_pct_error_p"].max(), sub["abs_pct_error_f"].max()) * 1.05)
        ax.plot([lo, hi], [lo, hi], color=C_NEUTRAL, linestyle="--", linewidth=1.0)
        _style_axes(
            ax,
            f"{eg} (n={len(sub)})",
            "Pooled |%| error",
            "Feature |%| error",
        )
    fig.suptitle(
        "Paired |%| error: feature vs pooled (points below diagonal favor feature)",
        fontsize=11,
        color=C_NEUTRAL,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_band(pred: pd.DataFrame, path: Path) -> None:
    g = pred[(pred["mode"] == "season_pb")].copy()
    if g.empty:
        return
    bands = ["750-950", "800-1000", "850-1050"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.35
    x = np.arange(len(bands))
    for offset, route, color in ((-width / 2, "feature", C_FEATURE), (width / 2, "pooled", C_POOLED)):
        meds, los, his = [], [], []
        for b in bands:
            sub = g[(g["band"] == b) & (g["route_kind"] == route)]["abs_pct_error"].to_numpy()
            med, lo, hi = bootstrap_ci(sub, np.median)
            meds.append(med)
            los.append(med - lo)
            his.append(hi - med)
        ax.bar(
            x + offset,
            meds,
            width=width,
            color=color,
            label=route,
            zorder=3,
        )
        ax.errorbar(x + offset, meds, yerr=[los, his], fmt="none", ecolor=C_NEUTRAL, capsize=3, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    _style_axes(
        ax,
        "Median |%| error by WA point band (season-PB) · 95% bootstrap CI",
        "World Athletics score band",
        "Median absolute percent error (%)",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_chrono_vs_pb(overall: pd.DataFrame, path: Path) -> None:
    g = overall[overall.get("event_group").isna() if "event_group" in overall.columns else True].copy()
    # Prefer rows without event_group (overall mixed) — filter those
    if "event_group" in overall.columns:
        g = overall[overall["event_group"].isna() | (overall["event_group"] == "")].copy()
        if g.empty:
            g = overall[overall["unit"] == "mixed_events_seconds_caution"].copy()
    else:
        g = overall.copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    modes = ["season_pb", "chronological"]
    x = np.arange(len(modes))
    width = 0.35
    for offset, route, color in ((-width / 2, "feature", C_FEATURE), (width / 2, "pooled", C_POOLED)):
        vals, los, his = [], [], []
        for mode in modes:
            row = g[(g["mode"] == mode) & (g["route_kind"] == route)]
            if row.empty:
                vals.append(np.nan)
                los.append(0)
                his.append(0)
                continue
            r = row.iloc[0]
            vals.append(r["median_abs_pct_error"])
            los.append(r["median_abs_pct_error"] - r["median_abs_pct_error_ci_lo"])
            his.append(r["median_abs_pct_error_ci_hi"] - r["median_abs_pct_error"])
        ax.bar(x + offset, vals, width=width, color=color, label=route, zorder=3)
        ax.errorbar(x + offset, vals, yerr=[los, his], fmt="none", ecolor=C_NEUTRAL, capsize=3, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Season PB → PB", "Chronological early → later"])
    _style_axes(
        ax,
        "Evaluation mode comparison · median |%| error · 95% bootstrap CI",
        "Evaluation mode",
        "Median absolute percent error (%)",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_stats_report(
    overall: pd.DataFrame,
    by_pair: pd.DataFrame,
    fvsp: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "Inferential Statistics Report — Men's Outdoor Time-Model Validation",
        "====================================================================",
        "",
        "Design notes",
        "------------",
        "  • External validation on NCAA Division I men's outdoor results (dated meets).",
        "  • Models were fit on club / National Running Club data — expect distribution shift.",
        "  • Primary cross-event metric: median absolute percent error (MedAPE).",
        "  • Absolute-second MedAE is interpretable only within an event pair or event group.",
        "  • Bootstrap: 2000 resamples, percentile 95% CI, seed=2026.",
        "  • Bias: Wilcoxon signed-rank on (pred − actual), two-sided α = 0.05.",
        "  • Feature vs pooled: paired Wilcoxon on |error| and |%| error.",
        "  • Tolerance hit rate: Clopper–Pearson exact 95% CI.",
        "",
    ]

    mix = overall[overall.get("unit", pd.Series(dtype=str)) == "mixed_events_seconds_caution"]
    if mix.empty and "event_group" in overall.columns:
        mix = overall[overall["event_group"].isna()]
    lines.append("Overall (MedAPE preferred; mixed-second MedAE shown for continuity)")
    lines.append("-------------------------------------------------------------------")
    for _, r in mix.iterrows():
        lines.append(
            f"  {r['mode']:14s} {r['route_kind']:8s}  n={int(r['n']):3d}  "
            f"MedAPE={r['median_abs_pct_error']:.2f}% "
            f"[{r['median_abs_pct_error_ci_lo']:.2f}, {r['median_abs_pct_error_ci_hi']:.2f}]  "
            f"tol={r['within_tol_rate']:.1%} "
            f"[{r['within_tol_ci_lo']:.1%}, {r['within_tol_ci_hi']:.1%}]  "
            f"Spearman ρ={r['spearman_rho']:.3f} (p={r['spearman_p']:.3g})  "
            f"bias_med={r['bias_median_error']:+.3f}s (Wilcoxon p={r['bias_wilcoxon_p']:.3g})"
        )
    lines.append("")

    eg = overall[overall.get("unit", pd.Series(dtype=str)) == "seconds_within_group"]
    if not eg.empty:
        lines.append("Within event group (absolute seconds are more interpretable)")
        lines.append("------------------------------------------------------------")
        for _, r in eg[eg["mode"] == "season_pb"].iterrows():
            lines.append(
                f"  {r['event_group']:10s} {r['route_kind']:8s}  n={int(r['n']):3d}  "
                f"MedAE={r['medae']:.3f}s [{r['medae_ci_lo']:.3f}, {r['medae_ci_hi']:.3f}]  "
                f"MedAPE={r['median_abs_pct_error']:.2f}%"
            )
        lines.append("")

    lines.append("Feature vs pooled (paired Wilcoxon; negative Δ favors feature)")
    lines.append("--------------------------------------------------------------")
    for _, r in fvsp.iterrows():
        lines.append(
            f"  {r['slice']:28s} n={int(r['n_paired']):3d}  "
            f"ΔMed|%|={r['median_delta_abs_pct_feat_minus_pool']:+.3f}  "
            f"p={r['wilcoxon_abs_pct_p']:.3g}  "
            f"feature_better={r['feature_better_abs_pct']}"
        )
    lines.append("")

    top = by_pair[
        (by_pair["mode"] == "season_pb")
        & (by_pair["route_kind"] == "feature")
        & (by_pair["n"] >= 5)
    ].nsmallest(8, "median_abs_pct_error")
    lines.append("Best-calibrated pairs (season-PB · feature · n≥5) by MedAPE")
    lines.append("-----------------------------------------------------------")
    for _, r in top.iterrows():
        lines.append(
            f"  {r['band']} {r['from_event']}→{r['to_event']}  n={int(r['n'])}  "
            f"MedAPE={r['median_abs_pct_error']:.2f}% "
            f"[{r['median_abs_pct_error_ci_lo']:.2f}, {r['median_abs_pct_error_ci_hi']:.2f}]  "
            f"MedAE={r['medae']:.3f}s"
        )
    lines.append("")

    worst = by_pair[
        (by_pair["mode"] == "season_pb")
        & (by_pair["route_kind"] == "feature")
        & (by_pair["n"] >= 4)
    ].nlargest(8, "median_abs_pct_error")
    lines.append("Weakest pairs (season-PB · feature · n≥4) by MedAPE")
    lines.append("--------------------------------------------------")
    for _, r in worst.iterrows():
        lines.append(
            f"  {r['band']} {r['from_event']}→{r['to_event']}  n={int(r['n'])}  "
            f"MedAPE={r['median_abs_pct_error']:.2f}% "
            f"[{r['median_abs_pct_error_ci_lo']:.2f}, {r['median_abs_pct_error_ci_hi']:.2f}]  "
            f"MedAE={r['medae']:.3f}s"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run_research_stats(predictions: pd.DataFrame | None = None) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pred = predictions if predictions is not None else pd.read_csv(PRED_PATH)
    if pred.empty:
        raise SystemExit("No predictions available for research stats.")

    overall = overall_inferential(pred)
    by_pair = by_pair_inferential(pred)
    fvsp = feature_vs_pooled_tests(pred)

    paths = {
        "overall": OUT_DIR / "inferential_overall.csv",
        "by_pair": OUT_DIR / "inferential_by_pair.csv",
        "feature_vs_pooled": OUT_DIR / "feature_vs_pooled_tests.csv",
        "report": OUT_DIR / "inferential_report.txt",
        "pred_vs_actual": PLOT_DIR / "pred_vs_actual_by_group.png",
        "bland_altman": PLOT_DIR / "bland_altman_pct.png",
        "pair_mape": PLOT_DIR / "medape_by_pair.png",
        "feature_vs_pooled_plot": PLOT_DIR / "feature_vs_pooled_pct.png",
        "error_by_band": PLOT_DIR / "medape_by_band.png",
        "chrono_vs_pb": PLOT_DIR / "medape_by_mode.png",
    }

    overall.to_csv(paths["overall"], index=False)
    by_pair.to_csv(paths["by_pair"], index=False)
    fvsp.to_csv(paths["feature_vs_pooled"], index=False)
    write_stats_report(overall, by_pair, fvsp, paths["report"])

    plot_pred_vs_actual(pred, paths["pred_vs_actual"])
    plot_bland_altman(pred, paths["bland_altman"])
    plot_pair_mape(by_pair, paths["pair_mape"])
    plot_feature_vs_pooled(pred, paths["feature_vs_pooled_plot"])
    plot_error_by_band(pred, paths["error_by_band"])
    plot_chrono_vs_pb(overall, paths["chrono_vs_pb"])

    print(paths["report"].read_text())
    for label, p in paths.items():
        if p.exists() and p.suffix != ".txt":
            print(f"Wrote {p}")
    print(f"Wrote {paths['report']}")
    return paths


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
