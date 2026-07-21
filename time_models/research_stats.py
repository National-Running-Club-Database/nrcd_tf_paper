"""Inferential layer for time_models feature-importance / point-band CV tables.

Reads existing CSVs under Point_Bands_Time_Models/Feature_Importance_* — does not
re-fit models. Produces bootstrap CIs, sign/Wilcoxon tests on strategy gains,
and research figures under research/plots/.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
FI_DIR = (
    ROOT
    / "Point_Bands_Time_Models"
    / "Feature_Importance_Point_Band_Time_Models"
)
OUT = ROOT / "research"
PLOT_DIR = OUT / "plots"
RNG = np.random.default_rng(2026)
N_BOOT = 2000
ALPHA = 0.05
C_MAIN = "#1B4F72"
C_ALT = "#7B241C"
C_NEUTRAL = "#2C3E50"
C_GRID = "#D5D8DC"


def bootstrap_ci(values: np.ndarray, stat_fn=np.median) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat_fn(values))
    if len(values) == 1:
        return (point, point, point)
    boots = np.empty(N_BOOT)
    n = len(values)
    for i in range(N_BOOT):
        boots[i] = float(stat_fn(values[RNG.integers(0, n, size=n)]))
    return point, float(np.quantile(boots, ALPHA / 2)), float(np.quantile(boots, 1 - ALPHA / 2))


def wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    z = stats.norm.ppf(1 - ALPHA / 2)
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0.0, center - half), min(1.0, center + half)


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=C_NEUTRAL, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6, zorder=0)


def strategy_gain_tests(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band, g in pair.groupby("band"):
        delta = g["delta_best_vs_pooled"].to_numpy(dtype=float)
        delta = delta[np.isfinite(delta)]
        med, lo, hi = bootstrap_ci(delta, np.median)
        n_pos = int(np.sum(delta > 0))
        n = int(len(delta))
        rate, rlo, rhi = wilson_ci(n_pos, n)
        wilcox_p = float("nan")
        if n >= 5 and np.any(delta != 0):
            try:
                _, wilcox_p = stats.wilcoxon(delta, zero_method="wilcox", alternative="greater")
            except ValueError:
                pass
        # Exact binomial: H0 best beats pooled half the time
        binom_p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue) if n else float("nan")
        rows.append(
            {
                "band": band,
                "n_pairs": n,
                "median_delta_best_vs_pooled": med,
                "median_delta_ci_lo": lo,
                "median_delta_ci_hi": hi,
                "mean_delta": float(np.mean(delta)) if n else float("nan"),
                "beat_rate": rate,
                "beat_rate_ci_lo": rlo,
                "beat_rate_ci_hi": rhi,
                "wilcoxon_greater_p": wilcox_p,
                "binom_beat_rate_p": binom_p,
            }
        )
    # Overall
    delta = pair["delta_best_vs_pooled"].to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    med, lo, hi = bootstrap_ci(delta, np.median)
    n_pos = int(np.sum(delta > 0))
    n = int(len(delta))
    rate, rlo, rhi = wilson_ci(n_pos, n)
    wilcox_p = float("nan")
    if n >= 5 and np.any(delta != 0):
        try:
            _, wilcox_p = stats.wilcoxon(delta, zero_method="wilcox", alternative="greater")
        except ValueError:
            pass
    binom_p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue) if n else float("nan")
    rows.append(
        {
            "band": "all",
            "n_pairs": n,
            "median_delta_best_vs_pooled": med,
            "median_delta_ci_lo": lo,
            "median_delta_ci_hi": hi,
            "mean_delta": float(np.mean(delta)) if n else float("nan"),
            "beat_rate": rate,
            "beat_rate_ci_lo": rlo,
            "beat_rate_ci_hi": rhi,
            "wilcoxon_greater_p": wilcox_p,
            "binom_beat_rate_p": binom_p,
        }
    )
    return pd.DataFrame(rows)


def strategy_win_counts(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band, g in pair.groupby("band"):
        vc = g["best_strategy"].value_counts()
        for strat, cnt in vc.items():
            rows.append({"band": band, "strategy": strat, "times_best": int(cnt), "n_pairs": int(len(g))})
    return pd.DataFrame(rows)


def plot_delta_by_band(tests: pd.DataFrame, path: Path) -> None:
    g = tests[tests["band"] != "all"].copy()
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(g))
    ax.bar(x, g["median_delta_best_vs_pooled"], color=C_MAIN, zorder=3)
    yerr_lo = np.clip(g["median_delta_best_vs_pooled"] - g["median_delta_ci_lo"], 0, None)
    yerr_hi = np.clip(g["median_delta_ci_hi"] - g["median_delta_best_vs_pooled"], 0, None)
    ax.errorbar(
        x,
        g["median_delta_best_vs_pooled"],
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor=C_NEUTRAL,
        capsize=3,
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(g["band"])
    style(
        ax,
        "Median CV MedAE gain of best feature strategy vs pooled (95% bootstrap CI)",
        "WA point band",
        "Δ MedAE (pooled − best) in seconds",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_beat_rate(tests: pd.DataFrame, path: Path) -> None:
    g = tests.copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(g))
    ax.bar(x, g["beat_rate"], color=C_ALT, zorder=3)
    yerr_lo = np.clip(g["beat_rate"] - g["beat_rate_ci_lo"], 0, None)
    yerr_hi = np.clip(g["beat_rate_ci_hi"] - g["beat_rate"], 0, None)
    ax.errorbar(
        x,
        g["beat_rate"],
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor=C_NEUTRAL,
        capsize=3,
        zorder=4,
    )
    ax.axhline(0.5, color=C_NEUTRAL, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(g["band"])
    style(
        ax,
        "Share of event pairs where best feature strategy beats pooled (Wilson 95% CI)",
        "WA point band",
        "Beat rate",
    )
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pair_deltas(pair: pd.DataFrame, path: Path) -> None:
    g = pair[pair["n"] >= 20].nlargest(15, "delta_best_vs_pooled")
    if g.empty:
        g = pair.nlargest(15, "delta_best_vs_pooled")
    if g.empty:
        return
    labels = [
        f"{r.band} {r.gender} {r.from_event}→{r.to_event} (n={int(r.n)})"
        for r in g.itertuples()
    ]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.35 * len(g) + 1.2)))
    ax.barh(y, g["delta_best_vs_pooled"], color=C_MAIN, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    style(
        ax,
        "Largest CV MedAE gains: best feature strategy vs pooled (n≥20 preferred)",
        "Δ MedAE (seconds)",
        "",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(tests: pd.DataFrame, wins: pd.DataFrame, pair: pd.DataFrame, path: Path) -> None:
    lines = [
        "time_models — Inferential Statistics Report (Feature Importance CV)",
        "===================================================================",
        "",
        "Source: Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/",
        "        pair_strategy_cv_by_band.csv (5-fold CV median |error|, seconds).",
        "Tests: Wilcoxon signed-rank on Δ = cv_pooled − cv_best (H1: Δ > 0);",
        "       exact binomial on beat-rate vs 50%; bootstrap 95% CI (B=2000, seed=2026).",
        "Note: CV MedAE values are already cross-validated point estimates; bootstrap here",
        "      reflects variability across event pairs, not re-sampling of athletes.",
        "",
        "Feature strategy vs pooled by band",
        "----------------------------------",
    ]
    for _, r in tests.iterrows():
        lines.append(
            f"  {r['band']:10s} n_pairs={int(r['n_pairs']):2d}  "
            f"median Δ={r['median_delta_best_vs_pooled']:.3f}s "
            f"[{r['median_delta_ci_lo']:.3f}, {r['median_delta_ci_hi']:.3f}]  "
            f"beat_rate={r['beat_rate']:.1%} [{r['beat_rate_ci_lo']:.1%}, {r['beat_rate_ci_hi']:.1%}]  "
            f"Wilcoxon p={r['wilcoxon_greater_p']:.3g}  binom p={r['binom_beat_rate_p']:.3g}"
        )
    lines.append("")
    lines.append("Most frequent winning strategies by band")
    lines.append("----------------------------------------")
    for band, g in wins.groupby("band"):
        top = g.sort_values("times_best", ascending=False).head(3)
        parts = [f"{r.strategy}×{int(r.times_best)}" for r in top.itertuples()]
        lines.append(f"  {band}: " + ", ".join(parts))
    lines.append("")
    top_pairs = pair.nlargest(8, "delta_best_vs_pooled")
    lines.append("Largest pair-level gains (best strategy vs pooled)")
    lines.append("--------------------------------------------------")
    for _, r in top_pairs.iterrows():
        best_cv = float(r["cv_pooled"] - r["delta_best_vs_pooled"])
        lines.append(
            f"  {r['band']} {r['gender']} {r['from_event']}→{r['to_event']}  "
            f"n={int(r['n'])}  strategy={r['best_strategy']}  "
            f"Δ={r['delta_best_vs_pooled']:.3f}s  "
            f"(pooled={r['cv_pooled']:.3f} → best={best_cv:.3f})"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run_research_stats() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    pair_path = FI_DIR / "pair_strategy_cv_by_band.csv"
    if not pair_path.is_file():
        raise SystemExit(f"Missing {pair_path}")

    pair = pd.read_csv(pair_path)
    tests = strategy_gain_tests(pair)
    wins = strategy_win_counts(pair)

    paths = {
        "tests": OUT / "strategy_gain_tests.csv",
        "wins": OUT / "strategy_win_counts.csv",
        "report": OUT / "inferential_report.txt",
        "delta_plot": PLOT_DIR / "medae_gain_by_band.png",
        "beat_plot": PLOT_DIR / "beat_rate_by_band.png",
        "pair_plot": PLOT_DIR / "largest_pair_gains.png",
    }
    tests.to_csv(paths["tests"], index=False)
    wins.to_csv(paths["wins"], index=False)
    write_report(tests, wins, pair, paths["report"])
    plot_delta_by_band(tests, paths["delta_plot"])
    plot_beat_rate(tests, paths["beat_plot"])
    plot_pair_deltas(pair, paths["pair_plot"])

    print(paths["report"].read_text())
    for p in paths.values():
        if p.exists():
            print(f"Wrote {p}")
    return paths


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
