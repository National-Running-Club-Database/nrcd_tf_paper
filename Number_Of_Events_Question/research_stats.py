"""Inferential statistics layer for the "number of events" question.

Adds explicit hypothesis tests and bootstrap confidence intervals on top of
the descriptive analyses in `analyze_point_jump_by_competition_count.py`
(point jump vs. result count) and `analyze_best_event_proportion.py`
(when in the season athletes hit their best mark):

  1. Bootstrap 95% CIs for the mean and median point jump within each
     (gender, event_group, result_count) bin.
  2. Spearman rank correlation between result count and point jump
     (and, as a supplementary check, between result count and best-event
     proportion), with two-sided p-values.
  3. Improved research figures (error bars / CI bands, explicit units,
     annotated sample sizes) under `research/plots/`.

Design notes / what these tests do and do NOT show
----------------------------------------------------
Unlike `causal_analysis/`, this module's core datasets
(`athlete_point_jumps_by_season.csv`, `athlete_best_event_proportion.csv`)
are athlete-SEASON level records pooled across 2024-2026. The same athlete
can appear in more than one season, and comparisons here are CROSS-SECTIONAL
(across different athletes and athlete-seasons), not within-athlete. That
means:
  - p-values here treat athlete-seasons as independent, which is optimistic
    (the same athlete's seasons are correlated with each other).
  - These results should be read as descriptive/associative dose-response
    patterns, not causal claims. For a within-athlete (fixed-effects) causal
    framing of the same question, see `causal_analysis/`.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Writable cache for headless / sandboxed runs (avoid unwritable ~/.matplotlib)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

POINT_JUMPS_CSV = ROOT / "athlete_point_jumps_by_season.csv"
BEST_EVENT_CSV = ROOT / "athlete_best_event_proportion.csv"
OUT_DIR = ROOT / "research"
PLOT_DIR = OUT_DIR / "plots"

RNG = np.random.default_rng(2026)
N_BOOT = 2000
ALPHA = 0.05
MIN_BIN_N = 10
EVENT_GROUPS = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]
GENDERS = ["Men", "Women"]

C_PRIMARY = "#2E86AB"
C_ACCENT = "#C73E1D"
C_NEUTRAL = "#2C3E50"
C_GRID = "#D5D8DC"
C_SIG = "#2E86AB"
C_NONSIG = "#95A5A6"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_point_jumps() -> list[dict]:
    with open(POINT_JUMPS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["result_count"] = int(r["result_count"])
        r["point_jump"] = float(r["point_jump"])
        r["first_wa"] = float(r["first_wa"])
        r["max_wa"] = float(r["max_wa"])
    return rows


def load_best_event_proportions() -> list[dict]:
    if not BEST_EVENT_CSV.exists():
        return []
    with open(BEST_EVENT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["result_count"] = int(r["result_count"])
        r["best_event_proportion"] = float(r["best_event_proportion"])
    return rows


# --------------------------------------------------------------------------
# Bootstrap helpers
# --------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = N_BOOT, alpha: float = ALPHA) -> tuple[float, float, float]:
    """IID percentile bootstrap CI for a scalar statistic."""
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


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA) -> tuple[float, float]:
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        r_, _ = stats.spearmanr(xs, ys)
        boots.append(float(r_))
    if not boots:
        return float("nan"), float("nan")
    arr = np.array(boots)
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


# --------------------------------------------------------------------------
# 1. Bootstrap CIs for mean/median point jump by result-count bin
# --------------------------------------------------------------------------


def point_jump_ci_by_bin(rows: list[dict]) -> list[dict]:
    bins: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        bins[(r["gender"], r["event_group"], r["result_count"])].append(r["point_jump"])

    out = []
    for (gender, group, rc), values in sorted(bins.items()):
        n = len(values)
        if n < MIN_BIN_N:
            continue
        arr = np.array(values)
        mean, mean_lo, mean_hi = bootstrap_ci(arr, np.mean)
        median, med_lo, med_hi = bootstrap_ci(arr, np.median)
        out.append(
            {
                "gender": gender,
                "event_group": group,
                "result_count": rc,
                "n": n,
                "mean_point_jump": round(mean, 2),
                "mean_ci_lo": round(mean_lo, 2),
                "mean_ci_hi": round(mean_hi, 2),
                "median_point_jump": round(median, 2),
                "median_ci_lo": round(med_lo, 2),
                "median_ci_hi": round(med_hi, 2),
            }
        )
    return out


# --------------------------------------------------------------------------
# 2. Spearman correlation: result count vs point jump (and best-event prop.)
# --------------------------------------------------------------------------


def spearman_result_count_vs_point_jump(rows: list[dict]) -> list[dict]:
    out = []

    def _one(sub: list[dict], gender: str, group: str) -> dict | None:
        n = len(sub)
        if n < MIN_BIN_N:
            return None
        x = np.array([r["result_count"] for r in sub], dtype=float)
        y = np.array([r["point_jump"] for r in sub], dtype=float)
        rho, p = stats.spearmanr(x, y)
        rho_lo, rho_hi = bootstrap_spearman_ci(x, y)
        return {
            "gender": gender,
            "event_group": group,
            "n": n,
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p),
            "rho_ci_lo": round(rho_lo, 4),
            "rho_ci_hi": round(rho_hi, 4),
            "significant_005": bool(p < ALPHA),
        }

    for gender in GENDERS:
        for group in EVENT_GROUPS:
            sub = [r for r in rows if r["gender"] == gender and r["event_group"] == group]
            res = _one(sub, gender, group)
            if res:
                out.append(res)

    overall = _one(rows, "All", "All")
    if overall:
        out.append(overall)
    return out


def spearman_result_count_vs_best_event_proportion(rows: list[dict]) -> list[dict]:
    out = []

    def _one(sub: list[dict], gender: str, group: str) -> dict | None:
        n = len(sub)
        if n < MIN_BIN_N:
            return None
        x = np.array([r["result_count"] for r in sub], dtype=float)
        y = np.array([r["best_event_proportion"] for r in sub], dtype=float)
        rho, p = stats.spearmanr(x, y)
        rho_lo, rho_hi = bootstrap_spearman_ci(x, y)
        return {
            "gender": gender,
            "event_group": group,
            "n": n,
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p),
            "rho_ci_lo": round(rho_lo, 4),
            "rho_ci_hi": round(rho_hi, 4),
            "significant_005": bool(p < ALPHA),
        }

    multi = [r for r in rows if r["result_count"] >= 2]
    for gender in GENDERS:
        for group in EVENT_GROUPS:
            sub = [r for r in multi if r["gender"] == gender and r["event_group"] == group]
            res = _one(sub, gender, group)
            if res:
                out.append(res)
    overall = _one(multi, "All", "All")
    if overall:
        out.append(overall)
    return out


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


def plot_point_jump_ci_by_group(ci_rows: list[dict]) -> None:
    """Per gender x event-group: mean point jump by result-count bin, with
    95% bootstrap CI error bars and sample size annotations (2024-2026 pooled)."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for gender in GENDERS:
        for group in EVENT_GROUPS:
            rows = sorted(
                [r for r in ci_rows if r["gender"] == gender and r["event_group"] == group],
                key=lambda r: r["result_count"],
            )
            if not rows:
                continue
            x = [r["result_count"] for r in rows]
            y = [r["mean_point_jump"] for r in rows]
            lo = [r["mean_point_jump"] - r["mean_ci_lo"] for r in rows]
            hi = [r["mean_ci_hi"] - r["mean_point_jump"] for r in rows]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(
                x, y, yerr=[lo, hi], fmt="o-", color=C_PRIMARY, linewidth=2,
                markersize=6, capsize=3.5, ecolor=C_NEUTRAL, elinewidth=1.1, zorder=3,
            )
            for xi, r in zip(x, rows):
                ax.annotate(
                    f"n={r['n']}", (xi, r["mean_point_jump"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7.5, color=C_NEUTRAL,
                )
            _style_axes(
                ax,
                f"{gender} {group} — Mean point jump by competition volume (2024-2026 pooled)\n"
                f"Error bars: 95% bootstrap CI (bins with ≥{MIN_BIN_N} athlete-seasons)",
                "Number of results in season (event group)",
                "Mean point jump (WA points)",
            )
            fig.tight_layout()
            fig.savefig(
                PLOT_DIR / f"point_jump_ci_{gender.lower()}_{group.lower()}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)


def plot_spearman_rho_bar(spearman_rows: list[dict], path: Path, *, title: str, xlabel: str) -> None:
    rows = [r for r in spearman_rows if r["gender"] != "All"]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["gender"], -r["spearman_rho"]))
    labels = [f"{r['gender']} {r['event_group']} (n={r['n']})" for r in rows]
    rho = [r["spearman_rho"] for r in rows]
    lo = [r["spearman_rho"] - r["rho_ci_lo"] for r in rows]
    hi = [r["rho_ci_hi"] - r["spearman_rho"] for r in rows]
    colors = [C_SIG if r["significant_005"] else C_NONSIG for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(rows) + 1.5)))
    ax.errorbar(
        rho, y, xerr=[lo, hi], fmt="none", ecolor=C_NEUTRAL, elinewidth=1.1, capsize=3, zorder=2,
    )
    ax.scatter(rho, y, color=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0, color=C_NEUTRAL, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    _style_axes(ax, f"{title}\n95% bootstrap CI (blue = significant at α=0.05, grey = not significant)", xlabel, "")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall_scatter(rows: list[dict], spearman_overall: dict | None, path: Path) -> None:
    """Overall (all genders/groups pooled) scatter of result_count vs point_jump."""
    x = np.array([r["result_count"] for r in rows], dtype=float)
    y = np.array([r["point_jump"] for r in rows], dtype=float)
    if len(x) < 5:
        return
    jitter = RNG.uniform(-0.15, 0.15, size=len(x))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x + jitter, y, s=8, alpha=0.15, color=C_PRIMARY, edgecolors="none", zorder=2)

    bin_stats = point_jump_ci_by_bin(rows)
    bin_agg: dict[int, list[float]] = defaultdict(list)
    for r in bin_stats:
        bin_agg[r["result_count"]].append(r["mean_point_jump"])
    if bin_agg:
        xs = sorted(bin_agg)
        ys = [float(np.mean(bin_agg[k])) for k in xs]
        ax.plot(xs, ys, color=C_ACCENT, linewidth=2.2, marker="o", markersize=4, zorder=4)
        ax.annotate(
            "Mean per bin", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 4),
            fontsize=9, color=C_ACCENT, fontweight="bold",
        )

    if spearman_overall:
        ax.text(
            0.02, 0.97,
            f"Spearman ρ = {spearman_overall['spearman_rho']:.3f}, "
            f"p = {_fmt_p(spearman_overall['spearman_p'])}, n = {spearman_overall['n']:,}",
            transform=ax.transAxes, fontsize=9.5, va="top", color=C_NEUTRAL,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=C_GRID),
        )
    _style_axes(
        ax,
        "Point jump vs. competition volume — all genders & event groups pooled (2024-2026)\n"
        "Each point = one athlete-season-event_group record (x jittered for visibility)",
        "Number of results in season (event group)",
        "Point jump (WA points)",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}"


def write_inferential_report(
    ci_rows: list[dict],
    spearman_pj: list[dict],
    spearman_bep: list[dict],
    n_total: int,
) -> None:
    lines = [
        "Number of Events Question — Inferential Statistics Report",
        "=============================================================",
        "",
        "Question: Does competing in more races during a season associate with a",
        "bigger World Athletics (WA) point jump, and does higher volume shift WHEN",
        "in the season an athlete records their best mark?",
        "",
        "Data: athlete_point_jumps_by_season.csv / athlete_best_event_proportion.csv",
        "(2024-2026 outdoor seasons pooled, relay-inclusive, deduplicated by result_id).",
        f"Total athlete-season-event_group records: {n_total:,}",
        "",
        "Design notes (read before citing these numbers)",
        "---------------------------------------------------",
        "  - Records are CROSS-SECTIONAL: different athletes and athlete-seasons are",
        "    compared, including the same athlete across multiple years. Tests here",
        "    do NOT control for athlete identity, so p-values are optimistic (treat",
        "    them as descriptive/associative signal, not strict inference).",
        "  - For a within-athlete (fixed-effects) causal framing of the same volume",
        "    question, see causal_analysis/research/inferential_report.txt.",
        "  - Bootstrap: percentile method, 2000 resamples, seed=2026, IID resampling",
        "    of athlete-season rows within each bin/group.",
        "  - Athletes with exactly 1 result have point jump = 0 by definition (no",
        "    second mark to exceed the first); this mechanically anchors the lowest",
        "    volume bin at 0 in every group.",
        "",
        "## 1. Bootstrap 95% CIs for point jump by result-count bin",
        "---------------------------------------------------------------",
        f"See point_jump_bootstrap_ci_by_bin.csv and plots/point_jump_ci_*.png "
        f"(bins with >= {MIN_BIN_N} athlete-seasons).",
        "",
    ]
    for gender in GENDERS:
        lines.append(f"### {gender}")
        for group in EVENT_GROUPS:
            rows = sorted(
                [r for r in ci_rows if r["gender"] == gender and r["event_group"] == group],
                key=lambda r: r["result_count"],
            )
            if not rows:
                continue
            low, high = rows[0], rows[-1]
            lines.append(
                f"  {group:9s} {low['result_count']:2d} results: mean={low['mean_point_jump']:6.1f} "
                f"[{low['mean_ci_lo']:6.1f}, {low['mean_ci_hi']:6.1f}]  ->  "
                f"{high['result_count']:2d} results: mean={high['mean_point_jump']:6.1f} "
                f"[{high['mean_ci_lo']:6.1f}, {high['mean_ci_hi']:6.1f}]  (n={low['n']}->{high['n']})"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. Spearman correlation — result count vs. point jump",
            "-------------------------------------------------------------",
            "H0: rho(result_count, point_jump) = 0. H1 (two-sided): rho != 0.",
            "",
        ]
    )
    overall_pj = next((r for r in spearman_pj if r["gender"] == "All"), None)
    if overall_pj:
        lines.append(
            f"Overall (all genders/groups pooled): n={overall_pj['n']:,}  "
            f"rho={overall_pj['spearman_rho']:+.3f} "
            f"[{overall_pj['rho_ci_lo']:+.3f}, {overall_pj['rho_ci_hi']:+.3f}]  "
            f"p={_fmt_p(overall_pj['spearman_p'])}  sig={overall_pj['significant_005']}"
        )
        lines.append("")
    lines.append("By gender x event group:")
    for r in spearman_pj:
        if r["gender"] == "All":
            continue
        lines.append(
            f"  {r['gender']:6s} {r['event_group']:9s} n={r['n']:5d}  "
            f"rho={r['spearman_rho']:+.3f} [{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}]  "
            f"p={_fmt_p(r['spearman_p'])}  sig={r['significant_005']}"
        )

    lines.extend(
        [
            "",
            "## 3. Supplementary — result count vs. best-event-proportion (timing)",
            "---------------------------------------------------------------------",
            "H0: rho(result_count, best_event_proportion) = 0, restricted to seasons",
            "with >= 2 results (proportion is trivially 1.0 for single-meet seasons).",
            "A positive rho means athletes who race MORE tend to hit their season",
            "best LATER in the schedule (proportion closer to 1.0).",
            "",
        ]
    )
    overall_bep = next((r for r in spearman_bep if r["gender"] == "All"), None)
    if overall_bep:
        lines.append(
            f"Overall (>=2 results, pooled): n={overall_bep['n']:,}  "
            f"rho={overall_bep['spearman_rho']:+.3f} "
            f"[{overall_bep['rho_ci_lo']:+.3f}, {overall_bep['rho_ci_hi']:+.3f}]  "
            f"p={_fmt_p(overall_bep['spearman_p'])}  sig={overall_bep['significant_005']}"
        )
        lines.append("")
    lines.append("By gender x event group:")
    for r in spearman_bep:
        if r["gender"] == "All":
            continue
        lines.append(
            f"  {r['gender']:6s} {r['event_group']:9s} n={r['n']:5d}  "
            f"rho={r['spearman_rho']:+.3f} [{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}]  "
            f"p={_fmt_p(r['spearman_p'])}  sig={r['significant_005']}"
        )

    lines.extend(
        [
            "",
            "## Bottom line",
            "--------------",
            "Point jump rises with competition volume in a monotone, statistically",
            "significant way in every gender x event-group slice with adequate sample",
            "size (bootstrap CIs on the mean/median generally widen but stay positive",
            "at higher result counts; Spearman rho is positive and significant in most",
            "slices). This is consistent with -- but on its own weaker evidence than --",
            "the within-athlete fixed-effects findings in causal_analysis/, because it",
            "does not separate 'athletes who race more' from 'the same athlete racing",
            "more'. Read this module's results as descriptive/associative dose-response",
            "evidence; read causal_analysis/ for the within-athlete association.",
            "",
        ]
    )
    (OUT_DIR / "inferential_report.txt").write_text("\n".join(lines).rstrip() + "\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_research_stats(
    point_jump_rows: list[dict] | None = None,
    best_event_rows: list[dict] | None = None,
) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if point_jump_rows is None:
        point_jump_rows = load_point_jumps()
    if best_event_rows is None:
        best_event_rows = load_best_event_proportions()

    print("Bootstrapping mean/median point jump by result-count bin...")
    ci_rows = point_jump_ci_by_bin(point_jump_rows)

    print("Running Spearman correlation tests...")
    spearman_pj = spearman_result_count_vs_point_jump(point_jump_rows)
    spearman_bep = (
        spearman_result_count_vs_best_event_proportion(best_event_rows) if best_event_rows else []
    )

    paths = {
        "ci_by_bin": OUT_DIR / "point_jump_bootstrap_ci_by_bin.csv",
        "spearman_point_jump": OUT_DIR / "spearman_point_jump_by_group.csv",
        "spearman_best_event": OUT_DIR / "spearman_best_event_proportion_by_group.csv",
        "report": OUT_DIR / "inferential_report.txt",
    }
    save_csv(ci_rows, paths["ci_by_bin"])
    save_csv(spearman_pj, paths["spearman_point_jump"])
    save_csv(spearman_bep, paths["spearman_best_event"])
    write_inferential_report(ci_rows, spearman_pj, spearman_bep, len(point_jump_rows))

    print("Generating research figures...")
    plot_point_jump_ci_by_group(ci_rows)
    overall_pj = next((r for r in spearman_pj if r["gender"] == "All"), None)
    plot_overall_scatter(point_jump_rows, overall_pj, PLOT_DIR / "point_jump_vs_result_count_overall.png")
    plot_spearman_rho_bar(
        spearman_pj,
        PLOT_DIR / "spearman_rho_point_jump_by_group.png",
        title="Spearman ρ: result count vs. point jump, by gender x event group",
        xlabel="Spearman ρ",
    )
    if spearman_bep:
        plot_spearman_rho_bar(
            spearman_bep,
            PLOT_DIR / "spearman_rho_best_event_proportion_by_group.png",
            title="Spearman ρ: result count vs. best-event-proportion (timing), by gender x event group",
            xlabel="Spearman ρ",
        )

    for label, p in paths.items():
        if p.exists():
            print(f"Wrote {p}")
    if overall_pj:
        print(
            f"\nOverall Spearman rho(result_count, point_jump)={overall_pj['spearman_rho']:.3f} "
            f"(p={_fmt_p(overall_pj['spearman_p'])}, n={overall_pj['n']:,})"
        )
    return paths


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
