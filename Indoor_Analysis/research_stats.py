"""Inferential statistics layer for indoor_analysis.

Adds hypothesis tests / confidence intervals on top of the existing RQ1A
best-event counts and the indoor<->outdoor season-opener WA comparison,
without re-scraping any source data:

  1. RQ1A best-event shares — chi-square goodness-of-fit (vs. a uniform
     null across individual events) and Wilson 95% CIs on each event's
     share of the individual-event best-event pool, per discipline x gender.
  2. RQ1A pairwise head-to-head counts — exact two-sided binomial test vs.
     a 50/50 null (mirrors non_relays_findings' improved-RQ1A approach),
     with Cohen's h effect size.
  3. Indoor -> outdoor season-opener WA transition (Indoor_Outdoor_Interplay/
     opener_wa_pair_comparisons_band_750_950.csv, the band-restricted sample
     already summarized in Summary_findings.md): paired Wilcoxon signed-rank
     test, a Wilson 95% CI on P(outdoor higher), and Spearman rho between
     indoor and outdoor opener WA — overall and by event pair (n >= 10).

All counts/CSVs are read from files already produced by analyze_indoor_rq1a.py
and Indoor_Outdoor_Interplay/analyze_indoor_outdoor_opener_wa.py.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT_DIR = ROOT / "research"
PLOT_DIR = OUT_DIR / "plots"
RQ1A_DIR = ROOT / "RQ1A_Best_Event"
OPENER_CSV = ROOT / "Indoor_Outdoor_Interplay" / "opener_wa_pair_comparisons_band_750_950.csv"

DISCIPLINES = ["Sprints", "Distance", "Hurdles", "Jumps", "Throws"]
GENDERS = ["men", "women"]

ALPHA = 0.05
Z = 1.959963984540054  # 97.5th pctile of standard normal, for Wilson CI
MIN_PAIR_N = 10

C_PRIMARY = "#2E86AB"
C_ACCENT = "#C73E1D"
C_NEUTRAL = "#2C3E50"
C_GRID = "#D5D8DC"


def wilson_ci(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    adj = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return max(0.0, (center - adj) / denom), min(1.0, (center + adj) / denom)


def cohens_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# --------------------------------------------------------------------------
# 1. RQ1A best-event share parsing + chi-square / Wilson CI
# --------------------------------------------------------------------------

COUNT_LINE = re.compile(r"^(?P<event>.+?) \(\d+ competed\): (?P<count>\d+)\s*$")


def parse_best_event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        m = COUNT_LINE.match(line.strip())
        if m:
            counts[m.group("event")] = int(m.group("count"))
    return counts


def is_relay(event_name: str) -> bool:
    return event_name.startswith("4x") or "relay" in event_name.lower()


def best_event_share_tests() -> list[dict]:
    rows: list[dict] = []
    for disc in DISCIPLINES:
        for gender in GENDERS:
            path = RQ1A_DIR / disc / f"best_event_counts_{gender}_2024_2026_2plus_individual.txt"
            if not path.exists():
                continue
            counts = {e: c for e, c in parse_best_event_counts(path).items() if not is_relay(e)}
            counts = {e: c for e, c in counts.items() if c > 0} or counts
            n = sum(counts.values())
            if n == 0 or len(counts) < 2:
                continue
            events = sorted(counts, key=lambda e: -counts[e])
            chi2, chi_p = float("nan"), float("nan")
            if len(events) >= 2:
                obs = np.array([counts[e] for e in events], dtype=float)
                if obs.sum() > 0:
                    chi2, chi_p = stats.chisquare(obs)
            for rank, e in enumerate(events, start=1):
                k = counts[e]
                lo, hi = wilson_ci(k, n)
                rows.append(
                    {
                        "discipline": disc,
                        "gender": gender.title(),
                        "event": e,
                        "rank": rank,
                        "n_best_event_pool": n,
                        "count": k,
                        "share": round(k / n, 4),
                        "share_ci_lo": round(lo, 4),
                        "share_ci_hi": round(hi, 4),
                        "chi2_vs_uniform": round(float(chi2), 3) if np.isfinite(chi2) else "",
                        "chi2_df": len(events) - 1,
                        "chi2_p_vs_uniform": chi_p,
                        "chi2_sig_005": bool(np.isfinite(chi_p) and chi_p < ALPHA),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# 2. Pairwise head-to-head — exact binomial vs 50/50
# --------------------------------------------------------------------------

PAIR_HEADER = re.compile(r"^(?P<a>.+?) vs\.\s+(?P<b>.+?) \((?P<n>\d+)[^)]*\):\s*$")
PAIR_COUNTS = re.compile(r"^(?P<a>.+?) best event: (?P<wa>\d+)\s+vs\.\s+(?P<b>.+?) best event: (?P<wb>\d+)\s*$")
PAIR_TIES = re.compile(r"^\(ties: (?P<ties>\d+)\)\s*$")


def parse_pairwise_file(path: Path) -> list[dict]:
    lines = [ln.rstrip() for ln in path.read_text().splitlines()]
    out = []
    i = 0
    while i < len(lines):
        m = PAIR_HEADER.match(lines[i])
        if m:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                m2 = PAIR_COUNTS.match(lines[j])
                if m2:
                    ties = 0
                    if j + 1 < len(lines):
                        mt = PAIR_TIES.match(lines[j + 1])
                        if mt:
                            ties = int(mt.group("ties"))
                    out.append(
                        {
                            "event_a": m2.group("a"),
                            "event_b": m2.group("b"),
                            "wins_a": int(m2.group("wa")),
                            "wins_b": int(m2.group("wb")),
                            "ties": ties,
                        }
                    )
            i = j
        i += 1
    return out


def pairwise_binomial_tests() -> list[dict]:
    rows = []
    for disc in DISCIPLINES:
        for gender in GENDERS:
            path = RQ1A_DIR / disc / f"pairwise_best_event_counts_{gender}_2024_2026_individual.txt"
            if not path.exists():
                continue
            for pair in parse_pairwise_file(path):
                wa, wb = pair["wins_a"], pair["wins_b"]
                n = wa + wb
                if n == 0:
                    continue
                res = stats.binomtest(wa, n, 0.5, alternative="two-sided")
                p = float(res.pvalue)
                h = cohens_h(wa / n, wb / n)
                lo, hi = wilson_ci(wa, n)
                rows.append(
                    {
                        "discipline": disc,
                        "gender": gender.title(),
                        "event_a": pair["event_a"],
                        "event_b": pair["event_b"],
                        "wins_a": wa,
                        "wins_b": wb,
                        "ties": pair["ties"],
                        "n_no_tie": n,
                        "share_a": round(wa / n, 4),
                        "share_a_ci_lo": round(lo, 4),
                        "share_a_ci_hi": round(hi, 4),
                        "binomial_p": p,
                        "cohens_h": round(float(h), 4),
                        "sig": _sig_stars(p),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# 3. Indoor -> outdoor season-opener WA transition
# --------------------------------------------------------------------------


def load_opener_csv() -> list[dict]:
    with open(OPENER_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["indoor_wa"] = float(r["indoor_wa"])
        r["outdoor_wa"] = float(r["outdoor_wa"])
        r["outdoor_higher"] = r["outdoor_higher"] == "True"
    return rows


def opener_tests(rows: list[dict]) -> dict:
    indoor = np.array([r["indoor_wa"] for r in rows])
    outdoor = np.array([r["outdoor_wa"] for r in rows])
    diff = outdoor - indoor
    n = len(rows)
    n_higher = sum(1 for d in diff if d > 0)
    n_lower = sum(1 for d in diff if d < 0)
    n_tie = n - n_higher - n_lower
    n_no_tie = n_higher + n_lower

    out = {
        "n": n,
        "n_outdoor_higher": n_higher,
        "n_indoor_higher": n_lower,
        "n_tie": n_tie,
        "pct_outdoor_higher": round(100 * n_higher / n, 1) if n else float("nan"),
        "outdoor_higher_ci_lo": float("nan"),
        "outdoor_higher_ci_hi": float("nan"),
        "mean_delta": round(float(diff.mean()), 2) if n else float("nan"),
        "median_delta": round(float(np.median(diff)), 2) if n else float("nan"),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p": float("nan"),
        "binomial_p_vs_half": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
    }
    if n_no_tie > 0:
        lo, hi = wilson_ci(n_higher, n_no_tie)
        out["outdoor_higher_ci_lo"] = round(lo, 4)
        out["outdoor_higher_ci_hi"] = round(hi, 4)
        out["binomial_p_vs_half"] = float(stats.binomtest(n_higher, n_no_tie, 0.5).pvalue)
    if n >= 5 and np.any(diff != 0):
        try:
            stat_, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_stat"] = float(stat_)
            out["wilcoxon_p"] = float(p)
        except ValueError:
            pass
    if n >= 3 and np.std(indoor) > 0 and np.std(outdoor) > 0:
        rho, p = stats.spearmanr(indoor, outdoor)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(p)
    return out


def opener_tests_by_pair(rows: list[dict]) -> list[dict]:
    out = []
    pairs = sorted({r["pair_label"] for r in rows})
    for pair in pairs:
        sub = [r for r in rows if r["pair_label"] == pair]
        if len(sub) < MIN_PAIR_N:
            continue
        t = opener_tests(sub)
        t["pair_label"] = pair
        out.append(t)
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
    ax.grid(True, axis="x", alpha=0.3, color=C_GRID, zorder=0)


def plot_best_event_shares(rows: list[dict]) -> None:
    """Horizontal bar chart of top-event share with Wilson 95% CI, one panel per discipline."""
    fig, axes = plt.subplots(len(DISCIPLINES), 1, figsize=(8, 2.3 * len(DISCIPLINES)))
    for ax, disc in zip(axes, DISCIPLINES):
        sub = [r for r in rows if r["discipline"] == disc]
        sub.sort(key=lambda r: (r["gender"], r["rank"]))
        labels = [f"{r['gender']} {r['event']} (n={r['n_best_event_pool']})" for r in sub]
        shares = [100 * r["share"] for r in sub]
        lo = [100 * r["share_ci_lo"] for r in sub]
        hi = [100 * r["share_ci_hi"] for r in sub]
        y = np.arange(len(sub))
        colors = [C_PRIMARY if g == "Men" else C_ACCENT for g in [r["gender"] for r in sub]]
        ax.barh(y, shares, color=colors, height=0.65, zorder=3)
        ax.errorbar(
            shares, y, xerr=[np.array(shares) - np.array(lo), np.array(hi) - np.array(shares)],
            fmt="none", ecolor=C_NEUTRAL, elinewidth=1.0, capsize=2.5, zorder=4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        _style_axes(ax, f"{disc} — best-event share (Wilson 95% CI)", "Share of best-event pool (%)", "")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "best_event_share_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_opener_scatter(rows: list[dict], overall: dict) -> None:
    x = np.array([r["indoor_wa"] for r in rows])
    y = np.array([r["outdoor_wa"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = [C_PRIMARY if r["outdoor_higher"] else C_ACCENT for r in rows]
    ax.scatter(x, y, s=22, alpha=0.55, c=colors, edgecolors="none", zorder=3)
    lo, hi = min(x.min(), y.min()) - 20, max(x.max(), y.max()) + 20
    ax.plot([lo, hi], [lo, hi], color=C_NEUTRAL, linewidth=1.2, linestyle="--", zorder=2, label="Indoor = Outdoor")
    ax.text(
        0.02, 0.97,
        f"n={overall['n']}  outdoor higher={overall['pct_outdoor_higher']}%\n"
        f"Spearman ρ={overall['spearman_rho']:.3f} (p={overall['spearman_p']:.2g})\n"
        f"Wilcoxon p={overall['wilcoxon_p']:.2g}",
        transform=ax.transAxes, fontsize=9, va="top", color=C_NEUTRAL,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor=C_GRID),
    )
    _style_axes(
        ax,
        "Indoor vs. outdoor season-opener WA (band-restricted, outdoor opener WA ∈ [750, 950))\n"
        "blue = outdoor opener higher, red = indoor opener higher",
        "Indoor opener WA",
        "Outdoor opener WA",
    )
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "opener_wa_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_opener_by_pair(by_pair: list[dict]) -> None:
    if not by_pair:
        return
    rows = sorted(by_pair, key=lambda r: r["pct_outdoor_higher"])
    labels = [f"{r['pair_label']} (n={r['n']})" for r in rows]
    shares = [r["pct_outdoor_higher"] for r in rows]
    lo = [100 * r["outdoor_higher_ci_lo"] for r in rows]
    hi = [100 * r["outdoor_higher_ci_hi"] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(rows) + 1.5)))
    ax.barh(y, shares, color=C_PRIMARY, height=0.65, zorder=3)
    ax.errorbar(
        shares, y, xerr=[np.array(shares) - np.array(lo), np.array(hi) - np.array(shares)],
        fmt="none", ecolor=C_NEUTRAL, elinewidth=1.0, capsize=2.5, zorder=4,
    )
    ax.axvline(50, color=C_ACCENT, linewidth=1.0, linestyle="--", zorder=2, label="50% (coin flip)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    _style_axes(
        ax,
        "P(outdoor opener WA > indoor opener WA) by event pair — Wilson 95% CI",
        "% comparisons where outdoor opener is higher",
        "",
    )
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "opener_higher_by_pair.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _fmt_p(p) -> str:
    if not isinstance(p, (int, float)) or not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}"


def write_report(share_rows, pair_rows, opener_overall, opener_by_pair) -> None:
    lines = [
        "indoor_analysis — Inferential Statistics Report",
        "=================================================",
        "",
        "Scope: lightweight hypothesis tests / CIs layered on top of the existing",
        "RQ1A best-event counts and the indoor->outdoor season-opener WA comparison.",
        "No raw data was re-scraped; all numbers derive from the *.txt / *.csv already",
        "produced by analyze_indoor_rq1a.py and",
        "Indoor_Outdoor_Interplay/analyze_indoor_outdoor_opener_wa.py.",
        "",
        "## 1. RQ1A best-event shares (chi-square vs. uniform + Wilson 95% CI)",
        "-----------------------------------------------------------------------",
        "H0 (chi-square): best event is uniformly distributed across the individual",
        "events considered for that discipline x gender cohort.",
        "",
    ]
    for disc in DISCIPLINES:
        for gender in ("Men", "Women"):
            sub = [r for r in share_rows if r["discipline"] == disc and r["gender"] == gender]
            if not sub:
                continue
            sub.sort(key=lambda r: r["rank"])
            chi_p = sub[0]["chi2_p_vs_uniform"]
            lines.append(
                f"{disc} {gender} (n={sub[0]['n_best_event_pool']}): "
                f"chi2={sub[0]['chi2_vs_uniform']} df={sub[0]['chi2_df']} p={_fmt_p(chi_p)}"
            )
            for r in sub:
                lines.append(
                    f"    {r['event']:24s} {r['count']:4d} ({100*r['share']:5.1f}%) "
                    f"[95% CI {100*r['share_ci_lo']:5.1f}%, {100*r['share_ci_hi']:5.1f}%]"
                )
            lines.append("")

    lines.extend(
        [
            "## 2. RQ1A pairwise head-to-head (exact binomial vs. 50/50, ties excluded)",
            "----------------------------------------------------------------------------",
            "",
        ]
    )
    for disc in DISCIPLINES:
        for gender in ("Men", "Women"):
            sub = [r for r in pair_rows if r["discipline"] == disc and r["gender"] == gender]
            if not sub:
                continue
            lines.append(f"{disc} {gender}:")
            for r in sub:
                lines.append(
                    f"    {r['event_a']} vs {r['event_b']}: {r['wins_a']}-{r['wins_b']} "
                    f"(ties {r['ties']}, n={r['n_no_tie']}) "
                    f"p={_fmt_p(r['binomial_p'])} Cohen's h={r['cohens_h']:+.3f} {r['sig']}"
                )
            lines.append("")

    lines.extend(
        [
            "## 3. Indoor -> outdoor season-opener WA transition",
            "-----------------------------------------------------",
            "Sample: outdoor-opener-WA-in-[750,950) band (matches Summary_findings.md).",
            "H0 (Wilcoxon): median(outdoor_wa - indoor_wa) = 0 across comparable opener pairs.",
            "H0 (binomial): P(outdoor higher) = 0.5, ties excluded.",
            "",
            f"Overall: n={opener_overall['n']} "
            f"(outdoor higher={opener_overall['n_outdoor_higher']}, "
            f"indoor higher={opener_overall['n_indoor_higher']}, ties={opener_overall['n_tie']})",
            f"  P(outdoor higher) = {opener_overall['pct_outdoor_higher']}% "
            f"[95% CI {100*opener_overall['outdoor_higher_ci_lo']:.1f}%, "
            f"{100*opener_overall['outdoor_higher_ci_hi']:.1f}%]  "
            f"binomial p={_fmt_p(opener_overall['binomial_p_vs_half'])}",
            f"  Mean delta (outdoor - indoor) = {opener_overall['mean_delta']:+.1f} WA, "
            f"median = {opener_overall['median_delta']:+.1f} WA",
            f"  Wilcoxon signed-rank: stat={opener_overall['wilcoxon_stat']:.1f} "
            f"p={_fmt_p(opener_overall['wilcoxon_p'])}",
            f"  Spearman rho (indoor_wa vs outdoor_wa) = {opener_overall['spearman_rho']:.3f} "
            f"(p={_fmt_p(opener_overall['spearman_p'])})",
            "",
            "By event pair (n >= 10):",
        ]
    )
    for r in sorted(opener_by_pair, key=lambda r: -r["n"]):
        lines.append(
            f"  {r['pair_label']:24s} n={r['n']:4d}  outdoor_higher={r['pct_outdoor_higher']:5.1f}% "
            f"[{100*r['outdoor_higher_ci_lo']:5.1f}%, {100*r['outdoor_higher_ci_hi']:5.1f}%]  "
            f"Δmed={r['median_delta']:+6.1f}  Wilcoxon p={_fmt_p(r['wilcoxon_p'])}  "
            f"binom p={_fmt_p(r['binomial_p_vs_half'])}"
        )
    lines.append("")
    lines.extend(
        [
            "## Bottom line",
            "--------------",
            "Best-event chi-square tests confirm the RQ1A specialization patterns already",
            "described in Summary_findings.md are not uniform-random artifacts (most cohorts",
            "reject the uniform null at p<0.05, small-n disciplines like indoor Hurdles/Throws",
            "aside). The opener Wilcoxon/binomial tests quantify how reliably outdoor season",
            "openers beat indoor openers for the same athlete and comparable event.",
            "",
        ]
    )
    (OUT_DIR / "inferential_report.txt").write_text("\n".join(lines).rstrip() + "\n")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_research_stats() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Testing RQ1A best-event shares (chi-square + Wilson CI)...")
    share_rows = best_event_share_tests()
    pair_rows = pairwise_binomial_tests()

    print("Testing indoor -> outdoor season-opener WA transition...")
    opener_rows = load_opener_csv()
    opener_overall = opener_tests(opener_rows)
    opener_by_pair = opener_tests_by_pair(opener_rows)

    paths = {
        "best_event_shares": OUT_DIR / "best_event_share_tests.csv",
        "pairwise_tests": OUT_DIR / "pairwise_binomial_tests.csv",
        "opener_overall": OUT_DIR / "opener_wa_tests_overall.csv",
        "opener_by_pair": OUT_DIR / "opener_wa_tests_by_pair.csv",
        "report": OUT_DIR / "inferential_report.txt",
    }
    save_csv(share_rows, paths["best_event_shares"])
    save_csv(pair_rows, paths["pairwise_tests"])
    save_csv([opener_overall], paths["opener_overall"])
    save_csv(opener_by_pair, paths["opener_by_pair"])
    write_report(share_rows, pair_rows, opener_overall, opener_by_pair)

    print("Building research figures...")
    plot_best_event_shares(share_rows)
    plot_opener_scatter(opener_rows, opener_overall)
    plot_opener_by_pair(opener_by_pair)

    for label, p in paths.items():
        if p.exists():
            print(f"Wrote {p}")
    print(
        f"\nOverall opener: outdoor higher {opener_overall['pct_outdoor_higher']}% "
        f"(Wilcoxon p={_fmt_p(opener_overall['wilcoxon_p'])}, "
        f"Spearman rho={opener_overall['spearman_rho']:.3f})"
    )
    return paths


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
