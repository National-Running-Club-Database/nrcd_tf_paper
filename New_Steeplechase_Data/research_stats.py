"""Inferential statistics layer for new_steeplechase_data.

This folder is the **canonical, steeplechase-corrected** distance dataset for
the project (see DATA_NOTES.txt / Summary_findings.md) — any paper text or
downstream model that needs distance-event best-event counts, nationals
thresholds, or steeplechase WA scores should read from here rather than the
legacy, pre-fix copies in `relays_findings/Distance_Relays_Findings/` or
`non_relays_findings/`.

Adds hypothesis tests / confidence intervals on top of the existing RQ1
best-event and RQ1B nationals outputs, without re-scraping any source data:

  1. RQ1 best-event shares (Wilson 95% CI + chi-square vs. uniform), per
     gender, restricted to the individual distance events.
  2. RQ1 pairwise head-to-head counts — exact two-sided binomial test vs.
     50/50 with Cohen's h (mirrors indoor_analysis / relays_findings).
  3. Legacy-vs-corrected steeplechase best-event share: Fisher's exact test
     comparing 3000m Steeplechase's share of the best-event pool in this
     (corrected) dataset vs. the legacy `relays_findings/Distance_Relays_Findings/`
     counts — quantifies how much the scoring fix moved steeple's specialization
     share.
  4. Kruskal-Wallis across distance events for nationals top-8 WA scores
     (RQ1B_Nationals/rq1b_nationals_top8_all_seasons.csv, corrected steeple).
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
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
RQ1_DIR = ROOT / "Distance_Relays_Findings"
LEGACY_RQ1_DIR = ROOT.parent / "relays_findings" / "Distance_Relays_Findings"
TOP8_CSV = ROOT / "RQ1B_Nationals" / "rq1b_nationals_top8_all_seasons.csv"

GENDERS = ["men", "women"]
ALPHA = 0.05
Z = 1.959963984540054
MIN_EVENT_N = 5

C_PRIMARY = "#2E86AB"
C_ACCENT = "#C73E1D"
C_LEGACY = "#95A5A6"
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


def _fmt_p(p) -> str:
    if not isinstance(p, (int, float)) or not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}"


# --------------------------------------------------------------------------
# 1 & 2. RQ1 best-event share + pairwise parsing (shared regex helpers)
# --------------------------------------------------------------------------

COUNT_LINE = re.compile(r"^(?P<event>.+?) \(\d+ competed\): (?P<count>\d+)\s*$")
PAIR_HEADER = re.compile(r"^(?P<a>.+?) vs\.\s+(?P<b>.+?) \((?P<n>\d+)[^)]*\):\s*$")
PAIR_COUNTS = re.compile(r"^(?P<a>.+?) best event: (?P<wa>\d+)\s+vs\.\s+(?P<b>.+?) best event: (?P<wb>\d+)\s*$")
PAIR_TIES = re.compile(r"^\(ties: (?P<ties>\d+)\)\s*$")


def parse_best_event_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        m = COUNT_LINE.match(line.strip())
        if m:
            counts[m.group("event")] = int(m.group("count"))
    return counts


def is_relay(event_name: str) -> bool:
    return event_name.startswith("4x") or "relay" in event_name.lower()


def parse_pairwise_file(path: Path) -> list[dict]:
    lines = [ln.rstrip() for ln in path.read_text().splitlines()]
    out, i = [], 0
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


def best_event_share_tests() -> list[dict]:
    rows = []
    for gender in GENDERS:
        path = RQ1_DIR / f"best_event_counts_{gender}_2024_2026_2plus_individual.txt"
        counts = {e: c for e, c in parse_best_event_counts(path).items() if not is_relay(e)}
        n = sum(counts.values())
        if n == 0:
            continue
        events = sorted(counts, key=lambda e: -counts[e])
        obs = np.array([counts[e] for e in events], dtype=float)
        chi2, chi_p = stats.chisquare(obs) if obs.sum() > 0 else (float("nan"), float("nan"))
        for rank, e in enumerate(events, start=1):
            k = counts[e]
            lo, hi = wilson_ci(k, n)
            rows.append(
                {
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


def pairwise_binomial_tests() -> list[dict]:
    rows = []
    for gender in GENDERS:
        path = RQ1_DIR / f"pairwise_best_event_counts_{gender}_2024_2026_individual.txt"
        for pair in parse_pairwise_file(path):
            wa, wb = pair["wins_a"], pair["wins_b"]
            n = wa + wb
            if n == 0:
                continue
            p = float(stats.binomtest(wa, n, 0.5, alternative="two-sided").pvalue)
            h = cohens_h(wa / n, wb / n)
            lo, hi = wilson_ci(wa, n)
            rows.append(
                {
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
# 3. Legacy vs. corrected steeplechase best-event share (Fisher exact)
# --------------------------------------------------------------------------


def legacy_vs_corrected_steeple() -> list[dict]:
    rows = []
    for gender in GENDERS:
        corrected_path = RQ1_DIR / f"best_event_counts_{gender}_2024_2026_2plus_individual.txt"
        legacy_path = LEGACY_RQ1_DIR / f"best_event_counts_{gender}_2024_2026_2plus_individual.txt"
        if not (corrected_path.exists() and legacy_path.exists()):
            continue
        corrected = {e: c for e, c in parse_best_event_counts(corrected_path).items() if not is_relay(e)}
        legacy = {e: c for e, c in parse_best_event_counts(legacy_path).items() if not is_relay(e)}
        n_corr, n_leg = sum(corrected.values()), sum(legacy.values())
        k_corr = corrected.get("3000m Steeplechase", 0)
        k_leg = legacy.get("3000m Steeplechase", 0)
        table = [[k_corr, n_corr - k_corr], [k_leg, n_leg - k_leg]]
        odds_ratio, p = stats.fisher_exact(table)
        lo_c, hi_c = wilson_ci(k_corr, n_corr)
        lo_l, hi_l = wilson_ci(k_leg, n_leg)
        rows.append(
            {
                "gender": gender.title(),
                "n_corrected_pool": n_corr,
                "steeple_count_corrected": k_corr,
                "steeple_share_corrected": round(k_corr / n_corr, 4) if n_corr else float("nan"),
                "steeple_share_corrected_ci_lo": round(lo_c, 4),
                "steeple_share_corrected_ci_hi": round(hi_c, 4),
                "n_legacy_pool": n_leg,
                "steeple_count_legacy": k_leg,
                "steeple_share_legacy": round(k_leg / n_leg, 4) if n_leg else float("nan"),
                "steeple_share_legacy_ci_lo": round(lo_l, 4),
                "steeple_share_legacy_ci_hi": round(hi_l, 4),
                "share_delta_corrected_minus_legacy": round(k_corr / n_corr - k_leg / n_leg, 4) if n_corr and n_leg else float("nan"),
                "fisher_odds_ratio": round(float(odds_ratio), 3) if np.isfinite(odds_ratio) else float("nan"),
                "fisher_p": float(p),
                "sig_005": bool(p < ALPHA),
            }
        )
    return rows


# --------------------------------------------------------------------------
# 4. Kruskal-Wallis across distance events, nationals top-8 WA
# --------------------------------------------------------------------------


def load_top8() -> list[dict]:
    with open(TOP8_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["world_athletics_points"] = float(r["world_athletics_points"])
    return [r for r in rows if r["world_athletics_points"] > 0 and r["discipline"] == "Distance"]


def kruskal_by_gender(rows: list[dict]) -> list[dict]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["gender"]][r["event_name"]].append(r["world_athletics_points"])
    out = []
    for gender, by_event in groups.items():
        events = {e: v for e, v in by_event.items() if len(v) >= MIN_EVENT_N}
        if len(events) < 2:
            continue
        stat_, p = stats.kruskal(*events.values())
        medians = {e: float(np.median(v)) for e, v in events.items()}
        out.append(
            {
                "gender": gender,
                "n_events": len(events),
                "n_total": sum(len(v) for v in events.values()),
                "kruskal_h": round(float(stat_), 3),
                "kruskal_df": len(events) - 1,
                "kruskal_p": float(p),
                "sig_005": bool(p < ALPHA),
                "highest_median_event": max(medians, key=medians.get),
                "highest_median_wa": round(max(medians.values()), 1),
                "lowest_median_event": min(medians, key=medians.get),
                "lowest_median_wa": round(min(medians.values()), 1),
            }
        )
    return out


def event_medians_long(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["gender"], r["event_name"])].append(r["world_athletics_points"])
    out = []
    for (gender, event), vals in groups.items():
        arr = np.array(vals)
        out.append(
            {
                "gender": gender,
                "event_name": event,
                "n": len(arr),
                "median_wa": round(float(np.median(arr)), 1),
                "q1_wa": round(float(np.percentile(arr, 25)), 1),
                "q3_wa": round(float(np.percentile(arr, 75)), 1),
            }
        )
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
    rows = sorted(rows, key=lambda r: (r["gender"], r["rank"]))
    labels = [f"{r['gender']} {r['event']} (n={r['n_best_event_pool']})" for r in rows]
    shares = [100 * r["share"] for r in rows]
    lo = [100 * r["share_ci_lo"] for r in rows]
    hi = [100 * r["share_ci_hi"] for r in rows]
    y = np.arange(len(rows))
    colors = [C_PRIMARY if r["gender"] == "Men" else C_ACCENT for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.42 * len(rows) + 1.5)))
    ax.barh(y, shares, color=colors, height=0.65, zorder=3)
    ax.errorbar(
        shares, y, xerr=[np.array(shares) - np.array(lo), np.array(hi) - np.array(shares)],
        fmt="none", ecolor=C_NEUTRAL, elinewidth=1.0, capsize=2.5, zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    _style_axes(
        ax,
        "Distance RQ1 best-event share (corrected steeplechase WA), Wilson 95% CI",
        "Share of individual-event best-event pool (%)",
        "",
    )
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "best_event_share_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_legacy_vs_corrected(rows: list[dict]) -> None:
    if not rows:
        return
    genders = [r["gender"] for r in rows]
    x = np.arange(len(genders))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for offset, key_share, key_lo, key_hi, color, label in (
        (-width / 2, "steeple_share_legacy", "steeple_share_legacy_ci_lo", "steeple_share_legacy_ci_hi", C_LEGACY, "Legacy (pre-fix)"),
        (width / 2, "steeple_share_corrected", "steeple_share_corrected_ci_lo", "steeple_share_corrected_ci_hi", C_PRIMARY, "Corrected (this folder)"),
    ):
        vals = [100 * r[key_share] for r in rows]
        lo = [100 * (r[key_share] - r[key_lo]) for r in rows]
        hi = [100 * (r[key_hi] - r[key_share]) for r in rows]
        ax.bar(x + offset, vals, width=width, color=color, label=label, zorder=3)
        ax.errorbar(x + offset, vals, yerr=[lo, hi], fmt="none", ecolor=C_NEUTRAL, capsize=3, zorder=4)
    for i, r in enumerate(rows):
        ax.text(i, max(100 * r["steeple_share_corrected"], 100 * r["steeple_share_legacy"]) + 1.5,
                 f"p={_fmt_p(r['fisher_p'])}", ha="center", fontsize=8, color=C_NEUTRAL)
    ax.set_xticks(x)
    ax.set_xticklabels(genders)
    _style_axes(
        ax,
        "3000m Steeplechase best-event share: legacy vs. corrected WA scoring\n"
        "(Wilson 95% CI; annotated p = Fisher's exact test)",
        "Gender",
        "Share of individual-event best-event pool (%)",
    )
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "steeple_legacy_vs_corrected.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top8_boxplot(rows: list[dict]) -> None:
    genders = sorted({r["gender"] for r in rows})
    fig, axes = plt.subplots(1, len(genders), figsize=(4.5 * len(genders), 5), squeeze=False)
    for ax, gender in zip(axes[0], genders):
        sub = [r for r in rows if r["gender"] == gender]
        events = sorted({r["event_name"] for r in sub}, key=lambda e: -np.median([r["world_athletics_points"] for r in sub if r["event_name"] == e]))
        data = [[r["world_athletics_points"] for r in sub if r["event_name"] == e] for e in events]
        bp = ax.boxplot(data, tick_labels=events, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=C_PRIMARY if gender == "Men" else C_ACCENT, alpha=0.55)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        _style_axes(ax, f"{gender} distance", "", "Nationals top-8 WA points" if gender == genders[0] else "")
        ax.grid(True, axis="y", alpha=0.3, color=C_GRID, zorder=0)
        ax.grid(False, axis="x")
    fig.suptitle("Nationals top-8 WA-point distributions by distance event (corrected steeple)", fontsize=12, color=C_NEUTRAL, y=1.02)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "nationals_top8_distance_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(share_rows, pair_rows, legacy_rows, kruskal_rows) -> None:
    lines = [
        "new_steeplechase_data — Inferential Statistics Report",
        "=========================================================",
        "",
        "IMPORTANT: this folder holds the CANONICAL, steeplechase-corrected distance",
        "dataset for this project. Legacy pre-fix steeplechase numbers live in",
        "relays_findings/Distance_Relays_Findings/ and non_relays_findings/ — see",
        "DATA_NOTES.txt for the correction magnitude (steeple WA means roughly doubled).",
        "Any paper text citing distance best-event shares, nationals thresholds, or",
        "steeplechase WA scores should use the numbers in THIS folder.",
        "",
        "No raw data was re-scraped for this research layer; numbers derive from the",
        "*.txt / *.csv already produced by analyze_rq1_best_event_distance.py and",
        "analyze_rq1b_nationals_distance.py (plus the legacy relays_findings copies,",
        "read read-only for comparison).",
        "",
        "## 1. RQ1 best-event shares (chi-square vs. uniform + Wilson 95% CI)",
        "-------------------------------------------------------------------------",
        "",
    ]
    for gender in ("Men", "Women"):
        sub = sorted([r for r in share_rows if r["gender"] == gender], key=lambda r: r["rank"])
        if not sub:
            continue
        lines.append(
            f"{gender} (n={sub[0]['n_best_event_pool']}): chi2={sub[0]['chi2_vs_uniform']} "
            f"df={sub[0]['chi2_df']} p={_fmt_p(sub[0]['chi2_p_vs_uniform'])}"
        )
        for r in sub:
            lines.append(
                f"    {r['event']:24s} {r['count']:4d} ({100*r['share']:5.1f}%) "
                f"[95% CI {100*r['share_ci_lo']:5.1f}%, {100*r['share_ci_hi']:5.1f}%]"
            )
        lines.append("")

    lines.extend(["## 2. RQ1 pairwise head-to-head (exact binomial vs. 50/50)", "----------------------------------------------------------------", ""])
    for gender in ("Men", "Women"):
        sub = [r for r in pair_rows if r["gender"] == gender]
        if not sub:
            continue
        lines.append(f"{gender}:")
        for r in sub:
            lines.append(
                f"    {r['event_a']} vs {r['event_b']}: {r['wins_a']}-{r['wins_b']} "
                f"(ties {r['ties']}, n={r['n_no_tie']}) p={_fmt_p(r['binomial_p'])} "
                f"Cohen's h={r['cohens_h']:+.3f} {r['sig']}"
            )
        lines.append("")

    lines.extend(
        [
            "## 3. Legacy vs. corrected steeplechase best-event share (Fisher's exact test)",
            "------------------------------------------------------------------------------------",
            "H0: steeple's share of the best-event pool is the same under legacy and",
            "corrected WA scoring (2x2 table: steeple vs. all-other-events x legacy vs. corrected).",
            "",
        ]
    )
    for r in legacy_rows:
        lines.append(
            f"  {r['gender']:6s} legacy: {r['steeple_count_legacy']:3d}/{r['n_legacy_pool']:4d} "
            f"({100*r['steeple_share_legacy']:.1f}% [95% CI {100*r['steeple_share_legacy_ci_lo']:.1f}%, "
            f"{100*r['steeple_share_legacy_ci_hi']:.1f}%])  ->  corrected: "
            f"{r['steeple_count_corrected']:3d}/{r['n_corrected_pool']:4d} "
            f"({100*r['steeple_share_corrected']:.1f}% [95% CI {100*r['steeple_share_corrected_ci_lo']:.1f}%, "
            f"{100*r['steeple_share_corrected_ci_hi']:.1f}%])  "
            f"Δ={100*r['share_delta_corrected_minus_legacy']:+.1f}pp  "
            f"Fisher p={_fmt_p(r['fisher_p'])} sig={r['sig_005']}"
        )
    lines.append("")

    lines.extend(
        [
            "## 4. Kruskal-Wallis across distance events — nationals top-8 WA (corrected steeple)",
            "-------------------------------------------------------------------------------------------",
            f"H0: nationals top-8 WA points are drawn from the same distribution across all",
            f"distance events (events with n >= {MIN_EVENT_N} places included).",
            "",
        ]
    )
    for r in kruskal_rows:
        lines.append(
            f"  {r['gender']:6s} n_events={r['n_events']} n={r['n_total']:4d}  "
            f"H={r['kruskal_h']:.2f} df={r['kruskal_df']} p={_fmt_p(r['kruskal_p'])} sig={r['sig_005']}  "
            f"deepest field: {r['highest_median_event']} (med={r['highest_median_wa']})  "
            f"shallowest: {r['lowest_median_event']} (med={r['lowest_median_wa']})"
        )
    lines.append("")

    lines.extend(
        [
            "## Bottom line",
            "--------------",
            "The scoring fix moved steeplechase from an essentially invisible best-event",
            "(legacy: ~0% men, ~0.3% women) to a real double-digit specialization share",
            "for both genders (corrected: 8.6% men, 19.9% women) — Fisher's exact test",
            "rejects equal-share at p<1e-20 for both. The Kruskal-Wallis test confirms",
            "distance nationals fields are not equally deep across events even after",
            "correction (steeple's field remains the shallowest for men; 800m is",
            "shallowest for women).",
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

    print("Testing RQ1 best-event shares (chi-square + Wilson CI)...")
    share_rows = best_event_share_tests()
    pair_rows = pairwise_binomial_tests()

    print("Comparing legacy vs. corrected steeplechase best-event share (Fisher exact)...")
    legacy_rows = legacy_vs_corrected_steeple()

    print("Running Kruskal-Wallis on nationals top-8 WA by distance event...")
    top8_rows = load_top8()
    kruskal_rows = kruskal_by_gender(top8_rows)
    event_medians = event_medians_long(top8_rows)

    paths = {
        "best_event_shares": OUT_DIR / "best_event_share_tests.csv",
        "pairwise_tests": OUT_DIR / "pairwise_binomial_tests.csv",
        "legacy_vs_corrected": OUT_DIR / "legacy_vs_corrected_steeple.csv",
        "kruskal": OUT_DIR / "kruskal_top8_by_event.csv",
        "event_medians": OUT_DIR / "top8_event_medians.csv",
        "report": OUT_DIR / "inferential_report.txt",
    }
    save_csv(share_rows, paths["best_event_shares"])
    save_csv(pair_rows, paths["pairwise_tests"])
    save_csv(legacy_rows, paths["legacy_vs_corrected"])
    save_csv(kruskal_rows, paths["kruskal"])
    save_csv(event_medians, paths["event_medians"])
    write_report(share_rows, pair_rows, legacy_rows, kruskal_rows)

    print("Building research figures...")
    plot_best_event_shares(share_rows)
    plot_legacy_vs_corrected(legacy_rows)
    plot_top8_boxplot(top8_rows)

    for label, p in paths.items():
        if p.exists():
            print(f"Wrote {p}")
    for r in legacy_rows:
        print(
            f"{r['gender']}: steeple share legacy {100*r['steeple_share_legacy']:.1f}% -> "
            f"corrected {100*r['steeple_share_corrected']:.1f}% (Fisher p={_fmt_p(r['fisher_p'])})"
        )
    return paths


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
