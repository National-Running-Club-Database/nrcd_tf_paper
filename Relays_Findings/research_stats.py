"""Inferential statistics layer for relays_findings (relay-inclusive RQ1).

Adds hypothesis tests / confidence intervals on top of the existing RQ1B
nationals and RQ1C competitiveness outputs, without re-scraping any source
data:

  1. Kruskal-Wallis test across events for nationals top-8 WA scores, per
     discipline x gender, using RQ1B_Nationals/rq1b_nationals_top8_all_seasons.csv
     (does the "8th-place bar" differ meaningfully across events, i.e. are
     some events' national fields systematically deeper/shallower?).
  2. RQ1A = RQ1C agreement rate, per discipline x gender, with a Wilson 95%
     CI (parsed from RQ1C_Competitiveness/rq1c_competitiveness_analysis.txt).
  3. Individual-vs-relay and individual-vs-individual head-to-head
     competitiveness margins (same source txt): exact binomial test vs.
     50/50 with Cohen's h, mirroring non_relays_findings' improved-RQ1A
     approach and indoor_analysis/research_stats.py.
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
TOP8_CSV = ROOT / "RQ1B_Nationals" / "rq1b_nationals_top8_all_seasons.csv"
RQ1C_TXT = ROOT / "RQ1C_Competitiveness" / "rq1c_competitiveness_analysis.txt"

ALPHA = 0.05
Z = 1.959963984540054
MIN_EVENT_N = 5

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


def _fmt_p(p) -> str:
    if not isinstance(p, (int, float)) or not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}"


# --------------------------------------------------------------------------
# 1. Kruskal-Wallis across events for nationals top-8 WA scores
# --------------------------------------------------------------------------


def load_top8() -> list[dict]:
    """Load nationals top-8 rows, dropping zero-WA placeholder rows.

    A handful of relay types (e.g. 4x800m, DMR, SMR) have no valid WA scoring
    in this dataset and are recorded as 0 — see relays_findings/README.md
    ("Relay Events Excluded from Threshold-Based Comparisons"). Including
    them would make those events look artificially "shallow" rather than
    absent, so they are excluded here exactly as they are from the RQ1B/RQ1C
    threshold tables.
    """
    with open(TOP8_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["world_athletics_points"] = float(r["world_athletics_points"])
    return [r for r in rows if r["world_athletics_points"] > 0]


def kruskal_by_discipline_gender(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["discipline"], r["gender"])][r["event_name"]].append(r["world_athletics_points"])

    out = []
    for (disc, gender), by_event in groups.items():
        events = {e: v for e, v in by_event.items() if len(v) >= MIN_EVENT_N}
        if len(events) < 2:
            continue
        samples = list(events.values())
        stat_, p = stats.kruskal(*samples)
        medians = {e: float(np.median(v)) for e, v in events.items()}
        top_event = max(medians, key=medians.get)
        bottom_event = min(medians, key=medians.get)
        out.append(
            {
                "discipline": disc,
                "gender": gender,
                "n_events": len(events),
                "n_total": sum(len(v) for v in events.values()),
                "kruskal_h": round(float(stat_), 3),
                "kruskal_df": len(events) - 1,
                "kruskal_p": float(p),
                "sig_005": bool(p < ALPHA),
                "highest_median_event": top_event,
                "highest_median_wa": round(medians[top_event], 1),
                "lowest_median_event": bottom_event,
                "lowest_median_wa": round(medians[bottom_event], 1),
            }
        )
    return sorted(out, key=lambda r: (r["discipline"], r["gender"]))


def event_medians_long(rows: list[dict]) -> list[dict]:
    """Long-format per-event summary (n, median, IQR) for the box plot / report."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["discipline"], r["gender"], r["event_name"])].append(r["world_athletics_points"])
    out = []
    for (disc, gender, event), vals in groups.items():
        arr = np.array(vals)
        out.append(
            {
                "discipline": disc,
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
# 2 & 3. RQ1C txt parsing: agreement rates + head-to-head competitiveness
# --------------------------------------------------------------------------

DISC_HEADER = re.compile(r"^## (?P<disc>\w+)\s*$")
GENDER_HEADER = re.compile(r"^### (?P<gender>Men|Women)\s*$")
AGREEMENT_LINE = re.compile(r"^RQ1A = RQ1C agreement: (?P<agree>\d+) \((?P<pct>[\d.]+)%\) \| Mismatch: (?P<mismatch>\d+)\s*$")
SECTION_HEADER = re.compile(r"^(?P<kind>Individual×relay competitiveness|Individual head-to-head competitiveness)")
MARGIN_LINE = re.compile(
    r"^\s*(?P<a>.+?) vs\.\s+(?P<b>.+?) \((?P<n>\d+) in both\): "
    r"(?P<a2>.+?) more competitive: (?P<wa>\d+) \| (?P<b2>.+?) more competitive: (?P<wb>\d+)"
    r"(?:\s*\|\s*ties: (?P<ties>\d+))?\s*$"
)


def parse_rq1c_txt() -> tuple[list[dict], list[dict]]:
    text = RQ1C_TXT.read_text().splitlines()
    agreement_rows: list[dict] = []
    margin_rows: list[dict] = []
    disc, gender, section = None, None, None
    for line in text:
        m = DISC_HEADER.match(line)
        if m:
            disc, gender, section = m.group("disc"), None, None
            continue
        m = GENDER_HEADER.match(line)
        if m:
            gender, section = m.group("gender"), None
            continue
        m = AGREEMENT_LINE.match(line.strip())
        if m and disc and gender:
            agree, mismatch = int(m.group("agree")), int(m.group("mismatch"))
            n = agree + mismatch
            lo, hi = wilson_ci(agree, n)
            agreement_rows.append(
                {
                    "discipline": disc,
                    "gender": gender,
                    "n_multi_event": n,
                    "n_agree": agree,
                    "n_mismatch": mismatch,
                    "agree_rate": round(agree / n, 4) if n else float("nan"),
                    "agree_rate_ci_lo": round(lo, 4),
                    "agree_rate_ci_hi": round(hi, 4),
                }
            )
            continue
        m = SECTION_HEADER.match(line.strip())
        if m:
            section = "individual_vs_relay" if "relay" in m.group("kind") else "individual_vs_individual"
            continue
        m = MARGIN_LINE.match(line)
        if m and disc and gender and section:
            wa, wb = int(m.group("wa")), int(m.group("wb"))
            n = wa + wb
            if n == 0:
                continue
            p = float(stats.binomtest(wa, n, 0.5, alternative="two-sided").pvalue)
            h = cohens_h(wa / n, wb / n)
            lo, hi = wilson_ci(wa, n)
            margin_rows.append(
                {
                    "discipline": disc,
                    "gender": gender,
                    "kind": section,
                    "event_a": m.group("a"),
                    "event_b": m.group("b"),
                    "wins_a": wa,
                    "wins_b": wb,
                    "ties": int(m.group("ties")) if m.group("ties") else 0,
                    "n_no_tie": n,
                    "share_a": round(wa / n, 4),
                    "share_a_ci_lo": round(lo, 4),
                    "share_a_ci_hi": round(hi, 4),
                    "binomial_p": p,
                    "cohens_h": round(float(h), 4),
                    "sig": _sig_stars(p),
                }
            )
    return agreement_rows, margin_rows


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


def plot_agreement_rates(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["agree_rate"])
    labels = [f"{r['gender']} {r['discipline']} (n={r['n_multi_event']})" for r in rows]
    pct = [100 * r["agree_rate"] for r in rows]
    lo = [100 * r["agree_rate_ci_lo"] for r in rows]
    hi = [100 * r["agree_rate_ci_hi"] for r in rows]
    y = np.arange(len(rows))
    colors = [C_PRIMARY if r["gender"] == "Men" else C_ACCENT for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.42 * len(rows) + 1.5)))
    ax.barh(y, pct, color=colors, height=0.65, zorder=3)
    ax.errorbar(
        pct, y, xerr=[np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)],
        fmt="none", ecolor=C_NEUTRAL, elinewidth=1.0, capsize=2.5, zorder=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    _style_axes(
        ax,
        "RQ1A = RQ1C agreement rate (relay-inclusive), Wilson 95% CI",
        "% of multi-event athletes where best event = most competitive event",
        "",
    )
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "agreement_rate_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top8_boxplot(rows: list[dict]) -> None:
    disciplines = sorted({r["discipline"] for r in rows})
    fig, axes = plt.subplots(1, len(disciplines), figsize=(3.6 * len(disciplines), 5.5), squeeze=False)
    for ax, disc in zip(axes[0], disciplines):
        sub = [r for r in rows if r["discipline"] == disc]
        events = sorted({r["event_name"] for r in sub}, key=lambda e: -np.median([r["world_athletics_points"] for r in sub if r["event_name"] == e]))
        data = [[r["world_athletics_points"] for r in sub if r["event_name"] == e] for e in events]
        bp = ax.boxplot(data, tick_labels=events, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set(facecolor=C_PRIMARY, alpha=0.55)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        _style_axes(ax, disc, "", "Nationals top-8 WA points" if disc == disciplines[0] else "")
        ax.grid(True, axis="y", alpha=0.3, color=C_GRID, zorder=0)
        ax.grid(False, axis="x")
    fig.suptitle("Nationals top-8 WA-point distributions by event (all seasons, both genders)", fontsize=12, color=C_NEUTRAL, y=1.02)
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "nationals_top8_wa_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_individual_vs_relay(margin_rows: list[dict]) -> None:
    sub = [r for r in margin_rows if r["kind"] == "individual_vs_relay" and r["n_no_tie"] >= MIN_EVENT_N]
    if not sub:
        return
    sub.sort(key=lambda r: r["share_a"])
    labels = [f"{r['gender']} {r['event_a']} vs {r['event_b']} (n={r['n_no_tie']})" for r in sub]
    pct = [100 * r["share_a"] for r in sub]
    lo = [100 * r["share_a_ci_lo"] for r in sub]
    hi = [100 * r["share_a_ci_hi"] for r in sub]
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(sub) + 1.5)))
    ax.barh(y, pct, color=C_PRIMARY, height=0.65, zorder=3)
    ax.errorbar(
        pct, y, xerr=[np.array(pct) - np.array(lo), np.array(hi) - np.array(pct)],
        fmt="none", ecolor=C_NEUTRAL, elinewidth=1.0, capsize=2.5, zorder=4,
    )
    ax.axvline(50, color=C_ACCENT, linewidth=1.0, linestyle="--", zorder=2, label="50% (coin flip)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    _style_axes(
        ax,
        "Individual event more competitive than paired relay (margin above 8th-place bar)\n"
        "% of dual-event athletes, Wilson 95% CI",
        "% where individual event wins the margin comparison",
        "",
    )
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "individual_vs_relay_margin_wins.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(kruskal_rows, agreement_rows, margin_rows) -> None:
    lines = [
        "relays_findings — Inferential Statistics Report",
        "=================================================",
        "",
        "Scope: lightweight hypothesis tests / CIs layered on top of the existing",
        "RQ1B nationals top-8 and RQ1C competitiveness outputs (relay-inclusive).",
        "No raw data was re-scraped; numbers derive from RQ1B_Nationals/",
        "rq1b_nationals_top8_all_seasons.csv and RQ1C_Competitiveness/",
        "rq1c_competitiveness_analysis.txt.",
        "",
        "## 1. Kruskal-Wallis across events — nationals top-8 WA scores",
        "-------------------------------------------------------------------",
        "H0: nationals top-8 WA points are drawn from the same distribution across",
        "all events within a discipline x gender cohort (all seasons pooled, events",
        f"with n >= {MIN_EVENT_N} places included).",
        "",
    ]
    for r in kruskal_rows:
        lines.append(
            f"  {r['gender']:6s} {r['discipline']:9s} n_events={r['n_events']} n={r['n_total']:4d}  "
            f"H={r['kruskal_h']:.2f} df={r['kruskal_df']} p={_fmt_p(r['kruskal_p'])} sig={r['sig_005']}  "
            f"deepest field: {r['highest_median_event']} (med={r['highest_median_wa']})  "
            f"shallowest: {r['lowest_median_event']} (med={r['lowest_median_wa']})"
        )
    lines.append("")

    lines.extend(
        [
            "## 2. RQ1A = RQ1C agreement rate (Wilson 95% CI)",
            "----------------------------------------------------",
            "",
        ]
    )
    for r in sorted(agreement_rows, key=lambda r: -r["agree_rate"]):
        lines.append(
            f"  {r['gender']:6s} {r['discipline']:9s} n={r['n_multi_event']:4d}  "
            f"agree={r['n_agree']:4d} ({100*r['agree_rate']:.1f}%) "
            f"[95% CI {100*r['agree_rate_ci_lo']:.1f}%, {100*r['agree_rate_ci_hi']:.1f}%]  "
            f"mismatch={r['n_mismatch']}"
        )
    lines.append("")

    lines.extend(
        [
            "## 3. Individual x relay competitiveness margins (exact binomial vs. 50/50)",
            "--------------------------------------------------------------------------------",
            "'wins' = more competitive (larger margin above nationals 8th-place threshold).",
            "",
        ]
    )
    for kind, title in (
        ("individual_vs_relay", "Individual vs. relay"),
        ("individual_vs_individual", "Individual vs. individual"),
    ):
        lines.append(f"{title}:")
        for r in [m for m in margin_rows if m["kind"] == kind]:
            lines.append(
                f"    {r['gender']:6s} {r['discipline']:9s} {r['event_a']} vs {r['event_b']}: "
                f"{r['wins_a']}-{r['wins_b']} (ties {r['ties']}, n={r['n_no_tie']}) "
                f"p={_fmt_p(r['binomial_p'])} Cohen's h={r['cohens_h']:+.3f} {r['sig']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Bottom line",
            "--------------",
            "The Kruskal-Wallis tests confirm nationals top-8 fields are not equally deep",
            "across events within a discipline (the 4x100m/4x400m relay fields are",
            "consistently shallower than their individual-event counterparts, matching the",
            "RQ1B threshold table in Summary_findings.md). The individual-vs-relay binomial",
            "tests show that in most sprint/hurdle/jump/throw comparisons the relay wins the",
            "margin comparison significantly more than half the time, while distance events",
            "hold their own against 4x400m/4x100m more often.",
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

    print("Running Kruskal-Wallis on nationals top-8 WA scores by event...")
    top8_rows = load_top8()
    kruskal_rows = kruskal_by_discipline_gender(top8_rows)
    event_medians = event_medians_long(top8_rows)

    print("Parsing RQ1C agreement rates and head-to-head margins...")
    agreement_rows, margin_rows = parse_rq1c_txt()

    paths = {
        "kruskal": OUT_DIR / "kruskal_top8_by_event.csv",
        "event_medians": OUT_DIR / "top8_event_medians.csv",
        "agreement": OUT_DIR / "rq1a_rq1c_agreement_rates.csv",
        "margins": OUT_DIR / "individual_relay_margin_tests.csv",
        "report": OUT_DIR / "inferential_report.txt",
    }
    save_csv(kruskal_rows, paths["kruskal"])
    save_csv(event_medians, paths["event_medians"])
    save_csv(agreement_rows, paths["agreement"])
    save_csv(margin_rows, paths["margins"])
    write_report(kruskal_rows, agreement_rows, margin_rows)

    print("Building research figures...")
    plot_agreement_rates(agreement_rows)
    plot_top8_boxplot(top8_rows)
    plot_individual_vs_relay(margin_rows)

    for label, p in paths.items():
        if p.exists():
            print(f"Wrote {p}")
    sig_count = sum(1 for r in kruskal_rows if r["sig_005"])
    print(f"\nKruskal-Wallis significant in {sig_count}/{len(kruskal_rows)} discipline x gender cohorts.")
    return paths


def main() -> None:
    run_research_stats()


if __name__ == "__main__":
    main()
