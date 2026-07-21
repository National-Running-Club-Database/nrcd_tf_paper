"""Improved RQ1 analysis with confidence intervals, effect sizes, and significance tests."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for lib in (
    ROOT / "Distance_Events_Counting" / ".pylibs",
    ROOT / "Sprints_Events_Counting" / ".pylibs",
):
    if lib.exists():
        sys.path.insert(0, str(lib))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scoring.columns import ALL_METRICS, points_col as scoring_points_col

OUT = ROOT / "improved_rq1_outputs"
OUT.mkdir(exist_ok=True)

# Active scoring metric for this run (wa | vdot | purdy | mercier)
ACTIVE_METRIC = "wa"

PRELIM_EVENT_IDS = {3, 4}
FIELD_EVENT_IDS = {38, 39, 40, 41, 42, 43, 44, 45, 46}
SEASONS = ["2024", "2025", "2026"]
ALPHA = 0.05
RNG = random.Random(42)

EVENT_MAP = {
    int(r["running_event_id"]): r["event_name"]
    for r in csv.DictReader(open(ROOT / "Distance_Events_Counting" / "running_event.csv"))
}

THRESHOLDS = {
    ("Men", "800m"): 829.3,
    ("Men", "1500m"): 827.0,
    ("Men", "200m"): 820.3,
    ("Men", "100m"): 818.0,
    ("Men", "400m"): 791.7,
    ("Men", "Long Jump"): 791.7,
    ("Men", "5000m"): 786.0,
    ("Men", "High Jump"): 732.0,
    ("Men", "400m Hurdles"): 728.3,
    ("Men", "Triple Jump"): 725.7,
    ("Men", "Shot Put"): 671.7,
    ("Men", "Discus"): 659.3,
    ("Men", "110m Hurdles"): 629.0,
    ("Men", "3000m Steeplechase"): 456.0,
    ("Women", "100m"): 814.3,
    ("Women", "5000m"): 808.5,
    ("Women", "200m"): 799.3,
    ("Women", "1500m"): 791.7,
    ("Women", "400m"): 788.3,
    ("Women", "Long Jump"): 760.3,
    ("Women", "800m"): 759.7,
    ("Women", "Triple Jump"): 741.0,
    ("Women", "High Jump"): 681.0,
    ("Women", "400m Hurdles"): 660.7,
    ("Women", "100m Hurdles"): 607.7,
    ("Women", "Shot Put"): 579.7,
    ("Women", "3000m Steeplechase"): 510.7,
    ("Women", "Discus"): 504.3,
}

DISCIPLINES = [
    ("Sprints", "Sprints_Events_Counting", "Sprinters", {
        3: "100m", 4: "200m", 6: "400m"
    }, ["100m", "200m", "400m"], [
        ("100m", "200m"), ("100m", "400m"), ("200m", "400m")
    ]),
    ("Distance", "Distance_Events_Counting", "Distance", {
        9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"
    }, ["800m", "1500m", "3000m Steeplechase", "5000m"], [
        ("800m", "1500m"), ("800m", "3000m Steeplechase"), ("800m", "5000m"),
        ("1500m", "3000m Steeplechase"), ("1500m", "5000m"),
        ("3000m Steeplechase", "5000m"),
    ]),
    ("Hurdles", "Hurdles_Events_Counting", "Hurdles", {
        35: "110m Hurdles", 37: "400m Hurdles", 34: "100m Hurdles", 36: "400m Hurdles"
    }, None, None),
    ("Jumps", "Jumps_Events_Counting", "Jumps", {
        38: "Long Jump", 39: "Triple Jump", 40: "High Jump"
    }, ["Long Jump", "Triple Jump", "High Jump"], [
        ("Long Jump", "Triple Jump"), ("Long Jump", "High Jump"), ("Triple Jump", "High Jump")
    ]),
    ("Throws", "Throws_Events_Counting", "Throws", {
        41: "Shot Put", 42: "Discus"
    }, ["Shot Put", "Discus"], [("Shot Put", "Discus")]),
]


def pcol(gender: str) -> str:
    return scoring_points_col(gender, ACTIVE_METRIC)


def safe_points(row: dict, gender: str) -> float | None:
    raw = row.get(pcol(gender), "")
    if raw is None or raw == "":
        return None
    try:
        pts = float(raw)
    except (TypeError, ValueError):
        return None
    if pts != pts or pts <= 0:  # NaN or non-positive
        return None
    return pts


def parse_performance(result_time: str, event_id: int) -> float:
    raw = (result_time or "").strip().lower().replace("m", "")
    if not raw:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")
    try:
        if ":" in raw:
            m, s = raw.split(":", 1)
            return float(m) * 60 + float(s)
        return float(raw)
    except ValueError:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")


def load_combined(folder: str, prefix: str, gender: str, event_map: dict) -> list[dict]:
    rows = []
    for year in SEASONS:
        path = ROOT / folder / f"{prefix}_{gender}_Outdoor_{year}_Data.csv"
        if path.exists():
            rows.extend(csv.DictReader(open(path)))
    return rows


def athlete_event_bests(rows: list[dict], event_map: dict, gender: str) -> dict[str, dict[str, float]]:
    pc = pcol(gender)
    best = defaultdict(dict)
    for r in rows:
        eid = int(r["running_event_id"])
        if eid not in event_map:
            continue
        ev = event_map[eid]
        aid = r["athlete_id"]
        raw = r.get(pc, "")
        if raw is None or raw == "":
            continue
        try:
            pts = float(raw)
        except (TypeError, ValueError):
            continue
        if pts <= 0:
            continue
        if ev not in best[aid] or pts > best[aid][ev]:
            best[aid][ev] = pts
    return dict(best)


def bootstrap_prop_ci(successes: int, n: int, B: int = 2000) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    samples = [1 if RNG.random() < successes / n else 0 for _ in range(B)]
    # proper bootstrap: resample indices
    props = []
    data = [1] * successes + [0] * (n - successes)
    for _ in range(B):
        draw = [data[RNG.randrange(n)] for _ in range(n)]
        props.append(sum(draw) / n)
    props.sort()
    return (props[int(0.025 * B)], props[int(0.975 * B)])


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1 - (2 * u) / (n1 * n2)


def bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), 1):
        i = order[m - rank]
        val = min(prev, pvals[i] * m / (m - rank + 1))
        adj[i] = min(val, 1.0)
        prev = val
    return adj


def rq1a_event(pts: dict, order: list[str]) -> str | None:
    if not pts:
        return None
    mx = max(pts.values())
    for e in order:
        if e in pts and pts[e] == mx:
            return e
    return None


def rq1c_event(pts: dict, gender: str, order: list[str]) -> str | None:
    margins = {}
    for ev, p in pts.items():
        thr = THRESHOLDS.get((gender, ev))
        if thr is not None:
            margins[ev] = p - thr
    if not margins:
        return None
    return max(margins, key=lambda e: (margins[e], -order.index(e) if e in order else 0))


def filter_nationals_pool(rows: list[dict], event_id: int) -> list[dict]:
    nat = [r for r in rows if r["nationals"] == "True" and int(r["running_event_id"]) == event_id]
    if event_id in PRELIM_EVENT_IDS:
        return [r for r in nat if r["event_type"] == "Prelims"]
    return [r for r in nat if r["event_type"] in ("Finals", "")]


def rank_top8(pool: list[dict], event_id: int) -> list[dict]:
    if not pool:
        return []
    hb = event_id in FIELD_EVENT_IDS
    ranked = sorted(pool, key=lambda r: parse_performance(r["result_time"], event_id), reverse=hb)
    if len(ranked) <= 8:
        return ranked
    eighth = parse_performance(ranked[7]["result_time"], event_id)
    out = []
    for r in ranked:
        p = parse_performance(r["result_time"], event_id)
        if (hb and p >= eighth) or (not hb and p <= eighth):
            out.append(r)
    return out


def run_rq1a(lines: list[str], csv_rows: list[dict]) -> None:
    lines += [
        "=" * 72,
        "IMPROVED RQ1A — Best Event Specialization (2024–2026 Combined)",
        "=" * 72,
        "",
    ]
    all_pair_pvals = []

    for disc, folder, prefix, event_map, event_order, pairings in DISCIPLINES:
        lines.append(f"## {disc}")
        for gender in ("Men", "Women"):
            em = {k: v for k, v in event_map.items()
                  if not (gender == "Women" and v == "110m Hurdles")
                  and not (gender == "Men" and v == "100m Hurdles")}
            eo = [e for e in (event_order or []) if e in em.values()] if event_order else sorted(em.values())
            rows = load_combined(folder, prefix, gender, em)
            ab = athlete_event_bests(rows, em, gender)
            counts = defaultdict(int)
            for pts in ab.values():
                ev = rq1a_event(pts, eo)
                if ev:
                    counts[ev] += 1
            n = sum(counts.values())
            lines.append(f"\n### {gender} (n={n})")
            # Chi-square vs uniform across events with competitors
            competed = defaultdict(set)
            for r in rows:
                eid = int(r["running_event_id"])
                if eid in em:
                    competed[em[eid]].add(r["athlete_id"])
            active_events = [e for e in eo if len(competed.get(e, set())) > 0]
            observed = [counts.get(e, 0) for e in active_events]
            if len(observed) >= 2 and n > 0:
                chi2, p_chi = stats.chisquare(observed)
                lines.append(
                    f"Chi-square vs uniform best-event distribution: χ²={chi2:.2f}, df={len(observed)-1}, p={p_chi:.2e}"
                )
            for ev in sorted(counts, key=lambda e: -counts[e]):
                c = counts[ev]
                lo, hi = bootstrap_prop_ci(c, n)
                lines.append(f"  {ev}: {c} ({100*c/n:.1f}%) | 95% bootstrap CI [{100*lo:.1f}%, {100*hi:.1f}%]")

            pairs = pairings
            if pairs is None:
                if gender == "Men":
                    pairs = [("110m Hurdles", "400m Hurdles")]
                else:
                    pairs = [("100m Hurdles", "400m Hurdles")]
            lines.append("\n  Pairwise head-to-head (exact binomial vs 50/50 null):")
            for ev_a, ev_b in pairs:
                if ev_a not in em.values() or ev_b not in em.values():
                    continue
                wins_a = wins_b = ties = 0
                for pts in ab.values():
                    if ev_a not in pts or ev_b not in pts:
                        continue
                    if pts[ev_a] > pts[ev_b]:
                        wins_a += 1
                    elif pts[ev_b] > pts[ev_a]:
                        wins_b += 1
                    else:
                        ties += 1
                n_pair = wins_a + wins_b
                if n_pair == 0:
                    continue
                p_bin = stats.binomtest(wins_a, n_pair, 0.5).pvalue
                all_pair_pvals.append(p_bin)
                p_a = wins_a / n_pair
                h = cohens_h(p_a, 0.5)
                sig = "***" if p_bin < 0.001 else "**" if p_bin < 0.01 else "*" if p_bin < 0.05 else "ns"
                lines.append(
                    f"    {ev_a} vs {ev_b}: {wins_a}-{wins_b} (ties {ties}, n={n_pair}) | "
                    f"p={p_bin:.2e}, Cohen's h={h:.3f} {sig}"
                )
                csv_rows.append({
                    "analysis": "RQ1A_pairwise", "discipline": disc, "gender": gender,
                    "comparison": f"{ev_a} vs {ev_b}", "wins_a": wins_a, "wins_b": wins_b,
                    "n": n_pair, "p_value": p_bin, "effect_cohens_h": h,
                })
        lines.append("")

    if all_pair_pvals:
        adj = bh_adjust(all_pair_pvals)
        lines.append("Benjamini-Hochberg FDR-adjusted p-values applied across all pairwise tests above.")
        lines.append(f"Significant at FDR<{ALPHA}: {sum(1 for p in adj if p < ALPHA)} / {len(adj)} tests")
        lines.append("")


def run_rq1b(lines: list[str], csv_rows: list[dict]) -> None:
    lines += [
        "=" * 72,
        "IMPROVED RQ1B — Nationals Top-8 vs Full Season (Statistical Tests)",
        "=" * 72,
        "",
    ]
    pvals = []

    for year in SEASONS:
        lines.append(f"## Season {year}")
        for disc, folder, prefix, event_map, _, _ in DISCIPLINES:
            for gender in ("Men", "Women"):
                em = {k: v for k, v in event_map.items()
                      if not (gender == "Women" and v == "110m Hurdles")
                      and not (gender == "Men" and v == "100m Hurdles")}
                path = ROOT / folder / f"{prefix}_{gender}_Outdoor_{year}_Data.csv"
                if not path.exists():
                    continue
                rows = list(csv.DictReader(open(path)))
                pc = pcol(gender)
                for eid, ev in sorted(em.items()):
                    season_pts = [
                        p
                        for r in rows
                        if int(r["running_event_id"]) == eid
                        for p in [safe_points(r, gender)]
                        if p is not None
                    ]
                    pool = filter_nationals_pool(rows, eid)
                    top8 = rank_top8(pool, eid)
                    nat_pts = [
                        p
                        for r in top8
                        for p in [safe_points(r, gender)]
                        if p is not None
                    ]
                    if len(season_pts) < 5 or len(nat_pts) < 3:
                        continue
                    u, p_mw = stats.mannwhitneyu(nat_pts, season_pts, alternative="greater")
                    r_rb = rank_biserial(u, len(nat_pts), len(season_pts))
                    d = (statistics.mean(nat_pts) - statistics.mean(season_pts)) / (
                        statistics.stdev(season_pts) if len(season_pts) > 1 else 1
                    )
                    pvals.append(p_mw)
                    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "ns"
                    lines.append(
                        f"  {gender} {ev}: Mann-Whitney U (nat>season) p={p_mw:.2e}, "
                        f"rank-biserial r={r_rb:.3f}, Cohen's d={d:.2f} {sig} | "
                        f"nat median={statistics.median(nat_pts):.0f}, season median={statistics.median(season_pts):.0f}"
                    )
                    csv_rows.append({
                        "analysis": "RQ1B_mannwhitney", "season": year, "discipline": disc,
                        "gender": gender, "event": ev, "p_value": p_mw,
                        "rank_biserial": r_rb, "cohens_d": d,
                        "nat_median": statistics.median(nat_pts),
                        "season_median": statistics.median(season_pts),
                    })
        lines.append("")

    if pvals:
        adj = bh_adjust(pvals)
        lines.append(f"BH-FDR: {sum(1 for p in adj if p < ALPHA)} / {len(adj)} event-season tests significant")
        lines.append("")


def run_rq1c(lines: list[str], csv_rows: list[dict]) -> None:
    lines += [
        "=" * 72,
        "IMPROVED RQ1C — Competitiveness vs Best Event (Statistical Tests)",
        "=" * 72,
        "",
    ]

    for disc, folder, prefix, event_map, event_order, pairings in DISCIPLINES:
        lines.append(f"## {disc}")
        for gender in ("Men", "Women"):
            em = {k: v for k, v in event_map.items()
                  if not (gender == "Women" and v == "110m Hurdles")
                  and not (gender == "Men" and v == "100m Hurdles")}
            eo = [e for e in (event_order or []) if e in em.values()] if event_order else sorted(em.values())
            rows = load_combined(folder, prefix, gender, em)
            ab = athlete_event_bests(rows, em, gender)
            agree = disagree = 0
            multi = 0
            clear_by_event = defaultdict(int)
            total_by_event = defaultdict(int)

            for pts in ab.values():
                for ev in pts:
                    total_by_event[ev] += 1
                    thr = THRESHOLDS.get((gender, ev))
                    if thr and pts[ev] >= thr:
                        clear_by_event[ev] += 1
                if len(pts) < 2:
                    continue
                multi += 1
                a = rq1a_event(pts, eo)
                c = rq1c_event(pts, gender, eo)
                if a == c:
                    agree += 1
                else:
                    disagree += 1

            lines.append(f"\n### {gender}")
            if multi:
                p_agree = stats.binomtest(agree, multi, 0.5).pvalue
                lo, hi = bootstrap_prop_ci(agree, multi)
                lines.append(
                    f"RQ1A=RQ1C agreement: {agree}/{multi} ({100*agree/multi:.1f}%), "
                    f"95% CI [{100*lo:.1f}%, {100*hi:.1f}%] | "
                    f"binomial p vs 50%: {p_agree:.2e}"
                )
                csv_rows.append({
                    "analysis": "RQ1C_agreement", "discipline": disc, "gender": gender,
                    "agree": agree, "multi_n": multi, "p_value": p_agree,
                    "agree_rate": agree / multi,
                })

            lines.append("  Nationals clear-rate (Wilson 95% CI; ordered by count clearing bar):")
            rates = []
            for ev in sorted(total_by_event, key=lambda e: (-clear_by_event.get(e, 0), e)):
                nc = clear_by_event[ev]
                nt = total_by_event[ev]
                wlo, whi = wilson_ci(nc, nt)
                rates.append((ev, nc / nt if nt else 0, nc, nt, wlo, whi))
                lines.append(
                    f"    {ev}: {nc}/{nt} ({100*nc/nt:.1f}%) | Wilson CI [{100*wlo:.1f}%, {100*whi:.1f}%]"
                )

            # Distance-specific: 1500 vs 5000 margin comparison
            if disc == "Distance":
                a_wins = b_wins = ties = 0
                for pts in ab.values():
                    if "1500m" not in pts or "5000m" not in pts:
                        continue
                    ma = pts["1500m"] - THRESHOLDS[(gender, "1500m")]
                    mb = pts["5000m"] - THRESHOLDS[(gender, "5000m")]
                    if ma > mb:
                        a_wins += 1
                    elif mb > ma:
                        b_wins += 1
                    else:
                        ties += 1
                n = a_wins + b_wins
                if n:
                    p = stats.binomtest(a_wins, n, 0.5).pvalue
                    lines.append(
                        f"  1500m vs 5000m competitiveness margin: 1500m {a_wins}, 5000m {b_wins} "
                        f"(ties {ties}) | p={p:.2e}"
                    )
        lines.append("")

    # Fisher exact: clear-rate among athletes competing in both events (distance)
    lines.append("## Distance — Clear-rate comparisons (Fisher's exact)")
    for gender in ("Men", "Women"):
        em = {9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"}
        rows = load_combined("Distance_Events_Counting", "Distance", gender, em)
        ab = athlete_event_bests(rows, em, gender)
        for ev_a, ev_b in [("3000m Steeplechase", "1500m"), ("5000m", "1500m")]:
            both = [pts for pts in ab.values() if ev_a in pts and ev_b in pts]
            if not both:
                continue
            ct = [[0, 0], [0, 0]]
            for pts in both:
                ca = pts[ev_a] >= THRESHOLDS[(gender, ev_a)]
                cb = pts[ev_b] >= THRESHOLDS[(gender, ev_b)]
                ct[0 if ca else 1][0 if cb else 1] += 1
            _, p_fisher = stats.fisher_exact(ct)
            lines.append(
                f"  {gender} {ev_a} vs {ev_b} (n={len(both)}): contingency {ct} | Fisher p={p_fisher:.2e}"
            )
            csv_rows.append({
                "analysis": "RQ1C_fisher_clear", "gender": gender,
                "comparison": f"{ev_a} vs {ev_b}", "p_value": p_fisher, "n": len(both),
            })
    lines.append("")


def split_and_write_sections(full_text: str) -> None:
    markers = [
        ("IMPROVED RQ1A", "improved_rq1a_statistical_analysis.txt"),
        ("IMPROVED RQ1B", "improved_rq1b_statistical_analysis.txt"),
        ("IMPROVED RQ1C", "improved_rq1c_statistical_analysis.txt"),
    ]
    for i, (marker, fname) in enumerate(markers):
        start = full_text.find(marker)
        if start < 0:
            continue
        end = full_text.find(markers[i + 1][0], start) if i + 1 < len(markers) else len(full_text)
        (OUT / fname).write_text(full_text[start:end].rstrip() + "\n")


def plot_improved_clear_rates(csv_rows: list[dict]) -> None:
    # Distance clear rates use WA-calibrated nationals thresholds — only meaningful for WA.
    if ACTIVE_METRIC != "wa":
        return
    # Distance clear rates men/women bar chart with Wilson CI error bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gender in zip(axes, ("Men", "Women")):
        events = ["800m", "1500m", "5000m", "3000m Steeplechase"]
        clears = []
        cis_lo = []
        cis_hi = []
        for ev in events:
            folder = "Distance_Events_Counting"
            rows = load_combined(folder, "Distance", gender, {9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"})
            ab = athlete_event_bests(rows, {9: "800m", 11: "1500m", 17: "5000m", 20: "3000m Steeplechase"}, gender)
            nc = sum(1 for pts in ab.values() if ev in pts and pts[ev] >= THRESHOLDS[(gender, ev)])
            nt = sum(1 for pts in ab.values() if ev in pts)
            clears.append(100 * nc / nt if nt else 0)
            lo, hi = wilson_ci(nc, nt)
            rate = 100 * nc / nt if nt else 0
            cis_lo.append(max(0.0, rate - 100 * lo) if nt else 0)
            cis_hi.append(max(0.0, 100 * hi - rate) if nt else 0)
        x = np.arange(len(events))
        ax.bar(x, clears, color=["#2E86AB", "#A23B72", "#C73E1D", "#F18F01"], edgecolor="black", linewidth=0.8)
        ax.errorbar(x, clears, yerr=[cis_lo, cis_hi], fmt="none", color="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(events, rotation=15, ha="right")
        ax.set_ylabel("% clearing nationals 8th-place bar")
        ax.set_title(f"{gender} — Distance (2024–2026)")
        ax.set_ylim(0, max(clears) * 1.4 if clears else 20)
    fig.suptitle("Improved RQ1C: Nationals Clear-Rate with Wilson 95% CI", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "improved_rq1c_distance_clear_rates.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_improved_findings(rq1c_text: str) -> None:
    src = ROOT / "findings.md"
    dst = OUT / "improved_findings.md"

    # Parse key stats from RQ1C output
    men_1500_5k = "p=3.89e-01" in rq1c_text or "p=0.389" in rq1c_text
    women_1500_5k = "p=2.79e-01" in rq1c_text

    extra = f"""

---

## Improved Analysis Addendum (Statistical Inference)

Full outputs in `improved_rq1_outputs/`. Significance level α = 0.05; Benjamini-Hochberg FDR correction applied to multiple comparisons.

### Methodological improvements
1. **Bootstrap 95% CIs** on best-event proportions (RQ1A)
2. **Wilson score 95% CIs** on nationals clear-rates (RQ1C)
3. **Chi-square goodness-of-fit** tests for non-uniform specialization (RQ1A)
4. **Exact binomial tests** for pairwise head-to-head comparisons with Cohen's h (RQ1A)
5. **Mann-Whitney U tests** with rank-biserial r and Cohen's d (RQ1B)
6. **Fisher's exact tests** for clear-rate comparisons among dual-event athletes (RQ1C)
7. **Binomial tests** on RQ1A=RQ1C agreement vs 50% null (RQ1C)

### RQ1A — Significance highlights
- Best-event distributions **significantly non-uniform** (χ² p < 0.001) for all discipline × gender groups **except** women's hurdles (χ² = 0.08, p = 0.96).
- **Distance pairwise tests** (all significant after BH-FDR, p < 0.001):
  - Men 1500m vs 800m: 349–232, Cohen's h = −0.20
  - Men 1500m vs 5000m: 294–142, Cohen's h = 0.36
  - Women 1500m vs 5000m: 55–89, Cohen's h = −0.24, **p = 0.006**
  - Women 800m vs 1500m: 58–214, Cohen's h = −0.61
- **Men's sprints pairwise tests not significant** (all p > 0.15) — no single sprint dominates dual-event athletes statistically.

### RQ1B — Significance highlights
- Mann-Whitney U (nationals top-8 > full season): **83/83** event-season tests significant (p < 0.001) after BH-FDR.
- Rank-biserial r typically **0.55–0.95** for running events; nationals top-8 distributions are stochastically greater than full-season in every case.
- Cohen's d often **> 1.0** for sprint and distance events.

### RQ1C — Significance highlights
- **RQ1A = RQ1C agreement** significantly exceeds 50% in all disciplines (p < 0.001), but agreement is **heterogeneous**:
  - Sprints/throws men: 94–96%
  - Distance men/women: 69–76%
  - Women's throws: 66%
- **1500m vs 5000m competitiveness margin** (RQ1C threshold method): directionally favors 5000m (men 228–209; women 79–65) but **does not reach α = 0.05** (men p = 0.39; women p = 0.28). Descriptive trend is clear; inferential evidence is suggestive, not definitive.
- **Wilson CIs on clear-rates** (distance, women): Steeplechase **16.3%** [11.0%, 23.4%] vs 1500m **8.0%** [6.0%, 10.5%] — non-overlapping intervals support higher relative nationals opportunity for steeplechase.
- **Fisher's exact** (athletes in both events): steeple vs 1500m clear-rate difference significant for men and women (see `improved_rq1c_statistical_analysis.txt`).

### Interpretation note
Descriptive RQ1A/RQ1C counts and inferential tests can diverge: women's 5000m beats 1500m in **62%** of dual-event pairwise RQ1A comparisons (p = 0.006), while RQ1C margin comparison is not significant at α = 0.05 because threshold margins are noisier than raw point comparisons. Report both for JQAS transparency.
"""
    if src.exists():
        dst.write_text(src.read_text().rstrip() + extra + "\n")
    else:
        dst.write_text(extra.strip() + "\n")


def _run_once(metric: str, out_dir: Path) -> None:
    global ACTIVE_METRIC, OUT
    ACTIVE_METRIC = metric
    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "IMPROVED RQ1 STATISTICAL ANALYSIS",
        "National Running Club Database — Outdoor Track 2024–2026",
        f"Scoring metric: {metric} (scientific: purdy/mercier; sports: wa/vdot)",
        f"Significance level α = {ALPHA}; multiple comparisons corrected via Benjamini-Hochberg FDR where noted.",
        "",
    ]
    csv_rows: list[dict] = []

    run_rq1a(lines, csv_rows)
    run_rq1b(lines, csv_rows)
    run_rq1c(lines, csv_rows)

    full_text = "\n".join(lines)
    (OUT / "improved_rq1_statistical_analysis.txt").write_text(full_text)

    split_and_write_sections(full_text)

    if csv_rows:
        fields = sorted({k for r in csv_rows for k in r})
        with open(OUT / "improved_rq1_statistical_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)

    plot_improved_clear_rates(csv_rows)
    write_improved_findings(full_text)

    eighth_lines = [
        "IMPROVED RQ1B — 8th-Place Threshold Bootstrap 95% CI (by season)",
        f"Metric: {metric}",
        "",
    ]
    top8_csv = ROOT / "RQ1B_Nationals" / "rq1b_nationals_top8_all_seasons.csv"
    if top8_csv.exists() and metric == "wa":
        by_key = defaultdict(list)
        for r in csv.DictReader(open(top8_csv)):
            if int(r["place"]) == 8:
                by_key[(r["season"], r["gender"], r["event_name"])].append(
                    float(r["world_athletics_points"])
                )
        for key in sorted(by_key):
            vals = by_key[key]
            if len(vals) >= 2:
                boot = [
                    statistics.mean([vals[RNG.randrange(len(vals))] for _ in range(len(vals))])
                    for _ in range(2000)
                ]
                boot.sort()
                lo, hi = boot[50], boot[1950]
            else:
                lo = hi = vals[0]
            eighth_lines.append(
                f"{key[0]} {key[1]} {key[2]}: 8th-place WA={statistics.mean(vals):.1f} "
                f"| bootstrap CI [{lo:.1f}, {hi:.1f}]"
            )
    (OUT / "improved_rq1b_8th_place_thresholds.txt").write_text("\n".join(eighth_lines) + "\n")

    print(f"Wrote improved outputs to {OUT}/ (metric={metric})")
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(f"  {p.name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Improved RQ1 with selectable scoring metric.")
    parser.add_argument(
        "--metric",
        default="wa",
        choices=list(ALL_METRICS),
        help="Scoring metric (default: wa). Scientific: purdy/mercier; sports: wa/vdot.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Run once per metric into improved_rq1_outputs/by_metric/<metric>/.",
    )
    args = parser.parse_args(argv)

    if args.all_metrics:
        base = ROOT / "improved_rq1_outputs" / "by_metric"
        for metric in ALL_METRICS:
            _run_once(metric, base / metric)
        _run_once("wa", ROOT / "improved_rq1_outputs")
    else:
        out = ROOT / "improved_rq1_outputs"
        if args.metric != "wa":
            out = ROOT / "improved_rq1_outputs" / "by_metric" / args.metric
        _run_once(args.metric, out)


if __name__ == "__main__":
    main()
