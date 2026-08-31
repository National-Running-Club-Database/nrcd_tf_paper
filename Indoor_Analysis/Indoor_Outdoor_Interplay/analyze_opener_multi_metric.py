"""Indoor vs outdoor season-opener comparison — multi-metric (band 750–950).

Same protocol as analyze_indoor_outdoor_opener_wa.py band analysis:
  • ≥2 indoor and ≥2 outdoor meets; similar-event pairs at both openers
  • Inclusion: outdoor opener WA ∈ [750, 950)
  • Delta = outdoor_opener_score − indoor_opener_score under each metric

World Athletics band artifacts are copied from the parent folder (not regenerated).
VDOT / Purdy / Mercier are recomputed.

Outputs → Indoor_Outdoor_Interplay/multiple_metrics/
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring.columns import metric_framing, points_col
from scoring.marks import is_field_event

# Reuse pair definitions and path helpers from the WA opener script
from indoor_analysis.Indoor_Outdoor_Interplay.analyze_indoor_outdoor_opener_wa import (  # noqa: E402
    EVENT_NAMES,
    RELAY_IDS,
    SEASONS,
    SIMILAR_PAIRS,
    eligible_athletes,
    gender_from_row,
    iter_indoor_paths,
    iter_outdoor_paths,
    opener_date,
    opener_meet_id,
    parse_aid,
    parse_event_id,
)

INDOOR_ROOT = PROJECT_ROOT / "indoor_analysis"
PARENT = INDOOR_ROOT / "Indoor_Outdoor_Interplay"
OUT_DIR = PARENT / "multiple_metrics"
BAND = (750, 950)

METRIC_LABELS = {
    "wa": "World Athletics Points",
    "vdot": "VDOT",
    "purdy": "Gardner–Purdy",
    "mercier": "Mercier (1999)",
}


@dataclass
class ResultRow:
    athlete_id: str
    gender: str
    year: str
    season: str
    meet_id: str
    start_date: str
    event_id: int
    event_name: str
    scores: dict[str, float]
    result_id: str


@dataclass
class PairCompare:
    year: str
    athlete_id: str
    gender: str
    pair_label: str
    indoor_event: str
    outdoor_event: str
    indoor_score: float
    outdoor_score: float
    outdoor_wa: float
    delta: float
    indoor_opener_date: str
    outdoor_opener_date: str
    n_indoor_meets: int
    n_outdoor_meets: int
    metric: str


def _score_map(row: dict, gender: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ("wa", "vdot", "purdy", "mercier"):
        col = points_col(gender, metric)
        try:
            val = float(row.get(col) or 0)
        except (TypeError, ValueError):
            continue
        if val > 0 and math.isfinite(val):
            out[metric] = val
    return out


def load_results(year: str, season: str) -> list[ResultRow]:
    paths = iter_indoor_paths(year) if season == "indoor" else iter_outdoor_paths(year)
    seen: set[str] = set()
    out: list[ResultRow] = []
    for path, file_gender in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                eid = parse_event_id(row)
                if eid is None or eid in RELAY_IDS:
                    continue
                if eid not in EVENT_NAMES:
                    continue
                aid = parse_aid(row.get("athlete_id", ""))
                if not aid:
                    continue
                mid = (row.get("meet_id") or "").strip()
                date = (row.get("start_date") or "").strip()[:10]
                if not mid or not date:
                    continue
                gender = gender_from_row(row, file_gender)
                scores = _score_map(row, gender)
                if "wa" not in scores:
                    continue
                ename = EVENT_NAMES[eid]
                rid = (row.get("result_id") or "").strip()
                key = rid or f"{aid}|{mid}|{eid}|{date}|{scores['wa']}|{season}"
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    ResultRow(
                        athlete_id=aid,
                        gender=gender,
                        year=year,
                        season=season,
                        meet_id=mid,
                        start_date=date,
                        event_id=eid,
                        event_name=ename,
                        scores=scores,
                        result_id=rid or key,
                    )
                )
    return out


def best_score_at_meet(
    rows: list[ResultRow], meet_id: str, event_id: int, metric: str
) -> float | None:
    vals = [
        r.scores[metric]
        for r in rows
        if r.meet_id == meet_id and r.event_id == event_id and metric in r.scores
    ]
    return max(vals) if vals else None


def best_wa_at_meet(rows: list[ResultRow], meet_id: str, event_id: int) -> float | None:
    return best_score_at_meet(rows, meet_id, event_id, "wa")


def build_comparisons(year: str, metric: str) -> tuple[list[PairCompare], dict]:
    indoor_all = load_results(year, "indoor")
    outdoor_all = load_results(year, "outdoor")
    # eligible_athletes expects objects with athlete_id, meet_id, gender
    eligible = eligible_athletes(indoor_all, outdoor_all)  # type: ignore[arg-type]

    by_aid_i: dict[str, list[ResultRow]] = defaultdict(list)
    by_aid_o: dict[str, list[ResultRow]] = defaultdict(list)
    for r in indoor_all:
        if r.athlete_id in eligible:
            by_aid_i[r.athlete_id].append(r)
    for r in outdoor_all:
        if r.athlete_id in eligible:
            by_aid_o[r.athlete_id].append(r)

    comparisons: list[PairCompare] = []
    for aid, meta in eligible.items():
        i_rows = by_aid_i[aid]
        o_rows = by_aid_o[aid]
        i_meet = opener_meet_id(i_rows)  # type: ignore[arg-type]
        o_meet = opener_meet_id(o_rows)  # type: ignore[arg-type]
        if not i_meet or not o_meet:
            continue
        i_date = opener_date(i_rows, i_meet)  # type: ignore[arg-type]
        o_date = opener_date(o_rows, o_meet)  # type: ignore[arg-type]
        gender = meta["gender"]

        for label, i_eid, o_eid, genders in SIMILAR_PAIRS:
            if genders is not None and gender not in genders:
                continue
            i_name = EVENT_NAMES[i_eid]
            o_name = EVENT_NAMES[o_eid]
            if metric in {"purdy", "vdot"} and (
                is_field_event(i_name) or is_field_event(o_name)
            ):
                continue
            i_sc = best_score_at_meet(i_rows, i_meet, i_eid, metric)
            o_sc = best_score_at_meet(o_rows, o_meet, o_eid, metric)
            o_wa = best_wa_at_meet(o_rows, o_meet, o_eid)
            if i_sc is None or o_sc is None or o_wa is None:
                continue
            comparisons.append(
                PairCompare(
                    year=year,
                    athlete_id=aid,
                    gender=gender,
                    pair_label=label,
                    indoor_event=i_name,
                    outdoor_event=o_name,
                    indoor_score=i_sc,
                    outdoor_score=o_sc,
                    outdoor_wa=o_wa,
                    delta=o_sc - i_sc,
                    indoor_opener_date=i_date,
                    outdoor_opener_date=o_date,
                    n_indoor_meets=meta["n_indoor_meets"],
                    n_outdoor_meets=meta["n_outdoor_meets"],
                    metric=metric,
                )
            )

    meta_out = {
        "year": year,
        "n_eligible_athletes": len(eligible),
        "n_pair_comparisons": len(comparisons),
        "n_athletes_with_pair": len({c.athlete_id for c in comparisons}),
    }
    return comparisons, meta_out


def group_stats(rows: list[PairCompare]) -> dict:
    if not rows:
        return {
            "n": 0,
            "n_athletes": 0,
            "mean_indoor": None,
            "mean_outdoor": None,
            "mean_delta": None,
            "median_delta": None,
            "pct_outdoor_higher": None,
            "pct_indoor_higher": None,
            "pct_tie": None,
        }
    deltas = [r.delta for r in rows]
    n = len(rows)
    out_h = sum(1 for d in deltas if d > 0)
    in_h = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    return {
        "n": n,
        "n_athletes": len({r.athlete_id for r in rows}),
        "mean_indoor": round(statistics.mean(r.indoor_score for r in rows), 2),
        "mean_outdoor": round(statistics.mean(r.outdoor_score for r in rows), 2),
        "mean_delta": round(statistics.mean(deltas), 2),
        "median_delta": round(statistics.median(deltas), 2),
        "pct_outdoor_higher": round(100.0 * out_h / n, 1),
        "pct_indoor_higher": round(100.0 * in_h / n, 1),
        "pct_tie": round(100.0 * ties / n, 1),
    }


def fmt_stat(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    return (
        f"n={s['n']} athletes={s['n_athletes']}  "
        f"indoorμ={s['mean_indoor']:.1f} outdoorμ={s['mean_outdoor']:.1f}  "
        f"Δμ={s['mean_delta']:+.1f} Δmed={s['median_delta']:+.1f}  "
        f"outdoor_higher={s['pct_outdoor_higher']:.1f}%  "
        f"indoor_higher={s['pct_indoor_higher']:.1f}%"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_report(
    metric: str,
    band_comp: list[PairCompare],
    path: Path,
) -> None:
    label = METRIC_LABELS[metric]
    framing = metric_framing(metric)
    lo, hi = BAND
    overall = group_stats(band_comp)

    year_meta = []
    for year in SEASONS:
        year_rows = [c for c in band_comp if c.year == year]
        year_meta.append(
            {
                "year": year,
                "n_eligible_athletes": len({c.athlete_id for c in year_rows}),
                "n_pair_comparisons": len(year_rows),
                "n_athletes_with_pair": len({c.athlete_id for c in year_rows}),
            }
        )

    title = (
        f"Indoor vs Outdoor Season-Opener {label} "
        f"(outdoor opener WA in [{lo}, {hi}))"
    )
    lines = [
        title,
        "=" * len(title),
        f"Metric: {label} [{framing}]",
        "",
        "Question:",
        "  For athletes with ≥2 indoor meets and ≥2 outdoor meets in the same year,",
        f"  do outdoor season openers start with higher {label} than indoor season",
        "  openers for similar events?",
        f"  Restricted to comparisons where the outdoor opener result is in WA",
        f"  band [{lo}, {hi}).",
        "",
        "Method:",
        "  • Years: 2024, 2025, 2026 (indoor season file year paired with outdoor year).",
        "  • Eligibility: ≥2 distinct meet_ids with individual (non-relay) results indoors",
        "    AND ≥2 outdoors in that year.",
        "  • Season opener: earliest meet by start_date for that athlete in the season.",
        "  • Similar-event pairs compared only when the athlete contested the indoor",
        "    event at the indoor opener meet AND the paired outdoor event at the",
        "    outdoor opener meet.",
        f"  • Delta = outdoor_opener_{label} − indoor_opener_{label}.",
        "  • Outdoor distance uses new_steeplechase_data when available.",
        "  • Relays excluded.",
        f"  • Band filter: keep a pair only if outdoor opener WA ∈ [{lo}, {hi}).",
        "  • Purdy/VDOT: field-event pairs omitted (undefined).",
        "",
        "Similar-event pairs:",
    ]
    for label_p, *_rest in SIMILAR_PAIRS:
        lines.append(f"  • {label_p}")

    lines += ["", "Eligibility / sample", "-" * 20]
    for ym in year_meta:
        lines.append(
            f"  {ym['year']}: {ym['n_eligible_athletes']} athletes in band; "
            f"{ym['n_pair_comparisons']} opener pair comparisons "
            f"({ym['n_athletes_with_pair']} athletes with ≥1 comparable opener pair)"
        )
    lines.append(
        f"  Combined: {overall['n_athletes']} athletes contributing "
        f"{overall['n']} opener pair comparisons"
    )

    lines += [
        "",
        "Primary result — all years, all similar-event pairs",
        "-" * 51,
        f"  {fmt_stat(overall)}",
    ]
    if overall["n"] and overall["mean_delta"] is not None:
        if overall["mean_delta"] > 0:
            verdict = (
                f"YES — outdoor openers average higher {label} than indoor openers "
                "for comparable events among dual-season athletes."
            )
        else:
            verdict = (
                f"NO — indoor openers average higher {label} than outdoor openers "
                "for comparable events among dual-season athletes."
            )
        lines.append(f"  Verdict: {verdict}")

    lines += ["", "By year", "-" * 7]
    for year in SEASONS:
        s = group_stats([c for c in band_comp if c.year == year])
        lines.append(f"  {year}: {fmt_stat(s)}")

    lines += ["", "By gender", "-" * 9]
    for gender in ("Men", "Women"):
        s = group_stats([c for c in band_comp if c.gender == gender])
        lines.append(f"  {gender}: {fmt_stat(s)}")

    lines += ["", "By similar-event pair", "-" * 21]
    for pair_label, *_ in SIMILAR_PAIRS:
        s = group_stats([c for c in band_comp if c.pair_label == pair_label])
        if s["n"] == 0:
            continue
        lines.append(f"  {pair_label}: {fmt_stat(s)}")

    lines += ["", "By gender × event pair (n≥10)", "-" * 29]
    for gender in ("Men", "Women"):
        for pair_label, *_ in SIMILAR_PAIRS:
            sub = [
                c
                for c in band_comp
                if c.gender == gender and c.pair_label == pair_label
            ]
            s = group_stats(sub)
            if s["n"] < 10:
                continue
            lines.append(f"  {gender} | {pair_label}: {fmt_stat(s)}")

    # Athlete-level mean delta
    by_ath: dict[tuple[str, str], list[float]] = defaultdict(list)
    for c in band_comp:
        by_ath[(c.year, c.athlete_id)].append(c.delta)
    if by_ath:
        ath_means = [statistics.mean(v) for v in by_ath.values()]
        n_ath = len(ath_means)
        n_pos = sum(1 for m in ath_means if m > 0)
        lines += [
            "",
            "Athlete-level (mean delta across an athlete-year's comparable opener pairs)",
            "-" * 75,
            f"  n_athletes={n_ath}  mean_of_means={statistics.mean(ath_means):+.2f}  "
            f"median_of_means={statistics.median(ath_means):+.2f}  "
            f"pct_athletes_outdoor_higher={100.0 * n_pos / n_ath:.1f}%",
        ]

    lines += [
        "",
        "Caveats:",
        "  • Only opener-meet results count; later-season form is ignored.",
        "  • Athlete must run the paired events at BOTH openers — selection into",
        "    the comparison set is not random.",
        f"  • This file restricts to outdoor opener WA ∈ [{lo}, {hi}); indoor opener",
        "    score is unrestricted (may fall outside a comparable band).",
        "",
        "Source: indoor_analysis/Indoor_Outdoor_Interplay/analyze_opener_multi_metric.py",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_wa_artifacts() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = [
        (
            "indoor_outdoor_opener_wa_findings_band_750_950.txt",
            "opener_findings_band_750_950_wa.txt",
        ),
        (
            "opener_wa_pair_comparisons_band_750_950.csv",
            "opener_pair_comparisons_band_750_950_wa.csv",
        ),
        (
            "opener_wa_summary_slices_band_750_950.csv",
            "opener_summary_slices_band_750_950_wa.csv",
        ),
    ]
    for src_name, dst_name in mapping:
        src = PARENT / src_name
        dst = OUT_DIR / dst_name
        if not src.exists():
            raise SystemExit(f"Missing WA source: {src}")
        if src.suffix == ".txt":
            text = src.read_text(encoding="utf-8")
            lines = text.splitlines()
            if lines:
                out = [
                    lines[0],
                    "Metric: World Athletics Points [sports]",
                    "(Copied from parent Indoor_Outdoor_Interplay/; not regenerated.)",
                ]
                # skip duplicate blank after title underline
                rest = lines[1:]
                if rest and rest[0].startswith("="):
                    out.append(rest[0])
                    rest = rest[1:]
                out.extend(rest)
                dst.write_text("\n".join(out) + "\n", encoding="utf-8")
            else:
                dst.write_text(text, encoding="utf-8")
        else:
            # Add metric column to CSVs for consistency
            with open(src, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                dst.write_text("")
                continue
            out_rows = [{"metric": "wa", **r} for r in rows]
            # put metric first
            fields = ["metric"] + [k for k in rows[0].keys() if k != "metric"]
            with open(dst, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(out_rows)
        print(f"Copied WA → {dst.relative_to(PROJECT_ROOT)}")


def run_metric(metric: str) -> None:
    lo, hi = BAND
    all_comp: list[PairCompare] = []
    for year in SEASONS:
        print(f"  {metric} {year}...")
        comps, _meta = build_comparisons(year, metric)
        all_comp.extend(comps)

    band_comp = [c for c in all_comp if lo <= c.outdoor_wa < hi]
    detail = [
        {
            "metric": metric,
            "year": c.year,
            "athlete_id": c.athlete_id,
            "gender": c.gender,
            "pair_label": c.pair_label,
            "indoor_event": c.indoor_event,
            "outdoor_event": c.outdoor_event,
            "indoor_opener_date": c.indoor_opener_date,
            "outdoor_opener_date": c.outdoor_opener_date,
            "indoor_score": round(c.indoor_score, 4),
            "outdoor_score": round(c.outdoor_score, 4),
            "outdoor_wa": round(c.outdoor_wa, 2),
            "delta_outdoor_minus_indoor": round(c.delta, 4),
            "outdoor_higher": c.delta > 0,
            "n_indoor_meets": c.n_indoor_meets,
            "n_outdoor_meets": c.n_outdoor_meets,
        }
        for c in band_comp
    ]
    summary = []
    slices: list[tuple[str, list[PairCompare]]] = [("all", band_comp)]
    for year in SEASONS:
        slices.append((f"year_{year}", [c for c in band_comp if c.year == year]))
    for gender in ("Men", "Women"):
        slices.append((f"gender_{gender}", [c for c in band_comp if c.gender == gender]))
    for label, *_ in SIMILAR_PAIRS:
        slices.append((f"pair_{label}", [c for c in band_comp if c.pair_label == label]))
    for name, rows in slices:
        s = group_stats(rows)
        if s["n"] == 0:
            continue
        summary.append({"metric": metric, "slice": name, **s})

    detail_path = OUT_DIR / f"opener_pair_comparisons_band_750_950_{metric}.csv"
    summary_path = OUT_DIR / f"opener_summary_slices_band_750_950_{metric}.csv"
    report_path = OUT_DIR / f"opener_findings_band_750_950_{metric}.txt"
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    write_report(metric, band_comp, report_path)
    print(f"Wrote {report_path.relative_to(PROJECT_ROOT)} (n={len(band_comp)})")


def update_readme() -> None:
    readme = OUT_DIR / "README.txt"
    extra = """
Season-opener indoor vs outdoor (band 750–950)
----------------------------------------------
  opener_findings_band_750_950_wa.txt            (copied)
  opener_findings_band_750_950_vdot.txt
  opener_findings_band_750_950_purdy.txt
  opener_findings_band_750_950_mercier.txt
  opener_pair_comparisons_band_750_950_{metric}.csv
  opener_summary_slices_band_750_950_{metric}.csv

  Regenerate opener non-WA metrics:
    python indoor_analysis/Indoor_Outdoor_Interplay/analyze_opener_multi_metric.py
""".strip()
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if "opener_findings_band_750_950" not in text:
            text = text.rstrip() + "\n\n" + extra + "\n"
            readme.write_text(text, encoding="utf-8")
    else:
        readme.write_text(extra + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_wa_artifacts()
    for metric in ("vdot", "purdy", "mercier"):
        print(f"Running opener analysis for {metric}...")
        run_metric(metric)
    update_readme()
    print(f"Done. Outputs in {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
