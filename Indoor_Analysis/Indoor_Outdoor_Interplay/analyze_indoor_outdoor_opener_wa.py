"""Indoor vs outdoor season-opener WA comparison for dual-season athletes.

Question: Among athletes with ≥2 indoor meets and ≥2 outdoor meets in the same
year, do outdoor season openers start at higher World Athletics scores than
indoor season openers for similar events?

Year definition: indoor file year Y (typically Dec Y-1–Mar Y) paired with
outdoor file year Y (typically Mar–Jul Y).

Season opener: chronologically earliest meet (by start_date) with ≥1 individual
(non-relay) result for that athlete in the season.

Similar-event pairs (indoor → outdoor):
  60m→100m, 200m→200m, 400m→400m, 800m→800m,
  Mile→1500m, 1500m→1500m, 3000m→5000m, 5000m→5000m,
  60m Hurdles→110m Hurdles (Men) / 100m Hurdles (Women),
  LJ/TJ/HJ/Shot→same outdoor event.

Comparison unit: athlete-year × similar-event pair where the athlete has a
valid WA result in the indoor event at the indoor opener meet AND in the
paired outdoor event at the outdoor opener meet.

Delta = outdoor_opener_WA − indoor_opener_WA  (positive ⇒ outdoor opener higher).
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDOOR_ROOT = PROJECT_ROOT / "indoor_analysis"
OUTPUT_ROOT = INDOOR_ROOT / "Indoor_Outdoor_Interplay"

SEASONS = ("2024", "2025", "2026")
RELAY_IDS = {21, 22, 23, 24, 25, 26, 29, 30, 31}

EVENT_NAMES = {
    2: "60m",
    3: "100m",
    4: "200m",
    6: "400m",
    9: "800m",
    11: "1500m",
    13: "Mile",
    14: "3000m",
    17: "5000m",
    33: "60m Hurdles",
    34: "100m Hurdles",
    35: "110m Hurdles",
    38: "Long Jump",
    39: "Triple Jump",
    40: "High Jump",
    41: "Shot Put",
}

# (pair_label, indoor_event_id, outdoor_event_id, genders or None=both)
SIMILAR_PAIRS: list[tuple[str, int, int, tuple[str, ...] | None]] = [
    ("60m → 100m", 2, 3, None),
    ("200m → 200m", 4, 4, None),
    ("400m → 400m", 6, 6, None),
    ("800m → 800m", 9, 9, None),
    ("Mile → 1500m", 13, 11, None),
    ("1500m → 1500m", 11, 11, None),
    ("3000m → 5000m", 14, 17, None),
    ("5000m → 5000m", 17, 17, None),
    ("60mH → 110mH", 33, 35, ("Men",)),
    ("60mH → 100mH", 33, 34, ("Women",)),
    ("Long Jump → Long Jump", 38, 38, None),
    ("Triple Jump → Triple Jump", 39, 39, None),
    ("High Jump → High Jump", 40, 40, None),
    ("Shot Put → Shot Put", 41, 41, None),
]


@dataclass
class ResultRow:
    athlete_id: str
    gender: str
    year: str
    season: str  # indoor | outdoor
    meet_id: str
    start_date: str
    event_id: int
    event_name: str
    wa: float
    result_id: str


def points_col(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


def parse_aid(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    try:
        return str(int(float(raw)))
    except ValueError:
        return None


def parse_event_id(row: dict) -> int | None:
    try:
        return int(float(row["running_event_id"]))
    except (TypeError, ValueError, KeyError):
        return None


def gender_from_row(row: dict, file_gender: str) -> str:
    g = (row.get("gender") or "").strip().upper()
    if g in ("M", "MALE"):
        return "Men"
    if g in ("F", "FEMALE", "W"):
        return "Women"
    return file_gender


def iter_indoor_paths(year: str) -> list[tuple[Path, str]]:
    out = []
    for folder in sorted(INDOOR_ROOT.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("Indoor_"):
            continue
        if folder.name == "Indoor_Outdoor_Interplay":
            continue
        group = folder.name.replace("Indoor_", "")
        for gender in ("Men", "Women"):
            path = folder / f"Indoor_Relays_{group}_{gender}_{year}_Data.csv"
            if path.exists():
                out.append((path, gender))
    return out


def iter_outdoor_paths(year: str) -> list[tuple[Path, str]]:
    sources = [
        (PROJECT_ROOT / "relays_findings" / "Sprinters_Relays_Findings", "Sprinters"),
        (PROJECT_ROOT / "relays_findings" / "Hurdles_Relays_Findings", "Hurdles"),
        (PROJECT_ROOT / "relays_findings" / "Jumps_Relays_Findings", "Jumps"),
        (PROJECT_ROOT / "relays_findings" / "Throws_Relays_Findings", "Throws"),
        (PROJECT_ROOT / "new_steeplechase_data", "Distance"),
    ]
    out = []
    for folder, prefix in sources:
        for gender in ("Men", "Women"):
            path = folder / f"Relays_{prefix}_{gender}_Outdoor_{year}_Data.csv"
            if not path.exists() and prefix == "Distance":
                path = (
                    PROJECT_ROOT
                    / "relays_findings"
                    / "Distance_Relays_Findings"
                    / f"Relays_Distance_{gender}_Outdoor_{year}_Data.csv"
                )
            if path.exists():
                out.append((path, gender))
    return out


def load_results(year: str, season: str) -> list[ResultRow]:
    paths = iter_indoor_paths(year) if season == "indoor" else iter_outdoor_paths(year)
    seen_result: set[str] = set()
    out: list[ResultRow] = []
    for path, file_gender in paths:
        with open(path, newline="") as f:
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
                pcol = points_col(gender)
                try:
                    wa = float(row.get(pcol) or 0)
                except ValueError:
                    continue
                if wa <= 0:
                    continue
                rid = (row.get("result_id") or "").strip()
                # Deduplicate across overlapping discipline exports
                key = rid if rid else f"{aid}|{mid}|{eid}|{date}|{wa}"
                if key in seen_result:
                    continue
                seen_result.add(key)
                out.append(
                    ResultRow(
                        athlete_id=aid,
                        gender=gender,
                        year=year,
                        season=season,
                        meet_id=mid,
                        start_date=date,
                        event_id=eid,
                        event_name=EVENT_NAMES[eid],
                        wa=wa,
                        result_id=rid or key,
                    )
                )
    return out


def eligible_athletes(
    indoor: list[ResultRow], outdoor: list[ResultRow]
) -> dict[str, dict]:
    """athlete_id -> {gender, indoor_meets, outdoor_meets} for ≥2 meets each."""
    i_meets: dict[str, set[str]] = defaultdict(set)
    o_meets: dict[str, set[str]] = defaultdict(set)
    gender: dict[str, str] = {}
    for r in indoor:
        i_meets[r.athlete_id].add(r.meet_id)
        gender[r.athlete_id] = r.gender
    for r in outdoor:
        o_meets[r.athlete_id].add(r.meet_id)
        gender[r.athlete_id] = r.gender

    out = {}
    for aid in i_meets:
        if aid not in o_meets:
            continue
        if len(i_meets[aid]) >= 2 and len(o_meets[aid]) >= 2:
            out[aid] = {
                "gender": gender[aid],
                "n_indoor_meets": len(i_meets[aid]),
                "n_outdoor_meets": len(o_meets[aid]),
            }
    return out


def opener_meet_id(rows: list[ResultRow]) -> str | None:
    """Earliest meet by start_date; tie-break by meet_id."""
    if not rows:
        return None
    best = min(rows, key=lambda r: (r.start_date, r.meet_id))
    return best.meet_id


def best_wa_at_meet(rows: list[ResultRow], meet_id: str, event_id: int) -> float | None:
    vals = [r.wa for r in rows if r.meet_id == meet_id and r.event_id == event_id]
    return max(vals) if vals else None


def opener_date(rows: list[ResultRow], meet_id: str) -> str:
    dates = [r.start_date for r in rows if r.meet_id == meet_id]
    return min(dates) if dates else ""


@dataclass
class PairCompare:
    year: str
    athlete_id: str
    gender: str
    pair_label: str
    indoor_event: str
    outdoor_event: str
    indoor_wa: float
    outdoor_wa: float
    delta: float
    indoor_opener_date: str
    outdoor_opener_date: str
    n_indoor_meets: int
    n_outdoor_meets: int


def build_comparisons(year: str) -> tuple[list[PairCompare], dict]:
    indoor_all = load_results(year, "indoor")
    outdoor_all = load_results(year, "outdoor")
    eligible = eligible_athletes(indoor_all, outdoor_all)

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
        i_meet = opener_meet_id(i_rows)
        o_meet = opener_meet_id(o_rows)
        if not i_meet or not o_meet:
            continue
        i_date = opener_date(i_rows, i_meet)
        o_date = opener_date(o_rows, o_meet)
        gender = meta["gender"]

        for label, i_eid, o_eid, genders in SIMILAR_PAIRS:
            if genders is not None and gender not in genders:
                continue
            i_wa = best_wa_at_meet(i_rows, i_meet, i_eid)
            o_wa = best_wa_at_meet(o_rows, o_meet, o_eid)
            if i_wa is None or o_wa is None:
                continue
            comparisons.append(
                PairCompare(
                    year=year,
                    athlete_id=aid,
                    gender=gender,
                    pair_label=label,
                    indoor_event=EVENT_NAMES[i_eid],
                    outdoor_event=EVENT_NAMES[o_eid],
                    indoor_wa=i_wa,
                    outdoor_wa=o_wa,
                    delta=o_wa - i_wa,
                    indoor_opener_date=i_date,
                    outdoor_opener_date=o_date,
                    n_indoor_meets=meta["n_indoor_meets"],
                    n_outdoor_meets=meta["n_outdoor_meets"],
                )
            )

    meta = {
        "year": year,
        "n_eligible_athletes": len(eligible),
        "n_indoor_results": len(indoor_all),
        "n_outdoor_results": len(outdoor_all),
        "n_pair_comparisons": len(comparisons),
        "n_athletes_with_pair": len({c.athlete_id for c in comparisons}),
    }
    return comparisons, meta


def summarize(deltas: list[float]) -> dict:
    if not deltas:
        return {
            "n": 0,
            "mean_delta": None,
            "median_delta": None,
            "pct_outdoor_higher": None,
            "pct_indoor_higher": None,
            "pct_tie": None,
            "mean_indoor": None,
            "mean_outdoor": None,
        }
    return {
        "n": len(deltas),
        "mean_delta": round(statistics.mean(deltas), 2),
        "median_delta": round(statistics.median(deltas), 2),
    }


def group_stats(rows: list[PairCompare]) -> dict:
    if not rows:
        return {
            "n": 0,
            "n_athletes": 0,
            "mean_indoor_wa": None,
            "mean_outdoor_wa": None,
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
        "mean_indoor_wa": round(statistics.mean(r.indoor_wa for r in rows), 2),
        "mean_outdoor_wa": round(statistics.mean(r.outdoor_wa for r in rows), 2),
        "mean_delta": round(statistics.mean(deltas), 2),
        "median_delta": round(statistics.median(deltas), 2),
        "pct_outdoor_higher": round(100.0 * out_h / n, 1),
        "pct_indoor_higher": round(100.0 * in_h / n, 1),
        "pct_tie": round(100.0 * ties / n, 1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt_stat(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    return (
        f"n={s['n']} athletes={s['n_athletes']}  "
        f"indoorμ={s['mean_indoor_wa']:.1f} outdoorμ={s['mean_outdoor_wa']:.1f}  "
        f"Δμ={s['mean_delta']:+.1f} Δmed={s['median_delta']:+.1f}  "
        f"outdoor_higher={s['pct_outdoor_higher']:.1f}%  "
        f"indoor_higher={s['pct_indoor_higher']:.1f}%"
    )


def write_report(
    all_comp: list[PairCompare],
    year_meta: list[dict],
    path: Path,
    *,
    outdoor_band: tuple[int, int] | None = None,
) -> None:
    band_note = ""
    if outdoor_band is not None:
        lo, hi = outdoor_band
        all_comp = [c for c in all_comp if lo <= c.outdoor_wa < hi]
        band_note = f" (outdoor opener WA in [{lo}, {hi}))"
        year_meta = []
        for year in SEASONS:
            year_rows = [c for c in all_comp if c.year == year]
            year_meta.append(
                {
                    "year": year,
                    "n_eligible_athletes": len({c.athlete_id for c in year_rows}),
                    "n_pair_comparisons": len(year_rows),
                    "n_athletes_with_pair": len({c.athlete_id for c in year_rows}),
                }
            )

    overall = group_stats(all_comp)
    title = "Indoor vs Outdoor Season-Opener World Athletics Scores" + band_note
    lines = [
        title,
        "=" * len(title),
        "",
        "Question:",
        "  For athletes with ≥2 indoor meets and ≥2 outdoor meets in the same year,",
        "  do outdoor season openers start with higher WA scores than indoor season",
        "  openers for similar events?",
    ]
    if outdoor_band is not None:
        lo, hi = outdoor_band
        lines.extend(
            [
                f"  Restricted to comparisons where the outdoor opener result is in WA",
                f"  band [{lo}, {hi}).",
            ]
        )
    lines.extend(
        [
            "",
            "Method:",
            "  • Years: 2024, 2025, 2026 (indoor season file year paired with outdoor year).",
            "  • Eligibility: ≥2 distinct meet_ids with individual (non-relay) results indoors",
            "    AND ≥2 outdoors in that year.",
            "  • Season opener: earliest meet by start_date for that athlete in the season.",
            "  • Similar-event pairs compared only when the athlete contested the indoor",
            "    event at the indoor opener meet AND the paired outdoor event at the",
            "    outdoor opener meet.",
            "  • Delta = outdoor_opener_WA − indoor_opener_WA.",
            "  • Outdoor distance uses new_steeplechase_data (corrected WA) when available.",
            "  • Relays excluded.",
        ]
    )
    if outdoor_band is not None:
        lo, hi = outdoor_band
        lines.append(
            f"  • Band filter: keep a pair only if outdoor opener WA ∈ [{lo}, {hi})."
        )
    lines.append("")
    lines.append("Similar-event pairs:")
    for label, i_eid, o_eid, genders in SIMILAR_PAIRS:
        gnote = f" [{'/'.join(genders)} only]" if genders else ""
        lines.append(f"  • {label}{gnote}")
    lines.append("")

    lines.append("Eligibility / sample")
    lines.append("--------------------")
    for m in year_meta:
        lines.append(
            f"  {m['year']}: {m['n_eligible_athletes']} eligible athletes; "
            f"{m['n_pair_comparisons']} opener pair comparisons "
            f"({m['n_athletes_with_pair']} athletes with ≥1 comparable opener pair)"
        )
    lines.append(
        f"  Combined: {overall['n_athletes']} athletes contributing "
        f"{overall['n']} opener pair comparisons"
    )
    lines.append("")

    lines.append("Primary result — all years, all similar-event pairs")
    lines.append("---------------------------------------------------")
    lines.append(f"  {fmt_stat(overall)}")
    if overall["n"]:
        if overall["mean_delta"] > 0 and overall["pct_outdoor_higher"] > 50:
            verdict = (
                "YES — outdoor openers average higher WA than indoor openers "
                "for comparable events among dual-season athletes."
            )
        elif overall["mean_delta"] < 0 and overall["pct_indoor_higher"] > 50:
            verdict = (
                "NO — indoor openers average higher WA than outdoor openers "
                "for comparable events."
            )
        else:
            verdict = (
                "MIXED — mean delta and win-rate do not strongly favor one season."
            )
        lines.append(f"  Verdict: {verdict}")
    lines.append("")

    lines.append("By year")
    lines.append("-------")
    for year in SEASONS:
        s = group_stats([c for c in all_comp if c.year == year])
        lines.append(f"  {year}: {fmt_stat(s)}")
    lines.append("")

    lines.append("By gender")
    lines.append("---------")
    for gender in ("Men", "Women"):
        s = group_stats([c for c in all_comp if c.gender == gender])
        lines.append(f"  {gender}: {fmt_stat(s)}")
    lines.append("")

    lines.append("By similar-event pair")
    lines.append("---------------------")
    labels = []
    for label, *_ in SIMILAR_PAIRS:
        if label not in labels:
            labels.append(label)
    for label in labels:
        s = group_stats([c for c in all_comp if c.pair_label == label])
        if s["n"] == 0:
            continue
        lines.append(f"  {label}: {fmt_stat(s)}")
    lines.append("")

    lines.append("By gender × event pair (n≥10)")
    lines.append("-----------------------------")
    for gender in ("Men", "Women"):
        for label in labels:
            s = group_stats(
                [c for c in all_comp if c.gender == gender and c.pair_label == label]
            )
            if s["n"] < 10:
                continue
            lines.append(f"  {gender} {label}: {fmt_stat(s)}")
    lines.append("")

    # Athlete-level: among athletes with ≥1 pair, is mean outdoor opener WA higher?
    by_ath: dict[tuple[str, str], list[PairCompare]] = defaultdict(list)
    for c in all_comp:
        by_ath[(c.year, c.athlete_id)].append(c)
    ath_deltas = []
    for comps in by_ath.values():
        ath_deltas.append(statistics.mean(c.delta for c in comps))
    if ath_deltas:
        n = len(ath_deltas)
        out_h = sum(1 for d in ath_deltas if d > 0)
        in_h = sum(1 for d in ath_deltas if d < 0)
        lines.extend(
            [
                "Athlete-level (mean delta across an athlete-year's comparable opener pairs)",
                "--------------------------------------------------------------------------",
                f"  n_athlete_years={n}  mean Δ={statistics.mean(ath_deltas):+.2f}  "
                f"median Δ={statistics.median(ath_deltas):+.2f}",
                f"  outdoor_higher={100*out_h/n:.1f}%  indoor_higher={100*in_h/n:.1f}%  "
                f"tie={100*(n-out_h-in_h)/n:.1f}%",
                "",
            ]
        )

    lines.extend(
        [
            "Caveats",
            "-------",
            "  • Only opener-meet results count; later-season form is ignored.",
            "  • Athlete must run the paired events at BOTH openers — selection into",
            "    comparable pairs is not random (e.g., many milers may open outdoor in 800).",
            "  • Mile→1500m and 3000m→5000m are approximate cross-event WA comparisons.",
            "  • 60m→100m compares different distances on the WA scale (by design).",
            "  • Outdoor March 1+ filter is not applied beyond what the outdoor CSVs contain.",
        ]
    )
    if outdoor_band is not None:
        lo, hi = outdoor_band
        lines.append(
            f"  • This file restricts to outdoor opener WA ∈ [{lo}, {hi}); indoor opener"
        )
        lines.append(
            "    WA is unrestricted (may fall outside the band)."
        )
    lines.extend(
        [
            "",
            "Source: indoor_analysis/Indoor_Outdoor_Interplay/analyze_indoor_outdoor_opener_wa.py",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_comp: list[PairCompare] = []
    year_meta: list[dict] = []
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    for year in SEASONS:
        print(f"Building {year}...")
        comps, meta = build_comparisons(year)
        all_comp.extend(comps)
        year_meta.append(meta)
        print(
            f"  eligible={meta['n_eligible_athletes']} "
            f"pair_comps={meta['n_pair_comparisons']} "
            f"athletes_with_pair={meta['n_athletes_with_pair']}"
        )
        for c in comps:
            detail_rows.append(
                {
                    "year": c.year,
                    "athlete_id": c.athlete_id,
                    "gender": c.gender,
                    "pair_label": c.pair_label,
                    "indoor_event": c.indoor_event,
                    "outdoor_event": c.outdoor_event,
                    "indoor_opener_date": c.indoor_opener_date,
                    "outdoor_opener_date": c.outdoor_opener_date,
                    "indoor_wa": round(c.indoor_wa, 2),
                    "outdoor_wa": round(c.outdoor_wa, 2),
                    "delta_outdoor_minus_indoor": round(c.delta, 2),
                    "outdoor_higher": c.delta > 0,
                    "n_indoor_meets": c.n_indoor_meets,
                    "n_outdoor_meets": c.n_outdoor_meets,
                }
            )

    # Summary tables
    slices = [("all", all_comp)]
    for year in SEASONS:
        slices.append((f"year_{year}", [c for c in all_comp if c.year == year]))
    for gender in ("Men", "Women"):
        slices.append((f"gender_{gender}", [c for c in all_comp if c.gender == gender]))
    for label, *_ in SIMILAR_PAIRS:
        slices.append((f"pair_{label}", [c for c in all_comp if c.pair_label == label]))

    for name, rows in slices:
        s = group_stats(rows)
        if s["n"] == 0:
            continue
        summary_rows.append({"slice": name, **s})

    write_csv(OUTPUT_ROOT / "opener_wa_pair_comparisons.csv", detail_rows)
    write_csv(OUTPUT_ROOT / "opener_wa_summary_slices.csv", summary_rows)
    write_report(
        all_comp, year_meta, OUTPUT_ROOT / "indoor_outdoor_opener_wa_findings.txt"
    )

    # Band-restricted analysis: outdoor opener WA in [750, 950)
    band = (750, 950)
    band_comp = [c for c in all_comp if band[0] <= c.outdoor_wa < band[1]]
    band_detail = [
        {
            "year": c.year,
            "athlete_id": c.athlete_id,
            "gender": c.gender,
            "pair_label": c.pair_label,
            "indoor_event": c.indoor_event,
            "outdoor_event": c.outdoor_event,
            "indoor_opener_date": c.indoor_opener_date,
            "outdoor_opener_date": c.outdoor_opener_date,
            "indoor_wa": round(c.indoor_wa, 2),
            "outdoor_wa": round(c.outdoor_wa, 2),
            "delta_outdoor_minus_indoor": round(c.delta, 2),
            "outdoor_higher": c.delta > 0,
            "n_indoor_meets": c.n_indoor_meets,
            "n_outdoor_meets": c.n_outdoor_meets,
        }
        for c in band_comp
    ]
    band_summary = []
    band_slices = [("all", band_comp)]
    for year in SEASONS:
        band_slices.append((f"year_{year}", [c for c in band_comp if c.year == year]))
    for gender in ("Men", "Women"):
        band_slices.append(
            (f"gender_{gender}", [c for c in band_comp if c.gender == gender])
        )
    for label, *_ in SIMILAR_PAIRS:
        band_slices.append(
            (f"pair_{label}", [c for c in band_comp if c.pair_label == label])
        )
    for name, rows in band_slices:
        s = group_stats(rows)
        if s["n"] == 0:
            continue
        band_summary.append({"slice": name, **s})

    write_csv(
        OUTPUT_ROOT / "opener_wa_pair_comparisons_band_750_950.csv", band_detail
    )
    write_csv(
        OUTPUT_ROOT / "opener_wa_summary_slices_band_750_950.csv", band_summary
    )
    write_report(
        all_comp,
        year_meta,
        OUTPUT_ROOT / "indoor_outdoor_opener_wa_findings_band_750_950.txt",
        outdoor_band=band,
    )
    print(
        f"Band 750-950: {len(band_comp)} comparisons, "
        f"{len({c.athlete_id for c in band_comp})} athletes"
    )
    print(f"Wrote findings to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
