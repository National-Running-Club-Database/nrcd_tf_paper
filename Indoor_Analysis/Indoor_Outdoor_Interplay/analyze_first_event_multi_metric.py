"""First-event recommendations in outdoor-opener WA band [750, 950) — multi-metric.

Replicates the analysis in first_event_recommendations_band_750_950.txt for
VDOT, Gardner–Purdy, and Mercier. World Athletics artifacts are copied from the
parent Indoor_Outdoor_Interplay folder (not regenerated).

Inclusion (all metrics): same athlete-years as the WA report — outdoor first-event
WA ∈ [750, 950), dual indoor+outdoor season. First-event pick for ranking under
metric M = highest M score on the chronologically earliest meet date.

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

INDOOR_ROOT = PROJECT_ROOT / "indoor_analysis"
PARENT = INDOOR_ROOT / "Indoor_Outdoor_Interplay"
OUT_DIR = PARENT / "multiple_metrics"

SEASONS = ("2024", "2025", "2026")
RELAY_IDS = {21, 22, 23, 24, 25, 26, 29, 30, 31}
BAND_LO, BAND_HI = 750.0, 950.0
MIN_N_RECOMMEND = 5
MIN_N_OVERALL = 10

# Canonical event-id → name (fallback if CSV lacks running_event text)
EVENT_NAMES: dict[int, str] = {
    1: "55m",
    2: "60m",
    3: "100m",
    4: "200m",
    5: "300m",
    6: "400m",
    7: "500m",
    8: "600m",
    9: "800m",
    10: "1000m",
    11: "1500m",
    13: "Mile",
    14: "3000m",
    15: "2 Mile",
    17: "5000m",
    20: "3000m Steeplechase",
    32: "55m Hurdles",
    33: "60m Hurdles",
    34: "100m Hurdles",
    35: "110m Hurdles",
    37: "400m Hurdles",
    38: "Long Jump",
    39: "Triple Jump",
    40: "High Jump",
    41: "Shot Put",
    42: "Discus",
    43: "Hammer Throw",
    45: "Javelin Throw",
}

NAME_TO_GROUP: dict[str, str] = {
    "55m": "Sprints",
    "60m": "Sprints",
    "100m": "Sprints",
    "200m": "Sprints",
    "300m": "Sprints",
    "400m": "Sprints",
    "500m": "Sprints",
    "600m": "Sprints",
    "800m": "Distance",
    "1000m": "Distance",
    "1500m": "Distance",
    "Mile": "Distance",
    "3000m": "Distance",
    "2 Mile": "Distance",
    "5000m": "Distance",
    "3000m Steeplechase": "Distance",
    "55m Hurdles": "Hurdles",
    "60m Hurdles": "Hurdles",
    "100m Hurdles": "Hurdles",
    "110m Hurdles": "Hurdles",
    "400m Hurdles": "Hurdles",
    "Long Jump": "Jumps",
    "Triple Jump": "Jumps",
    "High Jump": "Jumps",
    "Shot Put": "Throws",
    "Discus": "Throws",
    "Hammer Throw": "Throws",
    "Javelin Throw": "Throws",
}

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
    event_group: str
    scores: dict[str, float]  # metric -> points
    result_id: str


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


def normalize_event_name(raw: str, eid: int) -> str:
    name = (raw or "").strip()
    if name:
        # light cleanup of common CSV variants
        aliases = {
            "3000m steeplechase": "3000m Steeplechase",
            "3000 steeplechase": "3000m Steeplechase",
            "long jump": "Long Jump",
            "triple jump": "Triple Jump",
            "high jump": "High Jump",
            "shot put": "Shot Put",
            "60m hurdles": "60m Hurdles",
            "55m hurdles": "55m Hurdles",
            "100m hurdles": "100m Hurdles",
            "110m hurdles": "110m Hurdles",
            "400m hurdles": "400m Hurdles",
            "2 mile": "2 Mile",
            "2-mile": "2 Mile",
        }
        key = name.lower()
        if key in aliases:
            return aliases[key]
        return name
    return EVENT_NAMES.get(eid, f"event_{eid}")


def event_group_for(name: str, folder_group: str) -> str:
    if name in NAME_TO_GROUP:
        return NAME_TO_GROUP[name]
    # folder group may be Sprinters → Sprints
    if folder_group == "Sprinters":
        return "Sprints"
    return folder_group


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


def iter_indoor_paths(year: str) -> list[tuple[Path, str, str]]:
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
                out.append((path, gender, group))
    return out


def iter_outdoor_paths(year: str) -> list[tuple[Path, str, str]]:
    sources = [
        (PROJECT_ROOT / "relays_findings" / "Sprinters_Relays_Findings", "Sprinters", "Sprints"),
        (PROJECT_ROOT / "relays_findings" / "Hurdles_Relays_Findings", "Hurdles", "Hurdles"),
        (PROJECT_ROOT / "relays_findings" / "Jumps_Relays_Findings", "Jumps", "Jumps"),
        (PROJECT_ROOT / "relays_findings" / "Throws_Relays_Findings", "Throws", "Throws"),
        (
            PROJECT_ROOT / "new_steeplechase_data" / "Distance_Relays_Findings",
            "Distance",
            "Distance",
        ),
        (PROJECT_ROOT / "new_steeplechase_data", "Distance", "Distance"),
    ]
    out = []
    seen_paths: set[Path] = set()
    for folder, prefix, group in sources:
        for gender in ("Men", "Women"):
            path = folder / f"Relays_{prefix}_{gender}_Outdoor_{year}_Data.csv"
            if path.exists() and path.resolve() not in seen_paths:
                seen_paths.add(path.resolve())
                out.append((path, gender, group))
    return out


def load_results(year: str, season: str) -> list[ResultRow]:
    paths = iter_indoor_paths(year) if season == "indoor" else iter_outdoor_paths(year)
    seen: set[str] = set()
    out: list[ResultRow] = []
    for path, file_gender, folder_group in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                eid = parse_event_id(row)
                if eid is None or eid in RELAY_IDS:
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
                ename = normalize_event_name(
                    row.get("running_event") or row.get("event") or "", eid
                )
                eg = event_group_for(ename, folder_group)
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
                        event_group=eg,
                        scores=scores,
                        result_id=rid or key,
                    )
                )
    return out


def first_event_on_opener_day(
    rows: list[ResultRow], metric: str
) -> ResultRow | None:
    """Earliest date; among that day's results with metric score, take max score."""
    scored = [r for r in rows if metric in r.scores]
    if metric in {"purdy", "vdot"}:
        scored = [r for r in scored if not is_field_event(r.event_name)]
    if not scored:
        return None
    first_date = min(r.start_date for r in scored)
    day = [r for r in scored if r.start_date == first_date]
    return max(day, key=lambda r: r.scores[metric])


def build_eligible_wa_sample() -> list[dict]:
    """Athlete-years with outdoor first-event WA in [750, 950)."""
    sample: list[dict] = []
    for year in SEASONS:
        indoor = load_results(year, "indoor")
        outdoor = load_results(year, "outdoor")
        by_i: dict[str, list[ResultRow]] = defaultdict(list)
        by_o: dict[str, list[ResultRow]] = defaultdict(list)
        for r in indoor:
            by_i[r.athlete_id].append(r)
        for r in outdoor:
            by_o[r.athlete_id].append(r)
        for aid in set(by_i) & set(by_o):
            i_first = first_event_on_opener_day(by_i[aid], "wa")
            o_first = first_event_on_opener_day(by_o[aid], "wa")
            if not i_first or not o_first:
                continue
            o_wa = o_first.scores["wa"]
            if not (BAND_LO <= o_wa < BAND_HI):
                continue
            sample.append(
                {
                    "year": year,
                    "athlete_id": aid,
                    "gender": o_first.gender,
                    "indoor_rows": by_i[aid],
                    "outdoor_rows": by_o[aid],
                }
            )
    return sample


def summarize_side(
    records: list[tuple[str, str, str, float]],
) -> dict[tuple[str, str], list[tuple[str, int, float, float]]]:
    """gender, group -> list of (event, n, mean, median) sorted by mean desc."""
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for gender, group, event, score in records:
        buckets[(gender, group, event)].append(score)

    by_cell: dict[tuple[str, str], list[tuple[str, int, float, float]]] = defaultdict(
        list
    )
    for (gender, group, event), vals in buckets.items():
        by_cell[(gender, group)].append(
            (
                event,
                len(vals),
                statistics.mean(vals),
                statistics.median(vals),
            )
        )
    for key in by_cell:
        by_cell[key].sort(key=lambda t: (-t[2], t[0]))
    return by_cell


def overall_top(
    records: list[tuple[str, str, str, float]], min_n: int
) -> dict[str, list[tuple[str, int, float, float]]]:
    by_ge: dict[tuple[str, str], list[float]] = defaultdict(list)
    for gender, _group, event, score in records:
        by_ge[(gender, event)].append(score)
    out: dict[str, list[tuple[str, int, float, float]]] = {"Men": [], "Women": []}
    for (gender, event), vals in by_ge.items():
        if len(vals) < min_n:
            continue
        out[gender].append(
            (event, len(vals), statistics.mean(vals), statistics.median(vals))
        )
    for g in out:
        out[g].sort(key=lambda t: (-t[2], t[0]))
    return out


def best_in_cell(
    ranked: list[tuple[str, int, float, float]], min_n: int
) -> tuple[str, float, int] | None:
    eligible = [t for t in ranked if t[1] >= min_n]
    pool = eligible if eligible else ranked
    if not pool:
        return None
    ev, n, mean, _med = pool[0]
    return ev, mean, n


def run_metric(metric: str, sample: list[dict]) -> None:
    detail_rows: list[dict] = []
    indoor_recs: list[tuple[str, str, str, float]] = []
    outdoor_recs: list[tuple[str, str, str, float]] = []

    for item in sample:
        i_first = first_event_on_opener_day(item["indoor_rows"], metric)
        o_first = first_event_on_opener_day(item["outdoor_rows"], metric)
        if not i_first or not o_first:
            continue
        i_sc = i_first.scores[metric]
        o_sc = o_first.scores[metric]
        detail_rows.append(
            {
                "year": item["year"],
                "athlete_id": item["athlete_id"],
                "gender": item["gender"],
                "metric": metric,
                "indoor_first_event": i_first.event_name,
                "indoor_first_group": i_first.event_group,
                "indoor_first_score": round(i_sc, 4),
                "indoor_first_date": i_first.start_date,
                "outdoor_first_event": o_first.event_name,
                "outdoor_first_group": o_first.event_group,
                "outdoor_first_score": round(o_sc, 4),
                "outdoor_first_date": o_first.start_date,
                "outdoor_first_wa": round(o_first.scores.get("wa", float("nan")), 4),
                "delta_out_minus_in": round(o_sc - i_sc, 4),
            }
        )
        indoor_recs.append(
            (item["gender"], i_first.event_group, i_first.event_name, i_sc)
        )
        outdoor_recs.append(
            (item["gender"], o_first.event_group, o_first.event_name, o_sc)
        )

    detail_path = OUT_DIR / f"first_event_band_750_950_detail_{metric}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "year",
            "athlete_id",
            "gender",
            "metric",
            "indoor_first_event",
            "indoor_first_group",
            "indoor_first_score",
            "indoor_first_date",
            "outdoor_first_event",
            "outdoor_first_group",
            "outdoor_first_score",
            "outdoor_first_date",
            "outdoor_first_wa",
            "delta_out_minus_in",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(detail_rows)

    report_path = OUT_DIR / f"first_event_recommendations_band_750_950_{metric}.txt"
    _write_report_fixed(metric, detail_rows, indoor_recs, outdoor_recs, report_path)
    print(f"Wrote {report_path.relative_to(PROJECT_ROOT)} (n={len(detail_rows)})")
    print(f"Wrote {detail_path.relative_to(PROJECT_ROOT)}")


def _write_report_fixed(
    metric: str,
    detail_rows: list[dict],
    indoor_recs: list[tuple[str, str, str, float]],
    outdoor_recs: list[tuple[str, str, str, float]],
    path: Path,
) -> None:
    label = METRIC_LABELS[metric]
    framing = metric_framing(metric)
    unit = "WA" if metric == "wa" else label

    i_scores = [r["indoor_first_score"] for r in detail_rows]
    o_scores = [r["outdoor_first_score"] for r in detail_rows]
    deltas = [r["delta_out_minus_in"] for r in detail_rows]

    lines: list[str] = [
        f"Best First Event of Season — Athletes with Outdoor Opener in WA {int(BAND_LO)}–{int(BAND_HI)}",
        f"Metric: {label} [{framing}]",
        "=" * 72,
        "",
        "Question:",
        f"  For athletes whose first outdoor-season result has WA in [{int(BAND_LO)}, {int(BAND_HI)}),",
        f"  which events tended to produce the highest {label} as the first indoor event,",
        "  and which as the first outdoor event — by gender and event group?",
        "",
        "Method:",
        "  • Years 2024–2026; athlete-year must have ≥1 indoor and ≥1 outdoor",
        "    individual (non-relay) result.",
        "  • First result of a season = results on the chronologically earliest meet",
        f"    date; if multiple events that day, the highest-{label} result that day",
        "    is used as the noted first-event performance.",
        f"  • Inclusion: outdoor first-event WA ∈ [{int(BAND_LO)}, {int(BAND_HI)})",
        "    (same cohort definition as the World Athletics report).",
        f"  • Within each gender × event-group cell, rank events by mean first-event {label}.",
        f"  • “Best” recommendation within a cell uses events with n≥{MIN_N_RECOMMEND} when available.",
        f"  • Sample: {len(detail_rows)} athlete-years.",
        "",
        f"Mean indoor first {unit}: {statistics.mean(i_scores):.1f}",
        f"Mean outdoor first {unit}: {statistics.mean(o_scores):.1f}",
        f"Mean Δ (out−in): {statistics.mean(deltas):+.1f}",
        "",
    ]

    i_top = overall_top(indoor_recs, MIN_N_OVERALL)
    o_top = overall_top(outdoor_recs, MIN_N_OVERALL)
    lines.append(
        f"Overall first-indoor events with highest mean {unit} (min n={MIN_N_OVERALL})"
    )
    for gender in ("Men", "Women"):
        lines.append(f"  {gender}:")
        for ev, n, mean, med in i_top[gender][:5]:
            lines.append(f"    {ev}: mean={mean:.1f} med={med:.1f} n={n}")
    lines.append("")
    lines.append(
        f"Overall first-outdoor events with highest mean {unit} (min n={MIN_N_OVERALL})"
    )
    for gender in ("Men", "Women"):
        lines.append(f"  {gender}:")
        for ev, n, mean, med in o_top[gender][:5]:
            lines.append(f"    {ev}: mean={mean:.1f} med={med:.1f} n={n}")
    lines.append("")

    headlines_i: list[str] = []
    headlines_o: list[str] = []

    def emit(title: str, recs: list, cells: dict, headlines: list[str]) -> None:
        nonlocal lines
        lines += [
            f"First {title} event — mean {unit} by gender × event group",
            "-" * 60,
            "",
        ]
        for gender in ("Men", "Women"):
            for group in ("Sprints", "Distance", "Hurdles", "Jumps", "Throws"):
                ranked = cells.get((gender, group), [])
                if not ranked:
                    continue
                cell_scores = [
                    s for g, grp, _e, s in recs if g == gender and grp == group
                ]
                n_ay = len(cell_scores)
                cell_mean = statistics.mean(cell_scores)
                lines.append(
                    f"{gender} — {group}  (n={n_ay} athlete-years, mean {unit}={cell_mean:.1f})"
                )
                lines.append(f"  {'Event':<22} {'n':>4}  {'mean':>8}   {'median':>7}")
                for ev, n, mean, med in ranked:
                    lines.append(f"  {ev:<22} {n:4d}  {mean:8.1f}   {med:7.1f}")
                best = best_in_cell(ranked, MIN_N_RECOMMEND)
                if best:
                    ev, mean, n = best
                    lines.append(
                        f"  → Highest mean {unit} (n≥{MIN_N_RECOMMEND}): "
                        f"{ev} ({mean:.1f}, n={n})"
                    )
                    headlines.append(
                        f"  {title.title()} | {gender} {group}: {ev} "
                        f"(mean {unit} {mean:.1f}, n={n})"
                    )
                lines.append("")

    emit("INDOOR", indoor_recs, summarize_side(indoor_recs), headlines_i)
    emit("OUTDOOR", outdoor_recs, summarize_side(outdoor_recs), headlines_o)

    lines += [
        f"Headline recommendations (n≥{MIN_N_RECOMMEND} within gender × group)",
        "-" * 52,
        "Indoor season openers:",
        *headlines_i,
        "Outdoor season openers:",
        *headlines_o,
        "",
        "Caveats:",
        f"  • Conditioning on outdoor opener already in {int(BAND_LO)}–{int(BAND_HI)} truncates",
        "    outdoor WA from below; outdoor event rankings are within that cohort.",
        "  • Indoor first-event scores are unrestricted and often lower.",
        f"  • Same-day multi-event openers: highest-{label} mark that day is used.",
        "  • Observational — does not prove causality of event choice.",
        "  • Purdy/VDOT undefined for field events — those marks are skipped for",
        "    first-event selection under those metrics.",
        "",
        f"Source detail CSV: first_event_band_750_950_detail_{metric}.csv",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_wa_artifacts() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_report = PARENT / "first_event_recommendations_band_750_950.txt"
    src_detail = PARENT / "first_event_band_750_950_detail.csv"
    dst_report = OUT_DIR / "first_event_recommendations_band_750_950_wa.txt"
    dst_detail = OUT_DIR / "first_event_band_750_950_detail_wa.csv"
    if not src_report.exists() or not src_detail.exists():
        raise SystemExit(f"Missing WA source artifacts under {PARENT}")

    original = src_report.read_text(encoding="utf-8")
    if original.startswith("Best First Event"):
        lines = original.splitlines()
        # After title + underline, insert metric note
        out_lines = [lines[0], "Metric: World Athletics Points [sports]", lines[1]]
        out_lines.append(
            "(Copied from parent Indoor_Outdoor_Interplay/; not regenerated.)"
        )
        out_lines.extend(lines[2:])
        text = "\n".join(out_lines) + "\n"
        text = text.replace(
            "Source detail CSV: first_event_band_750_950_detail.csv",
            "Source detail CSV: first_event_band_750_950_detail_wa.csv",
        )
    else:
        text = original
    dst_report.write_text(text, encoding="utf-8")

    with open(src_detail, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_rows = []
    for r in rows:
        out_rows.append(
            {
                "year": r["year"],
                "athlete_id": r["athlete_id"],
                "gender": r["gender"],
                "metric": "wa",
                "indoor_first_event": r["indoor_first_event"],
                "indoor_first_group": r["indoor_first_group"],
                "indoor_first_score": r["indoor_first_wa"],
                "indoor_first_date": r["indoor_first_date"],
                "outdoor_first_event": r["outdoor_first_event"],
                "outdoor_first_group": r["outdoor_first_group"],
                "outdoor_first_score": r["outdoor_first_wa"],
                "outdoor_first_date": r["outdoor_first_date"],
                "outdoor_first_wa": r["outdoor_first_wa"],
                "delta_out_minus_in": r["delta_out_minus_in"],
            }
        )
    with open(dst_detail, "w", newline="", encoding="utf-8") as f:
        fields = list(out_rows[0].keys()) if out_rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Copied WA report → {dst_report.relative_to(PROJECT_ROOT)}")
    print(f"Copied WA detail → {dst_detail.relative_to(PROJECT_ROOT)}")


def write_index() -> None:
    lines = [
        "First-event recommendations — multiple metrics",
        "==============================================",
        "",
        "Band: outdoor first-event World Athletics Points ∈ [750, 950).",
        "Same dual indoor+outdoor athlete-year cohort for all metrics.",
        "",
        "Files:",
        "  first_event_recommendations_band_750_950_wa.txt       (copied)",
        "  first_event_recommendations_band_750_950_vdot.txt",
        "  first_event_recommendations_band_750_950_purdy.txt",
        "  first_event_recommendations_band_750_950_mercier.txt",
        "  first_event_band_750_950_detail_{wa,vdot,purdy,mercier}.csv",
        "",
        "Regenerate non-WA metrics:",
        "  python indoor_analysis/Indoor_Outdoor_Interplay/analyze_first_event_multi_metric.py",
        "",
    ]
    (OUT_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_wa_artifacts()
    print("Building eligible WA-band sample for VDOT / Purdy / Mercier...")
    sample = build_eligible_wa_sample()
    print(f"Eligible athlete-years: {len(sample)}")
    for metric in ("vdot", "purdy", "mercier"):
        print(f"Running {metric}...")
        run_metric(metric, sample)
    write_index()
    print(f"Done. Outputs in {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
