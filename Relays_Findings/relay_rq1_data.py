"""Shared data loading for relay-inclusive RQ1B / RQ1C analyses."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring.columns import points_col as scoring_points_col  # noqa: E402

NON_RELAYS_ROOT = PROJECT_ROOT / "non_relays_findings"
EVENT_MAP = {
    int(r["running_event_id"]): r["event_name"]
    for r in csv.DictReader(open(NON_RELAYS_ROOT / "Distance_Events_Counting" / "running_event.csv"))
}

SEASONS = ["2024", "2025", "2026"]
STANDARD_RELAY_MAP = {
    21: "4x100m",
    22: "4x200m",
    24: "4x400m",
    26: "4x800m",
    29: "SMR",
    30: "DMR",
    31: "Swedish Relay",
}
STANDARD_RELAY_IDS = set(STANDARD_RELAY_MAP)
PRELIM_EVENT_IDS = {3, 4}
FIELD_EVENT_IDS = {38, 39, 40, 41, 42, 43, 44, 45, 46}

RELAY_SOURCE = ("Sprinters_Relays_Findings", "Sprinters")

DISCIPLINE_INDIVIDUAL = [
    ("Sprints", "Sprinters_Relays_Findings", "Sprinters", {3, 4, 6}, ["100m", "200m", "400m"]),
    ("Distance", "Distance_Relays_Findings", "Distance", {9, 11, 17, 20}, ["800m", "1500m", "3000m Steeplechase", "5000m"]),
    ("Jumps", "Jumps_Relays_Findings", "Jumps", {38, 39, 40}, ["Long Jump", "Triple Jump", "High Jump"]),
    ("Throws", "Throws_Relays_Findings", "Throws", {41, 42, 43, 45}, ["Shot Put", "Discus", "Hammer Throw", "Javelin Throw"]),
]

HURDLES_MEN = {35: "110m Hurdles", 37: "400m Hurdles"}
HURDLES_WOMEN = {34: "100m Hurdles", 37: "400m Hurdles"}


def points_col(gender: str, metric: str = "wa") -> str:
    """Column for World Athletics (default) or vdot / purdy / mercier."""
    return scoring_points_col(gender, metric)


def is_in_season(date_str: str) -> bool:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d >= datetime(d.year, 3, 1)


def parse_athlete_id(value: str) -> str | None:
    raw = (value or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    return str(int(float(raw)))


def leg_athlete_ids(row: dict) -> list[str]:
    ids = []
    for col in ("athlete_id", "athlete_id_2", "athlete_id_3", "athlete_id_4"):
        aid = parse_athlete_id(row.get(col, ""))
        if aid:
            ids.append(aid)
    return ids


def data_path(folder: str, prefix: str, gender: str, year: str) -> Path:
    return ROOT / folder / f"Relays_{prefix}_{gender}_Outdoor_{year}_Data.csv"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def season_rows(folder: str, prefix: str, gender: str, year: str) -> list[dict]:
    return load_csv(data_path(folder, prefix, gender, year))


def hurdles_event_ids(gender: str) -> set[int]:
    return set(HURDLES_MEN if gender == "Men" else HURDLES_WOMEN)


def hurdles_event_map(gender: str) -> dict[int, str]:
    return dict(HURDLES_MEN if gender == "Men" else HURDLES_WOMEN)


def all_event_sources(gender: str) -> list[tuple[str, str, str, set[int]]]:
    """Discipline label, folder, prefix, event ids for this gender."""
    sources = [(d, f, p, ids) for d, f, p, ids, _ in DISCIPLINE_INDIVIDUAL]
    hmap = hurdles_event_map(gender)
    sources.append(("Hurdles", "Hurdles_Relays_Findings", "Hurdles", set(hmap)))
    sources.append(("Relays", *RELAY_SOURCE, set(STANDARD_RELAY_IDS)))
    return sources


def discipline_event_map(discipline: str, gender: str) -> dict[int, str]:
    if discipline == "Hurdles":
        base = dict(hurdles_event_map(gender))
    else:
        base = {}
        for disc, _, _, ids, _ in DISCIPLINE_INDIVIDUAL:
            if disc == discipline:
                base = {eid: EVENT_MAP[eid] for eid in ids}
                break
    base.update(STANDARD_RELAY_MAP)
    return base


def discipline_event_order(discipline: str, gender: str) -> list[str]:
    if discipline == "Hurdles":
        individual = list(hurdles_event_map(gender).values())
    else:
        individual = []
        for disc, _, _, _, order in DISCIPLINE_INDIVIDUAL:
            if disc == discipline:
                individual = order
                break
    return individual + [STANDARD_RELAY_MAP[eid] for eid in sorted(STANDARD_RELAY_IDS)]


def load_athlete_event_bests_for_discipline(
    discipline: str,
    gender: str,
) -> dict[str, dict[str, float]]:
    """Personal bests for one discipline folder (individual + relay leg expansion)."""
    if discipline == "Hurdles":
        folder, prefix = "Hurdles_Relays_Findings", "Hurdles"
        allowed = set(hurdles_event_map(gender)) | STANDARD_RELAY_IDS
    else:
        folder = prefix = None
        allowed = None
        for disc, f, p, ids, _ in DISCIPLINE_INDIVIDUAL:
            if disc == discipline:
                folder, prefix = f, p
                allowed = set(ids) | STANDARD_RELAY_IDS
                break
        if folder is None:
            return {}

    pcol = points_col(gender)
    athlete_event_best: dict[str, dict[str, float]] = defaultdict(dict)

    for year in SEASONS:
        for row in season_rows(folder, prefix, gender, year):
            if not is_in_season(row["start_date"]):
                continue
            event_id = int(row["running_event_id"])
            if event_id not in allowed:
                continue
            event_name = EVENT_MAP[event_id]
            points = float(row[pcol])
            if points <= 0:
                continue

            if event_id in STANDARD_RELAY_IDS:
                athlete_ids = leg_athlete_ids(row)
            else:
                aid = parse_athlete_id(row.get("athlete_id", ""))
                athlete_ids = [aid] if aid else []

            for aid in athlete_ids:
                current = athlete_event_best[aid].get(event_name)
                if current is None or points > current:
                    athlete_event_best[aid][event_name] = points

    return dict(athlete_event_best)


def parse_performance(result_time: str, event_id: int) -> float:
    raw = (result_time or "").strip().lower().replace("m", "")
    if not raw:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")
    try:
        if ":" in raw:
            mins, secs = raw.split(":", 1)
            return float(mins) * 60 + float(secs)
        return float(raw)
    except ValueError:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")
