"""RQ1A best-event analysis for indoor track seasons 2024–2026.

Mirrors outdoor relays_findings RQ1A (overall + pairwise best event by WA points).
Does not answer RQ1B/RQ1C (no indoor Nationals).

Season window: all results in the indoor CSVs (typically Dec–March); no outdoor
March 1 filter.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDOOR_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = INDOOR_ROOT / "RQ1A_Best_Event"
PYLIBS = PROJECT_ROOT / "non_relays_findings" / "Sprints_Events_Counting" / ".pylibs"
if PYLIBS.exists():
    sys.path.insert(0, str(PYLIBS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

EVENT_COLORS = {
    "55m": "#1B9E77",
    "60m": "#2E86AB",
    "100m": "#2E86AB",
    "200m": "#A23B72",
    "300m": "#66A61E",
    "400m": "#F18F01",
    "500m": "#E6AB02",
    "600m": "#A6761D",
    "800m": "#1B9E77",
    "1000m": "#D95F02",
    "1500m": "#D95F02",
    "1600m": "#7570B3",
    "Mile": "#66A61E",
    "3000m": "#7570B3",
    "3200m": "#E7298A",
    "2 Mile": "#E6AB02",
    "5000m": "#666666",
    "55m Hurdles": "#E6AB02",
    "60m Hurdles": "#A6761D",
    "Long Jump": "#1F78B4",
    "Triple Jump": "#B2DF8A",
    "High Jump": "#FB9A99",
    "Shot Put": "#CAB2D6",
    "Discus": "#FDBF6F",
    "Javelin Throw": "#6A3D9A",
    "4x100m": "#28A745",
    "4x200m": "#6F42C1",
    "4x400m": "#C73E1D",
    "4x800m": "#17A2B8",
    "SMR": "#6C757D",
    "DMR": "#FFC107",
}


@dataclass(frozen=True)
class IndoorDiscipline:
    label: str
    data_folder: str  # e.g. Indoor_Sprints
    file_group: str  # e.g. Sprints (matches CSV name)
    individual_event_order: list[str]
    event_map_for_gender: Callable[[str], dict[int, str]]


def points_col(gender: str, metric: str = "wa") -> str:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents
    # walk up until scoring package is importable
    for p in Path(__file__).resolve().parents:
        if (p / "scoring" / "columns.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    from scoring.columns import points_col as _pc
    return _pc(gender, metric)


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


def merge_event_maps(*maps: dict[int, str]) -> dict[int, str]:
    merged: dict[int, str] = {}
    for event_map in maps:
        merged.update(event_map)
    return merged


def data_path(config: IndoorDiscipline, gender: str, year: str) -> Path:
    return (
        INDOOR_ROOT
        / config.data_folder
        / f"Indoor_Relays_{config.file_group}_{gender}_{year}_Data.csv"
    )


def event_order_for_gender(config: IndoorDiscipline, gender: str) -> list[str]:
    event_map = config.event_map_for_gender(gender)
    individual = individual_events_for_gender(config, gender)
    relays = [event_map[eid] for eid in sorted(event_map) if eid in STANDARD_RELAY_IDS]
    return individual + relays


def individual_events_for_gender(config: IndoorDiscipline, gender: str) -> list[str]:
    event_map = config.event_map_for_gender(gender)
    names = set(event_map.values())
    return [n for n in config.individual_event_order if n in names]


def load_athlete_event_bests(
    config: IndoorDiscipline, gender: str
) -> dict[str, dict[str, float]]:
    pcol = points_col(gender)
    event_map = config.event_map_for_gender(gender)
    athlete_event_best: dict[str, dict[str, float]] = defaultdict(dict)

    for year in SEASONS:
        path = data_path(config, gender, year)
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    event_id = int(float(row["running_event_id"]))
                except (TypeError, ValueError):
                    continue
                if event_id not in event_map:
                    continue
                event_name = event_map[event_id]
                points = float(row.get(pcol) or 0)
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


def best_event_for_athlete(
    event_points: dict[str, float], event_order: list[str]
) -> str | None:
    if not event_points:
        return None
    max_points = max(event_points.values())
    for event_name in event_order:
        if event_name in event_points and event_points[event_name] == max_points:
            return event_name
    return None


def filter_athletes_by_min_events(
    athlete_event_best: dict[str, dict[str, float]],
    min_events: int,
    allowed_events: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    filtered = {}
    for aid, event_points in athlete_event_best.items():
        events = set(event_points)
        if allowed_events is not None:
            events &= allowed_events
        if len(events) >= min_events:
            filtered[aid] = event_points
    return filtered


def compute_best_event_counts(
    athlete_event_best: dict[str, dict[str, float]],
    event_order: list[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    competed = {event_name: set() for event_name in event_order}
    best_counts = {event_name: 0 for event_name in event_order}
    for aid, event_points in athlete_event_best.items():
        for event_name in event_points:
            if event_name in competed:
                competed[event_name].add(aid)
        best_event = best_event_for_athlete(event_points, event_order)
        if best_event:
            best_counts[best_event] += 1
    return competed, best_counts


def compare_pairing(
    athlete_event_best: dict[str, dict[str, float]],
    ev_a: str,
    ev_b: str,
) -> tuple[int, int, int, int]:
    count_a = count_b = ties = athletes_in_both = 0
    for event_points in athlete_event_best.values():
        if ev_a not in event_points or ev_b not in event_points:
            continue
        athletes_in_both += 1
        pts_a = event_points[ev_a]
        pts_b = event_points[ev_b]
        if pts_a > pts_b:
            count_a += 1
        elif pts_b > pts_a:
            count_b += 1
        else:
            ties += 1
    return athletes_in_both, count_a, count_b, ties


def format_best_event_output(
    config: IndoorDiscipline,
    gender: str,
    competed: dict[str, set[str]],
    best_counts: dict[str, int],
    event_order: list[str],
    subtitle: str = "",
) -> str:
    label = "Male" if gender == "Men" else "Female"
    lines = [
        f"Indoor Track 2024-2026 {label} {config.label} (with Relays) — Best Event Count",
        "All indoor-season results in provided CSVs; best = highest World Athletics points per event.",
        "Relay events: team WA points credited to each leg athlete; athlete best = max across relay appearances.",
    ]
    if subtitle:
        lines.append(subtitle)
    lines.append("")
    for event_name in event_order:
        n_competed = len(competed.get(event_name, set()))
        if n_competed == 0:
            continue
        lines.append(f"{event_name} ({n_competed} competed): {best_counts.get(event_name, 0)}")
    return "\n".join(lines)


def save_best_event_histogram(
    best_counts: dict[str, int],
    competed: dict[str, set[str]],
    event_order: list[str],
    title: str,
    path: Path,
) -> None:
    events = [e for e in event_order if competed.get(e)]
    counts = [best_counts.get(e, 0) for e in events]
    if not events:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(events) * 0.9), 5))
    bars = ax.bar(events, counts, color="#2E86AB", edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Event")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
    plt.xticks(rotation=35, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def collect_pairwise_results(
    athlete_event_best: dict[str, dict[str, float]],
    pairings: list[tuple[str, str]],
) -> list[tuple[str, str, int, int, int, int]]:
    results = []
    for ev_a, ev_b in pairings:
        n, count_a, count_b, ties = compare_pairing(athlete_event_best, ev_a, ev_b)
        if n == 0:
            continue
        results.append((ev_a, ev_b, n, count_a, count_b, ties))
    return results


def save_pairwise_histogram(
    results: list[tuple[str, str, int, int, int, int]],
    title: str,
    path: Path,
    *,
    ncols: int = 3,
) -> None:
    if not results:
        return
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (ev_a, ev_b, n_athletes, count_a, count_b, _ties) in zip(axes_list, results):
        events = [ev_a, ev_b]
        counts = [count_a, count_b]
        colors = [EVENT_COLORS.get(ev, "#2E86AB") for ev in events]
        bars = ax.bar(events, counts, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(f"{ev_a} vs. {ev_b}\n({n_athletes} athletes)")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=11,
            )
    for ax in axes_list[len(results) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_pairwise_output(
    config: IndoorDiscipline,
    gender: str,
    athlete_event_best: dict[str, dict[str, float]],
    pairings: list[tuple[str, str]],
    path: Path,
    heading: str,
    hist_path: Path | None = None,
    hist_title: str = "",
) -> None:
    label = "Male" if gender == "Men" else "Female"
    lines = [
        f"Indoor Track 2024-2026 {label} {config.label} (with Relays) — Pairwise Best Event Comparison",
        heading,
        "",
    ]
    results = collect_pairwise_results(athlete_event_best, pairings)
    for ev_a, ev_b, n, count_a, count_b, ties in results:
        lines.append(f"{ev_a} vs. {ev_b} ({n} athletes in both):")
        lines.append(f"{ev_a} best event: {count_a}  vs. {ev_b} best event: {count_b}")
        if ties:
            lines.append(f"(ties: {ties})")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")
    if hist_path is not None and results:
        save_pairwise_histogram(results, hist_title, hist_path)


def build_relay_inclusive_pairings(
    athlete_event_best: dict[str, dict[str, float]],
    individual_events: list[str],
    relay_event_names: list[str],
    event_order: list[str],
) -> list[tuple[str, str]]:
    seen = set()
    relay_name_set = set(relay_event_names)
    for event_points in athlete_event_best.values():
        for event_name in event_points:
            if event_name in relay_name_set:
                seen.add(event_name)
    relay_events = [e for e in event_order if e in seen]
    pairings = list(combinations(individual_events, 2))
    for individual in individual_events:
        for relay in relay_events:
            pairings.append((individual, relay))
    for i, relay_a in enumerate(relay_events):
        for relay_b in relay_events[i + 1 :]:
            pairings.append((relay_a, relay_b))
    return pairings


def run_gender_analysis(config: IndoorDiscipline, gender: str, out_dir: Path) -> None:
    gender_slug = "men" if gender == "Men" else "women"
    event_order = event_order_for_gender(config, gender)
    individual_events = individual_events_for_gender(config, gender)
    event_map = config.event_map_for_gender(gender)
    relay_event_names = [
        event_map[eid] for eid in sorted(event_map) if eid in STANDARD_RELAY_IDS
    ]
    athlete_event_best = load_athlete_event_bests(config, gender)

    if len(individual_events) >= 2:
        ind_subtitle = (
            f"Athletes with results in at least 2 of {', '.join(individual_events)}"
        )
    elif individual_events:
        ind_subtitle = f"Athletes with results in {individual_events[0]}"
    else:
        ind_subtitle = "No individual events configured"

    variants = [
        ("", 1, None, "All athletes with at least one event result"),
        (
            "_2plus_events",
            2,
            None,
            "Athletes with results in at least 2 events (individual + relay)",
        ),
    ]
    if len(individual_events) >= 2:
        variants.append(
            ("_2plus_individual", 2, set(individual_events), ind_subtitle)
        )

    for suffix, min_events, allowed_events, subtitle in variants:
        subset = athlete_event_best
        if min_events > 1:
            subset = filter_athletes_by_min_events(subset, min_events, allowed_events)
        competed, best_counts = compute_best_event_counts(subset, event_order)
        text = format_best_event_output(
            config, gender, competed, best_counts, event_order, subtitle
        )
        text_path = out_dir / f"best_event_counts_{gender_slug}_2024_2026{suffix}.txt"
        text_path.write_text(text + "\n")
        hist_path = out_dir / f"best_event_histogram_{gender_slug}_2024_2026{suffix}.png"
        save_best_event_histogram(
            best_counts,
            competed,
            event_order,
            f"Indoor Track 2024-2026 {gender} {config.label} (with Relays)\n{subtitle}",
            hist_path,
        )
        print(text)
        print(f"Saved {text_path.relative_to(INDOOR_ROOT)} and {hist_path.name}\n")

    if len(individual_events) >= 2:
        write_pairwise_output(
            config,
            gender,
            athlete_event_best,
            list(combinations(individual_events, 2)),
            out_dir / f"pairwise_best_event_counts_{gender_slug}_2024_2026_individual.txt",
            f"Individual events only ({', '.join(individual_events)}).",
            hist_path=out_dir
            / f"pairwise_best_event_histogram_{gender_slug}_2024_2026_individual.png",
            hist_title=(
                f"Pairwise Best Event — Indoor Track 2024-2026 {gender} {config.label}\n"
                "Individual events only"
            ),
        )
    write_pairwise_output(
        config,
        gender,
        athlete_event_best,
        build_relay_inclusive_pairings(
            athlete_event_best, individual_events, relay_event_names, event_order
        ),
        out_dir / f"pairwise_best_event_counts_{gender_slug}_2024_2026.txt",
        "All individual pairings, every individual×relay combination, and relay×relay pairings.",
        hist_path=out_dir / f"pairwise_best_event_histogram_{gender_slug}_2024_2026.png",
        hist_title=(
            f"Pairwise Best Event — Indoor Track 2024-2026 {gender} {config.label}\n"
            "Individual + relay combinations"
        ),
    )


SPRINT_INDIVIDUAL = {2: "60m", 4: "200m", 6: "400m"}
DISTANCE_INDIVIDUAL = {9: "800m", 13: "Mile", 14: "3000m", 17: "5000m"}
HURDLES_INDIVIDUAL = {32: "55m Hurdles", 33: "60m Hurdles"}
JUMPS_INDIVIDUAL = {38: "Long Jump", 39: "Triple Jump", 40: "High Jump"}
THROWS_INDIVIDUAL = {41: "Shot Put", 42: "Discus", 45: "Javelin Throw"}

DISCIPLINES: list[IndoorDiscipline] = [
    IndoorDiscipline(
        label="Sprints",
        data_folder="Indoor_Sprints",
        file_group="Sprints",
        individual_event_order=["60m", "200m", "400m"],
        event_map_for_gender=lambda _g: merge_event_maps(
            SPRINT_INDIVIDUAL, STANDARD_RELAY_MAP
        ),
    ),
    IndoorDiscipline(
        label="Distance",
        data_folder="Indoor_Distance",
        file_group="Distance",
        individual_event_order=["800m", "Mile", "3000m", "5000m"],
        event_map_for_gender=lambda _g: merge_event_maps(
            DISTANCE_INDIVIDUAL, STANDARD_RELAY_MAP
        ),
    ),
    IndoorDiscipline(
        label="Hurdles",
        data_folder="Indoor_Hurdles",
        file_group="Hurdles",
        individual_event_order=["55m Hurdles", "60m Hurdles"],
        event_map_for_gender=lambda _g: merge_event_maps(
            HURDLES_INDIVIDUAL, STANDARD_RELAY_MAP
        ),
    ),
    IndoorDiscipline(
        label="Jumps",
        data_folder="Indoor_Jumps",
        file_group="Jumps",
        individual_event_order=["Long Jump", "Triple Jump", "High Jump"],
        event_map_for_gender=lambda _g: merge_event_maps(
            JUMPS_INDIVIDUAL, STANDARD_RELAY_MAP
        ),
    ),
    IndoorDiscipline(
        label="Throws",
        data_folder="Indoor_Throws",
        file_group="Throws",
        individual_event_order=["Shot Put", "Discus", "Javelin Throw"],
        event_map_for_gender=lambda _g: merge_event_maps(
            THROWS_INDIVIDUAL, STANDARD_RELAY_MAP
        ),
    ),
]


def write_summary(path: Path) -> None:
    lines = [
        "Indoor Track RQ1A — Best Event Summary (2024–2026)",
        "=================================================",
        "",
        "Question: For athletes competing in multiple events within a discipline,",
        "which event corresponds to their highest World Athletics point value?",
        "",
        "Method: overall best event + pairwise dual-event comparisons (same as outdoor RQ1A).",
        "Data: indoor_analysis CSVs; all dates in files (no outdoor March 1 filter).",
        "Relays: team WA points credited to each leg athlete.",
        "Not run: RQ1B / RQ1C (no indoor Nationals).",
        "",
        "Core individual events analyzed:",
        "  Sprints: 60m, 200m, 400m",
        "  Distance: 800m, Mile, 3000m, 5000m",
        "  Hurdles: 55m Hurdles, 60m Hurdles",
        "  Jumps: Long Jump, Triple Jump, High Jump",
        "  Throws: Shot Put, Discus, Javelin Throw",
        "",
        "Per-discipline outputs are in RQ1A_Best_Event/<Discipline>/.",
        "Prefer *_2plus_individual.txt / pairwise *_individual.txt for specialization",
        "among multi-event athletes (closest to the RQ1A framing).",
        "",
    ]
    # Quick headline stats from 2plus individual files
    for config in DISCIPLINES:
        out_dir = OUTPUT_ROOT / config.label
        lines.append(f"{config.label}")
        lines.append("-" * len(config.label))
        for gender, slug in (("Men", "men"), ("Women", "women")):
            path_2 = out_dir / f"best_event_counts_{slug}_2024_2026_2plus_individual.txt"
            path_1 = out_dir / f"best_event_counts_{slug}_2024_2026.txt"
            src = path_2 if path_2.exists() else path_1
            if not src.exists():
                lines.append(f"  {gender}: (no output)")
                continue
            body = src.read_text().strip().splitlines()
            # last event lines
            event_lines = [ln for ln in body if "(" in ln and "competed):" in ln]
            lines.append(f"  {gender} ({src.name}):")
            for ln in event_lines:
                lines.append(f"    {ln}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for config in DISCIPLINES:
        out_dir = OUTPUT_ROOT / config.label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {config.label} ===")
        for gender in ("Men", "Women"):
            run_gender_analysis(config, gender, out_dir)
    summary = OUTPUT_ROOT / "rq1a_indoor_summary.txt"
    write_summary(summary)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
