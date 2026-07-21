"""RQ1A best-event analysis for sprinters data including relays (2024–2026 outdoor)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PYLIBS = Path(__file__).resolve().parents[2] / "Non_Relays_Findings" / "Sprints_Events_Counting" / ".pylibs"
if PYLIBS.exists():
    sys.path.insert(0, str(PYLIBS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
SEASONS = ["2024", "2025", "2026"]

EVENT_MAP = {
    3: "100m",
    4: "200m",
    6: "400m",
    21: "4x100m",
    22: "4x200m",
    24: "4x400m",
    26: "4x800m",
    29: "SMR",
    30: "DMR",
    31: "Swedish Relay",
}
RELAY_IDS = {21, 22, 24, 26, 29, 30, 31}
EVENT_ORDER = list(EVENT_MAP.values())
INDIVIDUAL_EVENTS = ["100m", "200m", "400m"]
INDIVIDUAL_PAIRINGS = [
    ("100m", "200m"),
    ("100m", "400m"),
    ("200m", "400m"),
]
RELAY_EVENT_NAMES = [EVENT_MAP[eid] for eid in sorted(RELAY_IDS)]


def relay_events_in_data(athlete_event_best: dict[str, dict[str, float]]) -> list[str]:
    """Relay events with at least one athlete having a valid best score."""
    seen = set()
    for event_points in athlete_event_best.values():
        for event_name in event_points:
            if event_name in RELAY_EVENT_NAMES:
                seen.add(event_name)
    return [event_name for event_name in EVENT_ORDER if event_name in seen]


def build_relay_inclusive_pairings(athlete_event_best: dict[str, dict[str, float]]) -> list[tuple[str, str]]:
    """Every individual×relay combination, plus individual and relay head-to-heads."""
    relay_events = relay_events_in_data(athlete_event_best)
    pairings = list(INDIVIDUAL_PAIRINGS)

    for individual in INDIVIDUAL_EVENTS:
        for relay in relay_events:
            pairings.append((individual, relay))

    for i, relay_a in enumerate(relay_events):
        for relay_b in relay_events[i + 1 :]:
            pairings.append((relay_a, relay_b))

    return pairings

EVENT_COLORS = {
    "100m": "#2E86AB",
    "200m": "#A23B72",
    "400m": "#F18F01",
    "4x100m": "#28A745",
    "4x200m": "#6F42C1",
    "4x400m": "#C73E1D",
    "4x800m": "#17A2B8",
    "SMR": "#6C757D",
    "DMR": "#FFC107",
    "Swedish Relay": "#6610F2",
}


def points_col(gender: str) -> str:
    return "World_Athletics_Points_Men" if gender == "Men" else "World_Athletics_Points_Women"


def data_path(gender: str, year: str) -> Path:
    return ROOT / f"Relays_Sprinters_{gender}_Outdoor_{year}_Data.csv"


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


def load_athlete_event_bests(gender: str) -> dict[str, dict[str, float]]:
    pcol = points_col(gender)
    athlete_event_best: dict[str, dict[str, float]] = defaultdict(dict)

    for year in SEASONS:
        path = data_path(gender, year)
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not is_in_season(row["start_date"]):
                    continue
                event_id = int(row["running_event_id"])
                if event_id not in EVENT_MAP:
                    continue

                event_name = EVENT_MAP[event_id]
                points = float(row[pcol])
                if points <= 0:
                    continue

                if event_id in RELAY_IDS:
                    athlete_ids = leg_athlete_ids(row)
                else:
                    aid = parse_athlete_id(row.get("athlete_id", ""))
                    athlete_ids = [aid] if aid else []

                for aid in athlete_ids:
                    current = athlete_event_best[aid].get(event_name)
                    if current is None or points > current:
                        athlete_event_best[aid][event_name] = points

    return dict(athlete_event_best)


def best_event_for_athlete(event_points: dict[str, float]) -> str | None:
    if not event_points:
        return None
    max_points = max(event_points.values())
    for event_name in EVENT_ORDER:
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


def compute_best_event_counts(athlete_event_best: dict[str, dict[str, float]]) -> tuple[dict[str, set[str]], dict[str, int]]:
    competed = {event_name: set() for event_name in EVENT_ORDER}
    best_counts = {event_name: 0 for event_name in EVENT_ORDER}

    for aid, event_points in athlete_event_best.items():
        for event_name in event_points:
            competed[event_name].add(aid)

        best_event = best_event_for_athlete(event_points)
        if best_event:
            best_counts[best_event] += 1

    return competed, best_counts


def compare_pairing(
    athlete_event_best: dict[str, dict[str, float]],
    ev_a: str,
    ev_b: str,
) -> tuple[int, int, int, int]:
    count_a = count_b = ties = 0
    athletes_in_both = 0
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
    gender: str,
    competed: dict[str, set[str]],
    best_counts: dict[str, int],
    subtitle: str = "",
) -> str:
    label = "Male" if gender == "Men" else "Female"
    lines = [
        f"Outdoor Track 2024-2026 {label} Sprinters (with Relays) — Best Event Count",
        "Results on or after March 1 each year; best = highest World Athletics points per event.",
        "Relay events: team WA points credited to each leg athlete; athlete best = max across relay appearances.",
    ]
    if subtitle:
        lines.append(subtitle)
    lines.append("")
    for event_name in EVENT_ORDER:
        n_competed = len(competed[event_name])
        if n_competed == 0:
            continue
        lines.append(f"{event_name} ({n_competed} competed): {best_counts[event_name]}")
    return "\n".join(lines)


def save_best_event_histogram(
    best_counts: dict[str, int],
    competed: dict[str, set[str]],
    title: str,
    path: Path,
) -> None:
    events = [event_name for event_name in EVENT_ORDER if competed[event_name]]
    counts = [best_counts[event_name] for event_name in events]
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
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=11,
            )

    for ax in axes_list[len(results):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_pairwise_output(
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
        f"Outdoor Track 2024-2026 {label} Sprinters (with Relays) — Pairwise Best Event Comparison",
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


def run_gender_analysis(gender: str) -> None:
    gender_slug = "men" if gender == "Men" else "women"
    athlete_event_best = load_athlete_event_bests(gender)

    variants = [
        ("", 1, None, "All athletes with at least one event result"),
        ("_2plus_events", 2, None, "Athletes with results in at least 2 events (individual + relay)"),
        (
            "_2plus_individual",
            2,
            set(INDIVIDUAL_EVENTS),
            "Athletes with results in at least 2 of 100m, 200m, 400m",
        ),
    ]

    for suffix, min_events, allowed_events, subtitle in variants:
        subset = athlete_event_best
        if min_events > 1:
            subset = filter_athletes_by_min_events(subset, min_events, allowed_events)

        competed, best_counts = compute_best_event_counts(subset)
        text = format_best_event_output(gender, competed, best_counts, subtitle)
        text_path = ROOT / f"best_event_counts_{gender_slug}_2024_2026{suffix}.txt"
        text_path.write_text(text + "\n")

        hist_path = ROOT / f"best_event_histogram_{gender_slug}_2024_2026{suffix}.png"
        save_best_event_histogram(
            best_counts,
            competed,
            f"Outdoor Track 2024-2026 {gender} Sprinters (with Relays)\n{subtitle}",
            hist_path,
        )
        print(text)
        print(f"Saved {text_path.name} and {hist_path.name}\n")

    write_pairwise_output(
        gender,
        athlete_event_best,
        INDIVIDUAL_PAIRINGS,
        ROOT / f"pairwise_best_event_counts_{gender_slug}_2024_2026_individual.txt",
        "Individual events only (100m, 200m, 400m); same pairings as non-relay analysis.",
        hist_path=ROOT / f"pairwise_best_event_histogram_{gender_slug}_2024_2026_individual.png",
        hist_title=(
            f"Pairwise Best Event Comparison — Outdoor Track 2024-2026 {gender} Sprinters (with Relays)\n"
            "Individual events only; results on or after March 1 each year"
        ),
    )
    write_pairwise_output(
        gender,
        athlete_event_best,
        build_relay_inclusive_pairings(athlete_event_best),
        ROOT / f"pairwise_best_event_counts_{gender_slug}_2024_2026.txt",
        "All individual pairings, every individual×relay combination, and relay×relay pairings.",
        hist_path=ROOT / f"pairwise_best_event_histogram_{gender_slug}_2024_2026.png",
        hist_title=(
            f"Pairwise Best Event Comparison — Outdoor Track 2024-2026 {gender} Sprinters (with Relays)\n"
            "All individual×relay combinations included; results on or after March 1 each year"
        ),
    )
    print(
        f"Saved pairwise outputs for {gender_slug}: "
        f"pairwise_best_event_histogram_{gender_slug}_2024_2026_individual.png, "
        f"pairwise_best_event_histogram_{gender_slug}_2024_2026.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gender",
        choices=("Men", "Women", "Both"),
        default="Both",
        help="Which gender(s) to analyze",
    )
    args = parser.parse_args()

    genders = ("Men", "Women") if args.gender == "Both" else (args.gender,)
    for gender in genders:
        run_gender_analysis(gender)


if __name__ == "__main__":
    main()
