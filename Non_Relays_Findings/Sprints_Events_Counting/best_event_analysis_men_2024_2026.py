"""Determine best sprint event per athlete by max World Athletics Men's Points (2024-2026, on/after March 1 each year)."""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".pylibs"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "Sprinters_Men_Outdoor_2024_2026_Data.csv"
OUTPUT_DIR = Path(__file__).parent
EVENT_ORDER = ["100m", "200m", "400m"]
EVENT_MAP = {3: "100m", 4: "200m", 6: "400m"}


def is_in_season(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d >= datetime(d.year, 3, 1)


def load_filtered_results():
    rows = []
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            if not is_in_season(r["start_date"]):
                continue
            eid = int(r["running_event_id"])
            if eid not in EVENT_MAP:
                continue
            rows.append(
                {
                    "athlete_id": r["athlete_id"],
                    "event": EVENT_MAP[eid],
                    "points": float(r["World_Athletics_Points_Men"]),
                }
            )
    return rows


def filter_by_min_events(rows, min_events):
    athlete_events = defaultdict(set)
    for r in rows:
        athlete_events[r["athlete_id"]].add(r["event"])
    eligible = {aid for aid, evs in athlete_events.items() if len(evs) >= min_events}
    return [r for r in rows if r["athlete_id"] in eligible]


def compute_best_event_counts(rows):
    competed = {ev: set() for ev in EVENT_ORDER}
    for r in rows:
        competed[r["event"]].add(r["athlete_id"])

    athlete_best = {}
    for r in rows:
        aid = r["athlete_id"]
        if aid not in athlete_best or r["points"] > athlete_best[aid]["points"]:
            athlete_best[aid] = r

    best_counts = {ev: 0 for ev in EVENT_ORDER}
    for best in athlete_best.values():
        best_counts[best["event"]] += 1

    return competed, best_counts


def format_output(competed, best_counts, subtitle=""):
    lines = ["Outdoor Track 2024-2026 Male Sprinters Best Event Count:"]
    if subtitle:
        lines.append(subtitle)
    lines.append("")
    for ev in EVENT_ORDER:
        lines.append(f"{ev} ({len(competed[ev])} competed): {best_counts[ev]}")
    return "\n".join(lines)


def save_histogram(best_counts, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [best_counts[ev] for ev in EVENT_ORDER]
    bars = ax.bar(EVENT_ORDER, counts, color=["#2E86AB", "#A23B72", "#F18F01"], edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Event")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(count), ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_analysis(min_events, text_suffix, hist_suffix, subtitle):
    rows = load_filtered_results()
    if min_events > 1:
        rows = filter_by_min_events(rows, min_events)

    competed, best_counts = compute_best_event_counts(rows)
    text_output = format_output(competed, best_counts, subtitle)
    print(text_output)

    text_path = OUTPUT_DIR / f"best_event_counts_men_2024_2026{text_suffix}.txt"
    text_path.write_text(text_output + "\n")

    title = (
        "Outdoor Track 2024-2026 Male Sprinters — Best Event Count\n"
        f"(results on or after March 1 each year; {subtitle.strip('()')})"
        if subtitle
        else "Outdoor Track 2024-2026 Male Sprinters — Best Event Count\n(results on or after March 1 each year)"
    )
    hist_path = OUTPUT_DIR / f"best_event_histogram_men_2024_2026{hist_suffix}.png"
    save_histogram(best_counts, title, hist_path)

    print(f"\nSaved text output to {text_path}")
    print(f"Saved histogram to {hist_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-events",
        type=int,
        default=1,
        help="Minimum number of distinct sprint events an athlete must have competed in",
    )
    args = parser.parse_args()

    if args.min_events >= 2:
        run_analysis(
            min_events=args.min_events,
            text_suffix="_2plus_events",
            hist_suffix="_2plus_events",
            subtitle="(athletes with results in at least 2 of 100m, 200m, 400m)",
        )
    else:
        run_analysis(
            min_events=1,
            text_suffix="",
            hist_suffix="",
            subtitle="",
        )


if __name__ == "__main__":
    main()
