"""Pairwise best-event comparison for women: top result in event A vs top result in event B (2024)."""

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".pylibs"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "Sprinters_Women_Outdoor_2024_Data.csv"
OUTPUT_DIR = Path(__file__).parent
CUTOFF = datetime(2024, 3, 1)
EVENT_MAP = {3: "100m", 4: "200m", 6: "400m"}

PAIRINGS = [
    ("100m", "200m"),
    ("100m", "400m"),
    ("200m", "400m"),
]


def load_athlete_event_bests():
    athlete_event_best = defaultdict(dict)
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            if datetime.strptime(r["start_date"], "%Y-%m-%d") < CUTOFF:
                continue
            eid = int(r["running_event_id"])
            if eid not in EVENT_MAP:
                continue
            aid = r["athlete_id"]
            ev = EVENT_MAP[eid]
            pts = float(r["World_Athletics_Points_Women"])
            if ev not in athlete_event_best[aid] or pts > athlete_event_best[aid][ev]:
                athlete_event_best[aid][ev] = pts
    return athlete_event_best


def compare_pairing(athlete_event_best, ev_a, ev_b):
    athletes = [aid for aid, bests in athlete_event_best.items() if ev_a in bests and ev_b in bests]
    count_a = count_b = ties = 0
    for aid in athletes:
        pts_a = athlete_event_best[aid][ev_a]
        pts_b = athlete_event_best[aid][ev_b]
        if pts_a > pts_b:
            count_a += 1
        elif pts_b > pts_a:
            count_b += 1
        else:
            ties += 1
    return len(athletes), count_a, count_b, ties


def main():
    athlete_event_best = load_athlete_event_bests()
    lines = [
        "Outdoor Track 2024 Female Sprinters — Pairwise Best Event Comparison",
        "(results on or after 2024-03-01; best = highest World Athletics Women's Points per event)",
        "",
    ]
    results = []

    for ev_a, ev_b in PAIRINGS:
        n, count_a, count_b, ties = compare_pairing(athlete_event_best, ev_a, ev_b)
        lines.append(f"{ev_a} vs. {ev_b} ({n} athletes ran in both):")
        lines.append(f"{ev_a} best event: {count_a}  vs. {ev_b} best event: {count_b}")
        if ties:
            lines.append(f"(ties: {ties})")
        lines.append("")
        results.append((ev_a, ev_b, n, count_a, count_b))

    text_output = "\n".join(lines).rstrip() + "\n"
    print(text_output)

    text_path = OUTPUT_DIR / "pairwise_best_event_counts_women_2024.txt"
    text_path.write_text(text_output)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {"100m": "#2E86AB", "200m": "#A23B72", "400m": "#F18F01"}

    for ax, (ev_a, ev_b, n, count_a, count_b) in zip(axes, results):
        events = [ev_a, ev_b]
        counts = [count_a, count_b]
        bars = ax.bar(events, counts, color=[colors[ev_a], colors[ev_b]], edgecolor="black", linewidth=0.8)
        ax.set_title(f"{ev_a} vs. {ev_b}\n({n} athletes)")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count), ha="center", va="bottom", fontsize=11)

    fig.suptitle(
        "Pairwise Best Event Comparison — Outdoor Track 2024 Female Sprinters\n"
        "(top result per event compared; results on or after 2024-03-01)",
        fontsize=12,
    )
    plt.tight_layout()
    hist_path = OUTPUT_DIR / "pairwise_best_event_histogram_women_2024.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print(f"Saved text output to {text_path}")
    print(f"Saved histogram to {hist_path}")


if __name__ == "__main__":
    main()
