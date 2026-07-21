"""Best-event and pairwise analysis for collegiate outdoor jumps (2024–2026)."""

import csv
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

for lib in (
    Path(__file__).parent / ".pylibs",
    Path(__file__).parent.parent / "Sprints_Events_Counting" / ".pylibs",
    Path(__file__).parent.parent / "Distance_Events_Counting" / ".pylibs",
):
    if lib.exists():
        sys.path.insert(0, str(lib))
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent

EVENT_MAP = {38: "Long Jump", 39: "Triple Jump", 40: "High Jump"}
EVENT_ORDER = ["Long Jump", "Triple Jump", "High Jump"]


def season_cutoff(year: int) -> datetime:
    return datetime(year, 3, 1)


def is_in_season(date_str: str, year: int | None = None) -> bool:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    if year is not None:
        return d.year == year and d >= season_cutoff(year)
    return d >= season_cutoff(d.year)


def load_results(files: list[Path], points_col: str, year: int | None):
    rows = []
    for path in files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not is_in_season(r["start_date"], year):
                    continue
                eid = int(r["running_event_id"])
                if eid not in EVENT_MAP:
                    continue
                rows.append(
                    {
                        "athlete_id": r["athlete_id"],
                        "event": EVENT_MAP[eid],
                        "points": float(r[points_col]),
                    }
                )
    return rows


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

    return competed, best_counts, athlete_best


def format_best_event_output(gender_label, year_label, competed, best_counts):
    lines = [f"Outdoor Track {year_label} {gender_label} Jumps Best Event Count:"]
    lines.append("")
    for ev in EVENT_ORDER:
        lines.append(f"{ev} ({len(competed[ev])} competed): {best_counts[ev]}")
    return "\n".join(lines)


def save_best_histogram(best_counts, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [best_counts[ev] for ev in EVENT_ORDER]
    colors = ["#2E86AB", "#F18F01", "#A23B72"]
    bars = ax.bar(EVENT_ORDER, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Event")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_ylim(0, max(counts) * 1.15 if counts else 1)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def athlete_event_bests(rows):
    bests = defaultdict(dict)
    for r in rows:
        aid = r["athlete_id"]
        ev = r["event"]
        if ev not in bests[aid] or r["points"] > bests[aid][ev]:
            bests[aid][ev] = r["points"]
    return bests


def pairwise_comparison(bests, ev_a, ev_b):
    athletes = [aid for aid, m in bests.items() if ev_a in m and ev_b in m]
    count_a = count_b = ties = 0
    for aid in athletes:
        pts_a = bests[aid][ev_a]
        pts_b = bests[aid][ev_b]
        if pts_a > pts_b:
            count_a += 1
        elif pts_b > pts_a:
            count_b += 1
        else:
            ties += 1
    return len(athletes), count_a, count_b, ties


def format_pairwise_output(gender_label, year_label, points_label, results):
    lines = [
        f"Outdoor Track {year_label} {gender_label} Jumps — Pairwise Best Event Comparison",
        f"(best = highest World Athletics {points_label} per event)",
        "",
    ]
    for ev_a, ev_b, n, count_a, count_b, ties in results:
        lines.append(f"{ev_a} vs. {ev_b} ({n} athletes ran in both):")
        lines.append(f"{ev_a} best event: {count_a}  vs. {ev_b} best event: {count_b}")
        if ties:
            lines.append(f"(ties: {ties})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_pairwise_histogram(results, title, path):
    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    palette = {
        "Long Jump": "#2E86AB",
        "Triple Jump": "#F18F01",
        "High Jump": "#A23B72",
    }
    for ax, (ev_a, ev_b, n, count_a, count_b, _) in zip(axes, results):
        events = [ev_a, ev_b]
        counts = [count_a, count_b]
        bars = ax.bar(events, counts, color=[palette[ev_a], palette[ev_b]], edgecolor="black", linewidth=0.8)
        ax.set_title(f"{ev_a} vs. {ev_b}\n({n} athletes)")
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
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_gender_year(gender: str, year: int | None, suffix: str):
    is_men = gender == "men"
    points_col = "World_Athletics_Points_Men" if is_men else "World_Athletics_Points_Women"
    points_label = "Men's Points" if is_men else "Women's Points"
    gender_label = "Male" if is_men else "Female"

    if year is None:
        files = sorted(OUTPUT_DIR.glob(f"Jumps_{'Men' if is_men else 'Women'}_Outdoor_*_Data.csv"))
        year_label = "2024–2026"
        season_note = "results on or after March 1 of each season"
    else:
        files = [OUTPUT_DIR / f"Jumps_{'Men' if is_men else 'Women'}_Outdoor_{year}_Data.csv"]
        year_label = str(year)
        season_note = f"results on or after {year}-03-01"

    rows = load_results(files, points_col, year)
    competed, best_counts, _ = compute_best_event_counts(rows)

    best_text = format_best_event_output(gender_label, year_label, competed, best_counts)
    best_path = OUTPUT_DIR / f"best_event_counts_{gender}{suffix}.txt"
    best_path.write_text(best_text + "\n")

    best_title = f"Outdoor Track {year_label} {gender_label} Jumps — Best Event Count\n({season_note})"
    best_hist = OUTPUT_DIR / f"best_event_histogram_{gender}{suffix}.png"
    save_best_histogram(best_counts, best_title, best_hist)

    bests = athlete_event_bests(rows)
    pairwise_results = []
    for ev_a, ev_b in combinations(EVENT_ORDER, 2):
        n, count_a, count_b, ties = pairwise_comparison(bests, ev_a, ev_b)
        pairwise_results.append((ev_a, ev_b, n, count_a, count_b, ties))

    pairwise_text = format_pairwise_output(gender_label, year_label, points_label, pairwise_results)
    pairwise_path = OUTPUT_DIR / f"pairwise_best_event_counts_{gender}{suffix}.txt"
    pairwise_path.write_text(pairwise_text)

    pairwise_title = (
        f"Pairwise Best Event Comparison — Outdoor Track {year_label} {gender_label} Jumps\n"
        f"({season_note})"
    )
    pairwise_hist = OUTPUT_DIR / f"pairwise_best_event_histogram_{gender}{suffix}.png"
    save_pairwise_histogram(pairwise_results, pairwise_title, pairwise_hist)

    print(f"\n=== {gender.upper()} {year_label} ===")
    print(best_text)
    print()
    print(pairwise_text)
    print(f"Saved: {best_path.name}, {best_hist.name}, {pairwise_path.name}, {pairwise_hist.name}")


def main():
    configs = [
        (2024, "_2024"),
        (2025, "_2025"),
        (2026, "_2026"),
        (None, "_2024_2026"),
        (2026, ""),
    ]
    for gender in ("men", "women"):
        seen = set()
        for year, suffix in configs:
            key = (gender, year)
            if key in seen:
                continue
            seen.add(key)
            run_gender_year(gender, year, suffix)


if __name__ == "__main__":
    main()
