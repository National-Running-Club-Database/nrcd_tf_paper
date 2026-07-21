"""Count balanced vs specialized athletes within WA point bands.

Bands: 750–950, 800–1000, 850–1050 (width 200).
Inclusion: ≥1 individual result WA in the band (same as Point_Bands_Time_Models).

Balanced (wa_spread < 50):
  • sprints_balanced / distance_balanced by event group

Specialized (wa_spread ≥ 50):
  • short_sprints / long_sprints / mid_distance / long_distance
    (same Short/Long definitions as Short_Long_Specialization_Time_Models;
     requires season PBs in all 3 group events)
  • specialized_unclassified if fewer than 3 events or a tie

Unit: athlete-season. Relays and steeple excluded. March 1+.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIME_MODELS_ROOT = Path(__file__).resolve().parents[2]
BAND_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
MODEL_SEARCH_ROOT = TIME_MODELS_ROOT / "model_search"
SPEC_ROOT = TIME_MODELS_ROOT / "specialized_time_models"
SHORT_LONG_ROOT = TIME_MODELS_ROOT / "Short_Long_Specialization_Time_Models"
FEAT_ROOT = BAND_ROOT / "Feature_Importance_Point_Band_Time_Models"

sys.path.insert(0, str(PROJECT_ROOT / "Relays_Findings"))
sys.path.insert(0, str(TIME_MODELS_ROOT))
sys.path.insert(0, str(MODEL_SEARCH_ROOT))
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(SHORT_LONG_ROOT))
sys.path.insert(0, str(BAND_ROOT))
sys.path.insert(0, str(FEAT_ROOT))

from build_cross_event_time_models import GROUP_CONFIG  # noqa: E402
from analyze_specialization_time_models import specialization_label  # noqa: E402
from analyze_short_long_specialization import classify_short_long  # noqa: E402
from analyze_point_bands_time_models import in_band  # noqa: E402
from analyze_feature_importance_point_bands import (  # noqa: E402
    SPREAD_THRESHOLD,
    TARGET_BANDS,
    BandProfile,
    load_profiles,
)

# classify_short_long expects an object with .event_wa — BandProfile has that.


BALANCED_KEYS = ("sprints_balanced", "distance_balanced")
SPECIALIZED_KEYS = (
    "short_sprints",
    "long_sprints",
    "mid_distance",
    "long_distance",
    "specialized_unclassified",
)


def classify_athlete(profile: BandProfile, event_group: str) -> str:
    bal_spec = specialization_label(profile.wa_spread, SPREAD_THRESHOLD)
    if bal_spec == "balanced":
        return "sprints_balanced" if event_group == "Sprints" else "distance_balanced"

    # specialized → Short/Long axis
    short_long = classify_short_long(profile, event_group)
    if short_long in (
        "short_sprints",
        "long_sprints",
        "mid_distance",
        "long_distance",
    ):
        return short_long
    return "specialized_unclassified"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    profiles_by_gg: dict[tuple[str, str], dict[str, BandProfile]] = {}
    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            profiles = load_profiles(folder, prefix, gender, events)
            for p in profiles.values():
                p.event_group = event_group
            profiles_by_gg[(gender, event_group)] = profiles
            print(f"Loaded {gender} {event_group}: {len(profiles)}")

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    band_totals: dict[str, Counter] = {}

    for lo, hi in TARGET_BANDS:
        band = f"{lo}-{hi}"
        counts: Counter = Counter()
        by_gg: dict[tuple[str, str], Counter] = defaultdict(Counter)
        n_as = 0

        for (gender, event_group), profiles in profiles_by_gg.items():
            for p in profiles.values():
                if not in_band(p.result_was, lo, hi):
                    continue
                n_as += 1
                label = classify_athlete(p, event_group)
                counts[label] += 1
                by_gg[(gender, event_group)][label] += 1
                detail_rows.append(
                    {
                        "band": band,
                        "band_lo": lo,
                        "band_hi": hi,
                        "gender": gender,
                        "event_group": event_group,
                        "athlete_season_key": p.key,
                        "wa_spread": round(p.wa_spread, 1),
                        "bal_spec": specialization_label(p.wa_spread, SPREAD_THRESHOLD),
                        "classification": label,
                        "events_competed": p.events_competed,
                        "best_event": p.best_event,
                        "max_wa": round(p.max_wa, 1),
                    }
                )

        band_totals[band] = counts
        for gender in ("Men", "Women"):
            for event_group in ("Sprints", "Distance"):
                c = by_gg[(gender, event_group)]
                n = sum(c.values())
                row = {
                    "band": band,
                    "gender": gender,
                    "event_group": event_group,
                    "n_athlete_seasons": n,
                    "balanced_total": (
                        c["sprints_balanced"] + c["distance_balanced"]
                    ),
                    "specialized_total": n
                    - (c["sprints_balanced"] + c["distance_balanced"]),
                }
                for k in BALANCED_KEYS + SPECIALIZED_KEYS:
                    row[k] = c[k]
                summary_rows.append(row)

        # band-level aggregate row
        n = sum(counts.values())
        summary_rows.append(
            {
                "band": band,
                "gender": "All",
                "event_group": "All",
                "n_athlete_seasons": n,
                "balanced_total": counts["sprints_balanced"]
                + counts["distance_balanced"],
                "specialized_total": n
                - (
                    counts["sprints_balanced"] + counts["distance_balanced"]
                ),
                **{k: counts[k] for k in BALANCED_KEYS + SPECIALIZED_KEYS},
            }
        )
        print(f"Band {band}: {n_as} athlete-seasons")

    write_csv(OUTPUT_ROOT / "balanced_specialization_counts_summary.csv", summary_rows)
    write_csv(OUTPUT_ROOT / "athlete_season_classifications.csv", detail_rows)

    # Human-readable report
    lines = [
        "Balanced / Specialization Counts by WA Point Band",
        "=================================================",
        "",
        "Question: Within each high-WA band, how many athlete-seasons are",
        "balanced vs specialized — and for specialized athletes, how many",
        "are Short/Long (or Mid/Long) specialists?",
        "",
        "Definitions:",
        f"  • Band inclusion: ≥1 individual (non-relay, non-steeple) result with",
        "    WA points in [lo, hi). Sprints & Distance only. March 1+ outdoor.",
        f"  • Balanced: wa_spread < {SPREAD_THRESHOLD:.0f} across season PBs in the",
        "    event group → sprints_balanced or distance_balanced.",
        f"  • Specialized: wa_spread ≥ {SPREAD_THRESHOLD:.0f}, then Short/Long label:",
        "      Sprints — Short: |WA(100)−WA(200)| < |WA(200)−WA(400)|",
        "               Long:  |WA(400)−WA(200)| < |WA(100)−WA(200)|",
        "      Distance — Mid:  |WA(800)−WA(1500)| < |WA(1500)−WA(5000)|",
        "                 Long: |WA(1500)−WA(5000)| < |WA(1500)−WA(800)|",
        "    (Requires all 3 group-event PBs; else specialized_unclassified.)",
        "  • Unit: athlete-season (aid|year). An athlete competing in both",
        "    Sprints and Distance CSVs can contribute one row to each group.",
        "",
    ]

    for lo, hi in TARGET_BANDS:
        band = f"{lo}-{hi}"
        c = band_totals[band]
        n = sum(c.values())
        bal = c["sprints_balanced"] + c["distance_balanced"]
        spec = n - bal
        lines.extend(
            [
                f"=== Band {band} ===",
                f"Total athlete-seasons in band: {n}",
                f"  Balanced:    {bal}  ({100 * bal / n:.1f}%)" if n else "  Balanced: 0",
                f"    sprints_balanced:  {c['sprints_balanced']}",
                f"    distance_balanced: {c['distance_balanced']}",
                f"  Specialized: {spec}  ({100 * spec / n:.1f}%)" if n else "  Specialized: 0",
                f"    short_sprints:              {c['short_sprints']}",
                f"    long_sprints:               {c['long_sprints']}",
                f"    mid_distance:               {c['mid_distance']}",
                f"    long_distance:              {c['long_distance']}",
                f"    specialized_unclassified*:  {c['specialized_unclassified']}",
                "    (* missing all 3 events, or Short/Long tie)",
                "",
            ]
        )

        # gender × group breakdown
        lines.append("  By gender × event group:")
        for gender in ("Men", "Women"):
            for eg in ("Sprints", "Distance"):
                row = next(
                    r
                    for r in summary_rows
                    if r["band"] == band
                    and r["gender"] == gender
                    and r["event_group"] == eg
                )
                lines.append(
                    f"    {gender} {eg}: n={row['n_athlete_seasons']}  "
                    f"bal={row['balanced_total']}  "
                    f"spec={row['specialized_total']} "
                    f"(short/mid={row['short_sprints'] + row['mid_distance']}, "
                    f"long={row['long_sprints'] + row['long_distance']}, "
                    f"uncl={row['specialized_unclassified']})"
                )
        lines.append("")

    lines.extend(
        [
            "Source: Balanced_Specialization_Counts/"
            "count_balanced_specialization_by_band.py",
            "Short/Long definitions: Short_Long_Specialization_Time_Models/",
            "analyze_short_long_specialization.py",
        ]
    )
    (OUTPUT_ROOT / "balanced_specialization_counts_report.txt").write_text(
        "\n".join(lines).rstrip() + "\n"
    )

    # Short findings file
    findings = [
        "Balanced / Specialization Counts — Findings",
        "===========================================",
        "",
    ]
    for lo, hi in TARGET_BANDS:
        band = f"{lo}-{hi}"
        c = band_totals[band]
        n = sum(c.values())
        bal = c["sprints_balanced"] + c["distance_balanced"]
        findings.append(
            f"Band {band}: n={n}  balanced={bal} "
            f"(sprints={c['sprints_balanced']}, distance={c['distance_balanced']})  "
            f"specialized={n - bal} "
            f"(short_sprints={c['short_sprints']}, long_sprints={c['long_sprints']}, "
            f"mid_distance={c['mid_distance']}, long_distance={c['long_distance']}, "
            f"unclassified={c['specialized_unclassified']})"
        )
    findings.extend(
        [
            "",
            "See balanced_specialization_counts_report.txt for gender × group detail.",
            "Source: count_balanced_specialization_by_band.py",
        ]
    )
    (OUTPUT_ROOT / "balanced_specialization_counts_findings.txt").write_text(
        "\n".join(findings).rstrip() + "\n"
    )
    print(f"Wrote counts to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
