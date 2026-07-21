"""Best-event analysis for hurdles data including relays (2024–2026 outdoor)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relay_best_event_analysis import (
    DisciplineConfig,
    STANDARD_RELAY_MAP,
    merge_event_maps,
    run_discipline_analysis,
)

ROOT = Path(__file__).parent

MEN_HURDLES = {35: "110m Hurdles", 37: "400m Hurdles"}
WOMEN_HURDLES = {34: "100m Hurdles", 37: "400m Hurdles"}


def hurdles_event_map(gender: str) -> dict[int, str]:
    individual = MEN_HURDLES if gender == "Men" else WOMEN_HURDLES
    return merge_event_maps(individual, STANDARD_RELAY_MAP)


CONFIG = DisciplineConfig(
    discipline_label="Hurdles",
    file_prefix="Relays_Hurdles",
    output_root=ROOT,
    individual_event_order=["110m Hurdles", "100m Hurdles", "400m Hurdles"],
    event_map_for_gender=hurdles_event_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=("Men", "Women", "Both"), default="Both")
    args = parser.parse_args()
    genders = ("Men", "Women") if args.gender == "Both" else (args.gender,)
    run_discipline_analysis(CONFIG, genders)


if __name__ == "__main__":
    main()
