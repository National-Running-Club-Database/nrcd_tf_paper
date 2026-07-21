"""Best-event analysis for distance data including relays (2024–2026 outdoor)."""

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

DISTANCE_INDIVIDUAL = {
    9: "800m",
    11: "1500m",
    17: "5000m",
    20: "3000m Steeplechase",
}

CONFIG = DisciplineConfig(
    discipline_label="Distance",
    file_prefix="Relays_Distance",
    output_root=ROOT,
    individual_event_order=["800m", "1500m", "3000m Steeplechase", "5000m"],
    event_map_for_gender=lambda _gender: merge_event_maps(DISTANCE_INDIVIDUAL, STANDARD_RELAY_MAP),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=("Men", "Women", "Both"), default="Both")
    args = parser.parse_args()
    genders = ("Men", "Women") if args.gender == "Both" else (args.gender,)
    run_discipline_analysis(CONFIG, genders)


if __name__ == "__main__":
    main()
