"""Best-event analysis for throws data including relays (2024–2026 outdoor)."""

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

THROWS_INDIVIDUAL = {
    41: "Shot Put",
    42: "Discus",
    43: "Hammer Throw",
    45: "Javelin Throw",
}

CONFIG = DisciplineConfig(
    discipline_label="Throws",
    file_prefix="Relays_Throws",
    output_root=ROOT,
    individual_event_order=["Shot Put", "Discus", "Hammer Throw", "Javelin Throw"],
    event_map_for_gender=lambda _gender: merge_event_maps(THROWS_INDIVIDUAL, STANDARD_RELAY_MAP),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=("Men", "Women", "Both"), default="Both")
    args = parser.parse_args()
    genders = ("Men", "Women") if args.gender == "Both" else (args.gender,)
    run_discipline_analysis(CONFIG, genders)


if __name__ == "__main__":
    main()
