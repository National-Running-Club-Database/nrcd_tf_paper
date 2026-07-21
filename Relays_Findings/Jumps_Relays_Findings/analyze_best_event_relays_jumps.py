"""Best-event analysis for jumps data including relays (2024–2026 outdoor)."""

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

JUMPS_INDIVIDUAL = {
    38: "Long Jump",
    39: "Triple Jump",
    40: "High Jump",
}

CONFIG = DisciplineConfig(
    discipline_label="Jumps",
    file_prefix="Relays_Jumps",
    output_root=ROOT,
    individual_event_order=["Long Jump", "Triple Jump", "High Jump"],
    event_map_for_gender=lambda _gender: merge_event_maps(JUMPS_INDIVIDUAL, STANDARD_RELAY_MAP),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender", choices=("Men", "Women", "Both"), default="Both")
    args = parser.parse_args()
    genders = ("Men", "Women") if args.gender == "Both" else (args.gender,)
    run_discipline_analysis(CONFIG, genders)


if __name__ == "__main__":
    main()
