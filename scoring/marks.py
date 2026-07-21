"""Shared mark parsing and event taxonomy."""

from __future__ import annotations

import math
import re

TRACK_EVENTS = {
    "100m",
    "200m",
    "400m",
    "800m",
    "1500m",
    "Mile",
    "3000m",
    "3000m SC",
    "5000m",
    "10000m",
    "100mH",
    "110mH",
    "400mH",
    "4x100m",
    "4x400m",
}
FIELD_EVENTS = {"HJ", "PV", "LJ", "TJ", "SP", "DT", "HT", "JT"}

# Canonical club event keys used across scoring systems
EVENT_DISTANCES_M: dict[str, float] = {
    "100m": 100.0,
    "200m": 200.0,
    "400m": 400.0,
    "800m": 800.0,
    "1500m": 1500.0,
    "Mile": 1609.344,
    "3000m": 3000.0,
    "3000m SC": 3000.0,
    "5000m": 5000.0,
    "10000m": 10000.0,
    "100mH": 100.0,
    "110mH": 110.0,
    "400mH": 400.0,
    "4x100m": 400.0,
    "4x400m": 1600.0,
}

# Map NRCD / running_event.csv names → canonical keys
EVENT_ALIASES: dict[str, str] = {
    "100m": "100m",
    "200m": "200m",
    "400m": "400m",
    "800m": "800m",
    "1500m": "1500m",
    "Mile": "Mile",
    "3000m": "3000m",
    "5000m": "5000m",
    "10000m": "10000m",
    "3000m Steeplechase": "3000m SC",
    "3000m SC": "3000m SC",
    "3000 Steeplechase": "3000m SC",
    "100m Hurdles": "100mH",
    "100mH": "100mH",
    "110m Hurdles": "110mH",
    "110mH": "110mH",
    "400m Hurdles": "400mH",
    "400mH": "400mH",
    "4x100m": "4x100m",
    "4x400m": "4x400m",
    "Long Jump": "LJ",
    "LJ": "LJ",
    "Triple Jump": "TJ",
    "TJ": "TJ",
    "High Jump": "HJ",
    "HJ": "HJ",
    "Pole Vault": "PV",
    "PV": "PV",
    "Shot Put": "SP",
    "SP": "SP",
    "Discus": "DT",
    "DT": "DT",
    "Hammer Throw": "HT",
    "Hammer": "HT",
    "HT": "HT",
    "Javelin Throw": "JT",
    "Javelin": "JT",
    "JT": "JT",
}

RUNNING_EVENT_ID_TO_CANONICAL: dict[int, str] = {
    3: "100m",
    4: "200m",
    6: "400m",
    9: "800m",
    11: "1500m",
    13: "Mile",
    14: "3000m",
    17: "5000m",
    18: "10000m",
    20: "3000m SC",
    21: "4x100m",
    24: "4x400m",
    34: "100mH",
    35: "110mH",
    37: "400mH",
    38: "LJ",
    39: "TJ",
    40: "HJ",
    41: "SP",
    42: "DT",
    43: "HT",
    45: "JT",
    46: "PV",
}


def canonicalize_event(event: str | int | None) -> str | None:
    if event is None or (isinstance(event, float) and math.isnan(event)):
        return None
    if isinstance(event, (int, float)) and not isinstance(event, bool):
        return RUNNING_EVENT_ID_TO_CANONICAL.get(int(event))
    text = str(event).strip()
    if text.isdigit():
        return RUNNING_EVENT_ID_TO_CANONICAL.get(int(text))
    return EVENT_ALIASES.get(text)


def is_field_event(event: str) -> bool:
    canon = canonicalize_event(event) or event
    return canon in FIELD_EVENTS


def is_running_event(event: str) -> bool:
    canon = canonicalize_event(event) or event
    return canon in EVENT_DISTANCES_M


def parse_time_to_seconds(text: str) -> float | None:
    t = text.strip()
    if re.fullmatch(r"\d+\.\d{2}\.\d{2}", t):
        parts = t.split(".")
        t = f"{parts[0]}:{parts[1]}.{parts[2]}"
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    m = re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", t)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def parse_mark(event: str, raw: str | float | int | None) -> tuple[str | None, float | None]:
    """Return (display, numeric mark). Track → seconds; field → metres."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None, None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        canon = canonicalize_event(event) or str(event)
        if canon in FIELD_EVENTS:
            return f"{value:.2f}m", value
        return str(value), value

    text = str(raw).strip().rstrip("*")
    if not text or text.upper() in {
        "ND",
        "NM",
        "DNS",
        "DNF",
        "DQ",
        "FS",
        "FOUL",
        "NH",
        "—",
        "-",
        "NAN",
    }:
        return text or None, None

    canon = canonicalize_event(event) or str(event)
    if canon in FIELD_EVENTS:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*m?", text, re.I)
        if not m:
            return text, None
        metres = float(m.group(1))
        return f"{metres:.2f}m", metres

    seconds = parse_time_to_seconds(text)
    if seconds is None:
        return text, None
    return text, seconds
