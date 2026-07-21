"""World Athletics outdoor scoring (2025 quadratic tables)."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from scoring.marks import canonicalize_event

DATA_DIR = Path(__file__).resolve().parent / "data"
COEFF_PATH = DATA_DIR / "coefficients-2025.json"

# Canonical event → WA table key (gender-aware for hurdles)
WA_EVENT_KEYS_MEN = {
    "100m": "100m",
    "200m": "200m",
    "400m": "400m",
    "800m": "800m",
    "1500m": "1500m",
    "Mile": "Mile",
    "3000m": "3000m",
    "3000m SC": "3000m SC",
    "5000m": "5000m",
    "10000m": "10000m",
    "110mH": "110mH",
    "400mH": "400mH",
    "HJ": "HJ",
    "PV": "PV",
    "LJ": "LJ",
    "TJ": "TJ",
    "SP": "SP",
    "DT": "DT",
    "HT": "HT",
    "JT": "JT",
    "4x100m": "4x100m",
    "4x400m": "4x400m",
}
WA_EVENT_KEYS_WOMEN = {
    **WA_EVENT_KEYS_MEN,
    "100mH": "100mH",
    # Women do not use 110mH; map common mislabel to 100mH
    "110mH": "100mH",
}


@lru_cache(maxsize=1)
def load_coefficients() -> dict:
    with open(COEFF_PATH) as f:
        return json.load(f)


def _gender_key(gender: str) -> str:
    g = gender.strip().lower()
    if g in {"m", "men", "male", "man"}:
        return "men"
    if g in {"w", "women", "female", "woman", "f"}:
        return "women"
    raise ValueError(f"Unknown gender: {gender!r}")


def wa_table_key(event: str, gender: str) -> str | None:
    canon = canonicalize_event(event)
    if not canon:
        return None
    g = _gender_key(gender)
    table = WA_EVENT_KEYS_MEN if g == "men" else WA_EVENT_KEYS_WOMEN
    return table.get(canon)


def wa_points(
    coeffs: dict | None,
    gender: str,
    event: str,
    mark_value: float,
) -> int | None:
    """Score a mark with WA outdoor tables. Mark is seconds (track) or metres (field)."""
    if mark_value is None or not math.isfinite(mark_value) or mark_value <= 0:
        return None
    coeffs = coeffs or load_coefficients()
    g = _gender_key(gender)
    key = wa_table_key(event, gender)
    if not key or key not in coeffs.get(g, {}):
        return None
    a, b, c = coeffs[g][key]
    raw = a * mark_value * mark_value + b * mark_value + c
    if not math.isfinite(raw):
        return None
    return max(int(math.floor(raw)), 0)
