"""Mercier 1999 documented linear scoring (Mureika / Covington / Mercier).

Points = A * x + B, where x is weighted speed (track) or sqrt(mark) (field).
Coefficients are reconstructed from published Jan-2000 calibration tables
(max/mean/median performances at fixed scores + WR Mercier scores).

This is NOT the proprietary Mercier–Rioux commercial tables.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from scoring.marks import EVENT_DISTANCES_M, canonicalize_event

DATA_DIR = Path(__file__).resolve().parent / "data"
COEFF_PATH = DATA_DIR / "mercier_1999_coefficients.json"


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


def mercier_points(gender: str, event: str, mark_value: float) -> int | None:
    if mark_value is None or not math.isfinite(mark_value) or mark_value <= 0:
        return None
    canon = canonicalize_event(event)
    if not canon:
        return None
    g = _gender_key(gender)
    data = load_coefficients()
    entry = data["coefficients"].get(f"{g}:{canon}")
    if not entry:
        return None
    kind = entry["kind"]
    a = entry["A"]
    b = entry["B"]
    if kind == "track":
        dist = entry.get("distance_m") or EVENT_DISTANCES_M.get(canon)
        if not dist:
            return None
        x = dist / mark_value
    else:
        x = math.sqrt(mark_value)
    raw = a * x + b
    if not math.isfinite(raw):
        return None
    # Athletics Canada / Mercier practice: round up to next whole number
    return max(int(math.ceil(raw - 1e-9)), 0)
