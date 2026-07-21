"""Jack Daniels / Gilbert VDOT (Oxygen Power) for running distances."""

from __future__ import annotations

import math

from scoring.marks import EVENT_DISTANCES_M, canonicalize_event, is_field_event


def _percent_max(time_min: float) -> float:
    # Daniels & Gilbert Oxygen Power
    return (
        0.8
        + 0.1894393 * math.exp(-0.012778 * time_min)
        + 0.2989558 * math.exp(-0.1932605 * time_min)
    )


def _vo2(velocity_m_per_min: float) -> float:
    return -4.60 + 0.182258 * velocity_m_per_min + 0.000104 * velocity_m_per_min**2


def vdot_from_distance_time(distance_m: float, time_s: float) -> float | None:
    if distance_m <= 0 or time_s is None or not math.isfinite(time_s) or time_s <= 0:
        return None
    time_min = time_s / 60.0
    velocity = distance_m / time_min  # m/min
    pct = _percent_max(time_min)
    if pct <= 0:
        return None
    vo2 = _vo2(velocity)
    vdot = vo2 / pct
    if not math.isfinite(vdot) or vdot <= 0:
        return None
    return round(vdot, 2)


def vdot_points(gender: str, event: str, mark_value: float) -> float | None:
    """Return VDOT for a running mark; None for field / unknown events.

    Gender is accepted for API symmetry but Daniels VDOT is not gender-specific.
    """
    _ = gender
    canon = canonicalize_event(event)
    if not canon or is_field_event(canon):
        return None
    dist = EVENT_DISTANCES_M.get(canon)
    if dist is None:
        return None
    return vdot_from_distance_time(dist, mark_value)
