"""Gardner–Purdy points (Portuguese Scoring Tables / Hoffman implementation).

Matches *Computerized Running Training Programs* via Patrick Hoffman's C port
(Portuguese table speeds, 950-point scaling, startup + turn slowdown).
"""

from __future__ import annotations

import math

from scoring.marks import EVENT_DISTANCES_M, canonicalize_event, is_field_event

# Distance (m), straight-line speed (m/s) — Portuguese tables through 1936 WRs
PORTUGUESE_TABLE: list[tuple[float, float]] = [
    (40.0, 11.000),
    (50.0, 10.9960),
    (60.0, 10.9830),
    (70.0, 10.9620),
    (80.0, 10.934),
    (90.0, 10.9000),
    (100.0, 10.8600),
    (110.0, 10.8150),
    (120.0, 10.765),
    (130.0, 10.7110),
    (140.0, 10.6540),
    (150.0, 10.5940),
    (160.0, 10.531),
    (170.0, 10.4650),
    (180.0, 10.3960),
    (200.0, 10.2500),
    (220.0, 10.096),
    (240.0, 9.9350),
    (260.0, 9.7710),
    (280.0, 9.6100),
    (300.0, 9.455),
    (320.0, 9.3070),
    (340.0, 9.1660),
    (360.0, 9.0320),
    (380.0, 8.905),
    (400.0, 8.7850),
    (450.0, 8.5130),
    (500.0, 8.2790),
    (550.0, 8.083),
    (600.0, 7.9210),
    (700.0, 7.6690),
    (800.0, 7.4960),
    (900.0, 7.32000),
    (1000.0, 7.18933),
    (1200.0, 6.98066),
    (1500.0, 6.75319),
    (2000.0, 6.50015),
    (2500.0, 6.33424),
    (3000.0, 6.21913),
    (3500.0, 6.13510),
    (4000.0, 6.07040),
    (4500.0, 6.01822),
    (5000.0, 5.97432),
    (6000.0, 5.90181),
    (7000.0, 5.84156),
    (8000.0, 5.78889),
    (9000.0, 5.74211),
    (10000.0, 5.70050),
    (12000.0, 5.62944),
    (15000.0, 5.54300),
    (20000.0, 5.43785),
    (25000.0, 5.35842),
    (30000.0, 5.29298),
    (35000.0, 5.23538),
    (40000.0, 5.18263),
    (50000.0, 5.08615),
    (60000.0, 4.99762),
    (80000.0, 4.83617),
    (100000.0, 4.68988),
]

C1 = 0.20
C2 = 0.08
C3 = 0.0065


def fraction_on_turns(distance_m: float) -> float:
    if distance_m < 110:
        return 0.0
    laps = math.floor(distance_m / 400.0)
    meters = distance_m - laps * 400.0
    if meters <= 50:
        part_lap = 0.0
    elif meters <= 150:
        part_lap = meters - 50.0
    elif meters <= 250:
        part_lap = 100.0
    elif meters <= 350:
        part_lap = 100.0 + (meters - 250.0)
    else:
        part_lap = 200.0
    turn_distance = laps * 200.0 + part_lap
    return turn_distance / distance_m


def purdy_from_distance_time(distance_m: float, time_s: float) -> float | None:
    if distance_m <= 0 or time_s is None or not math.isfinite(time_s) or time_s <= 0:
        return None
    # Find bracketing Portuguese-table entries
    if distance_m < PORTUGUESE_TABLE[0][0] or distance_m > PORTUGUESE_TABLE[-1][0]:
        return None
    i = 0
    while i < len(PORTUGUESE_TABLE) and PORTUGUESE_TABLE[i][0] < distance_m:
        i += 1
    if i == 0:
        d3, speed3 = PORTUGUESE_TABLE[0]
        t3 = d3 / speed3
        t = t3
        v = distance_m / t
    else:
        d3, speed3 = PORTUGUESE_TABLE[i]
        d1, speed1 = PORTUGUESE_TABLE[i - 1]
        t3 = d3 / speed3
        t1 = d1 / speed1
        t = t1 + (t3 - t1) * (distance_m - d1) / (d3 - d1)
        v = distance_m / t

    t950 = t + C1 + C2 * v + C3 * fraction_on_turns(distance_m) * v * v
    k = 0.0654 - 0.00258 * v
    if abs(k) < 1e-12:
        return None
    a = 85.0 / k
    b = 1.0 - 950.0 / a
    points = a * (t950 / time_s - b)
    if not math.isfinite(points):
        return None
    return round(points, 2)


def purdy_points(gender: str, event: str, mark_value: float) -> float | None:
    """Gardner–Purdy points for running marks; None for field events."""
    _ = gender
    canon = canonicalize_event(event)
    if not canon or is_field_event(canon):
        return None
    dist = EVENT_DISTANCES_M.get(canon)
    if dist is None:
        return None
    return purdy_from_distance_time(dist, mark_value)
