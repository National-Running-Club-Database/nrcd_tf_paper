"""Linear time models for all individual sprint and distance event pairs.

Predicts event B time from event A time using outdoor season PBs (2024–2026,
March 1+). Excludes 3000m Steeplechase. Each ordered pair (A -> B) is a separate
regression: B_seconds = intercept + slope * A_seconds.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELAYS_ROOT = PROJECT_ROOT / "Relays_Findings"
OUTPUT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(RELAYS_ROOT))
from relay_rq1_data import (  # noqa: E402
    SEASONS,
    parse_performance,
    points_col,
    season_rows,
)

SPRINT_EVENTS = {
    "100m": 3,
    "200m": 4,
    "400m": 6,
}

DISTANCE_EVENTS = {
    "800m": 9,
    "1500m": 11,
    "5000m": 17,
}

GROUP_CONFIG = {
    "Sprints": ("Sprinters_Relays_Findings", "Sprinters", SPRINT_EVENTS),
    "Distance": ("Distance_Relays_Findings", "Distance", DISTANCE_EVENTS),
}


@dataclass(frozen=True)
class TimeModel:
    gender: str
    event_group: str
    from_event: str
    to_event: str
    n: int
    intercept: float
    slope: float
    r: float
    rmse: float
    median_abs_error: float
    ratio_median: float
    ratio_p25: float
    ratio_p75: float
    from_time_median: float
    to_time_median: float

    def formula(self) -> str:
        if abs(self.intercept) < 1e-9:
            return f"{self.to_event} = {self.slope:.3f} × {self.from_event}"
        if self.intercept >= 0:
            return f"{self.to_event} = {self.intercept:.3f} + {self.slope:.3f} × {self.from_event}"
        return f"{self.to_event} = {self.slope:.3f} × {self.from_event} − {abs(self.intercept):.3f}"

    def to_row(self) -> dict:
        return {
            "gender": self.gender,
            "event_group": self.event_group,
            "from_event": self.from_event,
            "to_event": self.to_event,
            "n_athletes": self.n,
            "intercept_seconds": round(self.intercept, 4),
            "slope": round(self.slope, 4),
            "r": round(self.r, 4),
            "rmse_seconds": round(self.rmse, 4),
            "median_abs_error_seconds": round(self.median_abs_error, 4),
            "ratio_median": round(self.ratio_median, 4),
            "ratio_p25": round(self.ratio_p25, 4),
            "ratio_p75": round(self.ratio_p75, 4),
            "from_event_median_seconds": round(self.from_time_median, 3),
            "to_event_median_seconds": round(self.to_time_median, 3),
            "formula": self.formula(),
        }


def linreg(pairs: list[tuple[float, float]]) -> tuple[float, float, float, float, float]:
    """Return intercept, slope, r, rmse, median_abs_error."""
    n = len(pairs)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in pairs]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    den_r = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    r = num / den_r if den_r else 0.0
    med_abs = statistics.median(abs(r) for r in resid)
    return intercept, slope, r, rmse, med_abs


def ratio_quartiles(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    ratios = sorted(y / x for x, y in pairs if x > 0)
    if not ratios:
        return 0.0, 0.0, 0.0
    n = len(ratios)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(p * (n - 1))))
        return ratios[idx]

    return statistics.median(ratios), pct(0.25), pct(0.75)


def load_season_pbs(
    folder: str,
    prefix: str,
    gender: str,
    event_name_to_id: dict[str, int],
) -> dict[str, dict[str, float]]:
    """athlete_id -> {event_name: best_time_seconds}."""
    bests: dict[str, dict[str, float]] = {}
    id_to_name = {eid: name for name, eid in event_name_to_id.items()}
    pcol = points_col(gender)

    for year in SEASONS:
        for row in season_rows(folder, prefix, gender, year):
            event_id = int(row["running_event_id"])
            if event_id not in id_to_name:
                continue
            wa = float(row.get(pcol) or 0)
            if wa <= 0:
                continue
            aid = (row.get("athlete_id") or "").strip()
            if not aid:
                continue
            t = parse_performance(row["result_time"], event_id)
            if math.isinf(t) or t <= 0:
                continue
            name = id_to_name[event_id]
            current = bests.get(aid, {}).get(name)
            if current is None or t < current:
                bests.setdefault(aid, {})[name] = t

    return bests


def fit_pairwise_models(
    gender: str,
    event_group: str,
    event_name_to_id: dict[str, int],
    athlete_bests: dict[str, dict[str, float]],
) -> list[TimeModel]:
    names = list(event_name_to_id.keys())
    models: list[TimeModel] = []

    for from_ev, to_ev in permutations(names, 2):
        pairs = [
            (bests[from_ev], bests[to_ev])
            for bests in athlete_bests.values()
            if from_ev in bests and to_ev in bests
        ]
        if len(pairs) < 10:
            continue
        intercept, slope, r, rmse, med_abs = linreg(pairs)
        r_med, r_p25, r_p75 = ratio_quartiles(pairs)
        from_times = [p[0] for p in pairs]
        to_times = [p[1] for p in pairs]
        models.append(
            TimeModel(
                gender=gender,
                event_group=event_group,
                from_event=from_ev,
                to_event=to_ev,
                n=len(pairs),
                intercept=intercept,
                slope=slope,
                r=r,
                rmse=rmse,
                median_abs_error=med_abs,
                ratio_median=r_med,
                ratio_p25=r_p25,
                ratio_p75=r_p75,
                from_time_median=statistics.median(from_times),
                to_time_median=statistics.median(to_times),
            )
        )

    return models


GROUP_ORDER = {"Sprints": 0, "Distance": 1}


def model_sort_key(m: TimeModel) -> tuple:
    return (GROUP_ORDER[m.event_group], m.gender, m.from_event, m.to_event)


def write_csv(models: list[TimeModel], path: Path) -> None:
    rows = [m.to_row() for m in sorted(models, key=model_sort_key)]
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(models: list[TimeModel], path: Path) -> None:
    lines = [
        "Cross-Event Linear Time Models — Sprints & Distance",
        "===================================================",
        "",
        "Method:",
        "  • Data: relay-inclusive outdoor CSVs, 2024–2026, results on or after March 1.",
        "  • Athlete-season PB: fastest valid mark per athlete per event (WA > 0).",
        "  • Model: to_event_seconds = intercept + slope × from_event_seconds.",
        "  • Each ordered pair (A -> B) is separate from (B -> A).",
        "  • 3000m Steeplechase excluded from distance.",
        "",
        "Use:",
        "  Predict time in an unrun event from a known PB, then convert predicted time",
        "  to World Athletics points in the TARGET event table. Treat median_abs_error",
        "  and RMSE as typical uncertainty bands, not guarantees.",
        "",
    ]

    current_section = None
    for m in sorted(models, key=model_sort_key):
        section = f"{m.event_group} — {m.gender}"
        if section != current_section:
            lines.extend(["", section, "-" * len(section)])
            current_section = section

        lines.extend(
            [
                f"",
                f"{m.from_event} -> {m.to_event}  (n={m.n}, r={m.r:.3f})",
                f"  Formula: {m.formula()}",
                f"  RMSE: {m.rmse:.3f}s | median |error|: {m.median_abs_error:.3f}s",
                f"  Ratio {m.to_event}/{m.from_event}: median {m.ratio_median:.3f} "
                f"(IQR {m.ratio_p25:.3f}–{m.ratio_p75:.3f})",
                f"  Median times in sample: {m.from_event} {m.from_time_median:.2f}s, "
                f"{m.to_event} {m.to_time_median:.2f}s",
            ]
        )

    lines.extend(
        [
            "",
            "Pair summary tables",
            "-------------------",
        ]
    )

    for group in ("Sprints", "Distance"):
        for gender in ("Men", "Women"):
            sub = [m for m in models if m.event_group == group and m.gender == gender]
            if not sub:
                continue
            lines.append(f"\n{group} — {gender}")
            lines.append(f"  {'From':<8} {'To':<8} {'n':>5} {'r':>6} {'RMSE':>8} {'Med|err|':>8}  Formula")
            for m in sorted(sub, key=lambda x: (x.from_event, x.to_event)):
                lines.append(
                    f"  {m.from_event:<8} {m.to_event:<8} {m.n:>5} {m.r:>6.3f} "
                    f"{m.rmse:>7.3f}s {m.median_abs_error:>7.3f}s  {m.formula()}"
                )

    lines.extend(
        [
            "",
            "Source: time_models/build_cross_event_time_models.py",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_models: list[TimeModel] = []

    for event_group, (folder, prefix, events) in GROUP_CONFIG.items():
        for gender in ("Men", "Women"):
            bests = load_season_pbs(folder, prefix, gender, events)
            all_models.extend(fit_pairwise_models(gender, event_group, events, bests))

    write_csv(all_models, OUTPUT_ROOT / "cross_event_time_models.csv")
    write_report(all_models, OUTPUT_ROOT / "cross_event_time_models_report.txt")
    print(f"Wrote {len(all_models)} models to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
