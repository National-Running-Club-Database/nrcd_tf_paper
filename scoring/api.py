"""Unified scoring API."""

from __future__ import annotations

from typing import Any

from scoring.marks import canonicalize_event, parse_mark
from scoring.mercier import mercier_points
from scoring.purdy import purdy_points
from scoring.vdot import vdot_points
from scoring.wa import load_coefficients, wa_points

MetricDict = dict[str, float | int | None]


def score(
    mark: str | float | int | None,
    event: str | int,
    gender: str,
    *,
    coeffs: dict | None = None,
) -> MetricDict:
    """Score a performance under all four systems.

    Returns keys: wa, vdot, purdy, mercier (None where undefined).
    """
    canon = canonicalize_event(event)
    _, value = parse_mark(canon or str(event), mark)
    if value is None or canon is None:
        return {"wa": None, "vdot": None, "purdy": None, "mercier": None}

    wa_coeffs = coeffs if coeffs is not None else load_coefficients()
    return {
        "wa": wa_points(wa_coeffs, gender, canon, value),
        "vdot": vdot_points(gender, canon, value),
        "purdy": purdy_points(gender, canon, value),
        "mercier": mercier_points(gender, canon, value),
    }


def scientific_scores(
    mark: str | float | int | None,
    event: str | int,
    gender: str,
) -> dict[str, float | int | None]:
    """Gardner–Purdy + Mercier 1999."""
    full = score(mark, event, gender)
    return {"purdy": full["purdy"], "mercier": full["mercier"]}


def sports_scores(
    mark: str | float | int | None,
    event: str | int,
    gender: str,
) -> dict[str, float | int | None]:
    """World Athletics Points + VDOT."""
    full = score(mark, event, gender)
    return {"wa": full["wa"], "vdot": full["vdot"]}


def score_row(row: dict[str, Any], *, gender: str | None = None) -> MetricDict:
    """Score a CSV-like row with result_time / running_event_id / gender."""
    g = gender or row.get("gender") or row.get("Gender")
    if g in {"M", "m"}:
        g = "men"
    elif g in {"W", "w", "F", "f"}:
        g = "women"
    event = row.get("running_event_id") or row.get("event") or row.get("Event")
    mark = row.get("result_time") or row.get("mark") or row.get("Mark")
    return score(mark, event, str(g))
