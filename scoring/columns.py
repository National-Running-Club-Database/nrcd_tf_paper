"""CSV column helpers for multi-metric scoring."""

from __future__ import annotations

METRIC_ALIASES = {
    "wa": "wa",
    "world_athletics": "wa",
    "world_athletics_points": "wa",
    "vdot": "vdot",
    "purdy": "purdy",
    "gardner_purdy": "purdy",
    "mercier": "mercier",
    "mercier_1999": "mercier",
}

# Framing groups for paper reporting
SCIENTIFIC_METRICS = ("purdy", "mercier")
SPORTS_METRICS = ("wa", "vdot")
ALL_METRICS = ("wa", "vdot", "purdy", "mercier")

COLUMN_TEMPLATES = {
    "wa": "World_Athletics_Points_{gender}",
    "vdot": "VDOT_{gender}",
    "purdy": "Purdy_Points_{gender}",
    "mercier": "Mercier_Points_{gender}",
}


def normalize_metric(metric: str) -> str:
    key = metric.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in METRIC_ALIASES:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from: {', '.join(ALL_METRICS)}"
        )
    return METRIC_ALIASES[key]


def _gender_label(gender: str) -> str:
    g = gender.strip().lower()
    if g in {"m", "men", "male", "man"}:
        return "Men"
    if g in {"w", "women", "female", "woman", "f"}:
        return "Women"
    if gender in {"Men", "Women"}:
        return gender
    raise ValueError(f"Unknown gender: {gender!r}")


def points_col(gender: str, metric: str = "wa") -> str:
    """Return the CSV column name for a gender + scoring metric."""
    m = normalize_metric(metric)
    label = _gender_label(gender)
    return COLUMN_TEMPLATES[m].format(gender=label)


def metric_framing(metric: str) -> str:
    m = normalize_metric(metric)
    if m in SCIENTIFIC_METRICS:
        return "scientific"
    if m in SPORTS_METRICS:
        return "sports"
    return "other"
