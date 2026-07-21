"""Multi-metric performance scoring for the track specialization paper.

Scientific framing: Gardner–Purdy + Mercier (1999 documented reconstruction)
Sports / coaching framing: World Athletics Points + VDOT (Daniels)
"""

from scoring.api import scientific_scores, score, sports_scores
from scoring.columns import METRIC_ALIASES, points_col

__all__ = [
    "score",
    "scientific_scores",
    "sports_scores",
    "points_col",
    "METRIC_ALIASES",
]
