"""README for the shared multi-metric scoring package.

## Framing

| Role | Metrics |
|------|---------|
| **Scientific** | Gardner–Purdy points + Mercier (1999 documented linear tables) |
| **Sports / coaching** | World Athletics Points + VDOT (Daniels) |

## CLI

```bash
python -m scoring.enrich_csvs          # append metric columns to discipline CSVs
python -m scoring.compare_metrics      # RQ1A-style best-event agreement across metrics
python main.py enrich-scores
python main.py compare-metrics
```

## API

```python
from scoring import score, points_col, scientific_scores, sports_scores

score(11.50, "100m", "men")
# -> {wa, vdot, purdy, mercier}

points_col("Men", "mercier")  # Mercier_Points_Men
```

## Sources

- WA: outdoor 2025 quadratic tables (`scoring/data/coefficients-2025.json`)
- VDOT: Daniels & Gilbert *Oxygen Power* equations
- Purdy: Gardner & Purdy Portuguese tables / Hoffman implementation (950 scaling)
- Mercier 1999: Mureika/Covington/Mercier how-to; coefficients reconstructed from
  published Table 3 calibrations — **not** proprietary Mercier–Rioux tables

## Limitations

- Purdy and VDOT are undefined for field events (use Mercier scientifically, WA for sports).
- Mercier 1999 is era-specific (1995–1998 world rankings).
- VDOT is a weak model for short sprints; still reported as the coaching aerobic metric.
"""
