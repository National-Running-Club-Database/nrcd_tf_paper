# Cross-metric scoring — summary findings

**Scientific framing:** Gardner–Purdy + Mercier (1999 documented reconstruction)  
**Sports / coaching framing:** World Athletics Points + VDOT (Daniels)

Artifacts: [`scoring/output/`](output/)

## Best-event agreement (RQ1A-style, multi-event athletes)

Mean pairwise agreement that the same event is “best” (outdoor disciplines, 2024–2026):

| Pair | Mean agreement |
|------|----------------|
| WA vs Purdy | **85.8%** |
| WA vs Mercier | **82.2%** |
| Purdy vs Mercier (scientific pair) | **74.9%** |
| WA vs VDOT | **71.3%** |
| VDOT vs Purdy / Mercier | ~57–59% |

Interpretation: specialization conclusions are **most stable between WA and Gardner–Purdy**; Mercier 1999 agrees closely with WA on field-inclusive slices; VDOT (aerobic coaching metric) reorders short-sprint / mixed cohorts more often.

## Pipeline wiring

- Discipline CSVs enriched with `VDOT_*`, `Purdy_Points_*`, `Mercier_Points_*` (legacy `World_Athletics_Points_*` preserved).
- `non_relays_findings/analyze_rq1_improved.py --metric {wa,vdot,purdy,mercier}`
- `number_of_events_question/... --metric ...` → `by_metric/<metric>/`
- Shared selector: `scoring.columns.points_col(gender, metric)`
- CLI: `python main.py enrich-scores` · `python main.py compare-metrics`

## Limitations

- Purdy/VDOT undefined for field events (Mercier + WA carry those analyses).
- Mercier coefficients are a 1999 linear reconstruction, not Mercier–Rioux.
- Nationals 8th-place thresholds remain WA-calibrated; RQ1C clear-rate plots only run for `--metric wa`.
