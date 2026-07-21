# new_steeplechase_data

Re-runs the Distance research questions (RQ1 best-event, RQ1B nationals, RQ1C
competitiveness) with **corrected 3000m Steeplechase World Athletics (WA) point
scoring**, mirroring the structure of `../relays_findings/` outputs. It also rebuilds
the point-band time models with steeplechase included as a distance event.

> This file supersedes `README.txt` (kept in place for compatibility / history — see the
> note at the top of that file).

## Why this folder exists

The legacy steeplechase WA scoring in `relays_findings/Distance_Relays_Findings/` and
`non_relays_findings/` undercounted steeplechase points (see `DATA_NOTES.txt`). After
the gender/scoring fix, steeplechase means roughly **doubled** relative to the legacy
copies. This folder re-derives every distance-dependent analysis from the corrected data
so results can be compared against the legacy (pre-fix) numbers.

## What this folder does

1. **`prepare_distance_data.py`** — validates gender and builds corrected Distance
   relay-inclusive working CSVs (`Distance_Relays_Findings/`) from the source
   `Relays_Distance_*_Outdoor_*_Data.csv` files in this folder.
2. **`analyze_rq1_best_event_distance.py`** — RQ1 best-event counts + pairwise
   comparisons for distance (relay-inclusive), using corrected steeple WA.
3. **`analyze_rq1b_nationals_distance.py`** — nationals top-8 WA distributions and
   8th-place thresholds for distance events, corrected steeple WA. Outputs to
   `RQ1B_Nationals/`.
4. **`analyze_rq1c_competitiveness_distance.py`** — most-competitive-event
   recommendation for distance, corrected steeple WA. Outputs to `RQ1C_Competitiveness/`.
5. **`update_rq1b_all_events_rankings.py`** — compares legacy vs. corrected 8th-place
   thresholds across **all** events (not just distance) to quantify the ranking impact.
6. **`Feature_Importance_Point_Band_Time_Models/analyze_feature_importance_with_steeple.py`**
   — rebuilds the point-band feature-importance time models (WA bands 750–950,
   800–1000, 850–1050) with steeplechase included as a distance event, and checks
   whether steeplechase inclusion adds athlete-seasons to each band (a "gate check").
7. **`coaches_analysis/`** — coaching recommendation reports (nationals scoring focus,
   points maximization) rebuilt with the corrected distance/steeple numbers; sprints,
   hurdles, jumps, and throws recommendations are carried over unchanged from
   `../coaches_analysis/`.

## How to run

```bash
cd new_steeplechase_data
python main.py                    # full distance RQ pipeline (default: all)
python main.py distance_rq        # prepare_distance_data -> RQ1 -> RQ1B -> RQ1C
python main.py rankings           # legacy-vs-corrected 8th-place threshold comparison
python main.py feature_importance # steeplechase-inclusive point-band models
python main.py all
```

`run_distance_rq_analyses.py` (wired as `distance_rq`) is the primary pipeline script;
it chains `prepare_distance_data.py → analyze_rq1_best_event_distance.py →
analyze_rq1b_nationals_distance.py → analyze_rq1c_competitiveness_distance.py`. Each
step can also be run individually (see `README.txt` for the step-by-step commands).

## Layout

```
new_steeplechase_data/
├── main.py
├── run_distance_rq_analyses.py           # primary pipeline (4 steps, chained)
├── prepare_distance_data.py
├── analyze_rq1_best_event_distance.py
├── analyze_rq1b_nationals_distance.py
├── analyze_rq1c_competitiveness_distance.py
├── update_rq1b_all_events_rankings.py    # legacy vs corrected threshold comparison
├── README.txt                            # legacy quick-reference (superseded by this file)
├── DATA_NOTES.txt                        # gender/scoring validation notes per source CSV
├── Relays_Distance_{Men,Women}_Outdoor_{2024,2025,2026}_Data.csv   # source data
├── Distance_Relays_Findings/             # working copies + best-event outputs
├── RQ1B_Nationals/                       # nationals top-8 distributions + thresholds
├── RQ1C_Competitiveness/                 # most-competitive-event recommendations
├── Feature_Importance_Point_Band_Time_Models/  # steeplechase-inclusive point-band models
└── coaches_analysis/                     # coaching reports with corrected steeple numbers
```

## Data note

See `DATA_NOTES.txt` for per-file validation. After the fix, all six
`Relays_Distance_*` CSVs pass 100% gender-match validation, and steeplechase WA means
roughly doubled vs. the legacy copies (e.g., men's 2024: legacy mean 202.6 → corrected
mean 458.2).

## Dependencies on other folders

- **Reads from**: nothing outside this folder for the core pipeline — source CSVs and
  intermediate working files are self-contained here.
- **Compared against**: `relays_findings/RQ1B_Nationals/` and `RQ1C_Competitiveness/`
  (legacy, pre-fix thresholds) via `update_rq1b_all_events_rankings.py`.
- **Downstream**: `coaches_analysis/` (top-level) is the original, pre-fix coaching
  report; `new_steeplechase_data/coaches_analysis/` is the corrected mirror.

## Related documents

- [Summary_findings.md](Summary_findings.md) — key numbers and limitations
- `coaches_analysis/README.txt` — what changed in the coaching recommendations
- `Feature_Importance_Point_Band_Time_Models/steeple_band_gate_findings.txt` — band gate-check results
