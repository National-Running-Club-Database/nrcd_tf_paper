# non_relays_findings

RQ1 (event specialization & competitiveness) and RQ3 (population-level score dispersion)
analyses for **individual (non-relay) events**, NIRCA club outdoor track 2024–2026. The
companion relay-inclusive version of RQ1B/RQ1C lives in `../relays_findings/`.

## What this folder does

### Discipline-level "best event" counting (per-discipline subfolders)

`Distance_Events_Counting/`, `Sprints_Events_Counting/`, `Hurdles_Events_Counting/`,
`Jumps_Events_Counting/`, `Throws_Events_Counting/` each hold discipline-specific
`analyze_*.py` scripts that compute:

- **RQ1A best-event counts**: which event is each athlete's single highest World
  Athletics (WA) score, per gender/season/combined window.
- **RQ1A pairwise comparisons**: among athletes with results in both of two events,
  which event scores higher more often.

These subfolders are self-contained (own data CSVs, own `.pylibs/` vendored
dependencies in Distance/Sprints) and are not wired into `main.py` directly — see
"Dependencies" below.

### Top-level cross-discipline analyses (wired into `main.py`)

1. **`analyze_rq1_improved.py`** — re-runs RQ1A/B/C with proper inferential statistics:
   bootstrap CIs, chi-square goodness-of-fit, exact binomial + Cohen's h, Mann-Whitney U
   + rank-biserial r, Fisher's exact tests. Outputs to `improved_rq1_outputs/`.
2. **`analyze_rq1b_nationals.py`** — nationals top-8 WA distributions per event/year and
   3-year average 8th-place thresholds. Outputs to `RQ1B_Nationals/`.
3. **`analyze_rq1c_competitiveness.py`** — for each athlete, recommends the event with
   the largest margin above the 8th-place nationals threshold (or smallest deficit).
   Outputs to `RQ1C_Competitiveness/`.
4. **`analyze_rq3_population_iqr.py`** — ranks events by median/IQR of athletes'
   personal-best WA scores (population-level dispersion, not nationals-specific).
   Outputs `RQ3_Population_Dispersion/rq3_population_wa_iqr_by_event.txt` (+ `.csv`)
   and the analogous "marks by 25th percentile" files.

## How to run

```bash
cd non_relays_findings
python main.py                 # all four top-level analyses (WA baseline)
python main.py rq1_improved
python main.py rq1b_nationals
python main.py rq1c
python main.py rq3

# Multi-metric RQ1 (scientific: purdy/mercier; sports: wa/vdot)
python analyze_rq1_improved.py --metric mercier
python analyze_rq1_improved.py --metric vdot
python analyze_rq1_improved.py --all-metrics   # → improved_rq1_outputs/by_metric/
```

Scoring columns (`VDOT_*`, `Purdy_Points_*`, `Mercier_Points_*`) are added by
`python main.py enrich-scores` from the repo root. See [`../scoring/`](../scoring/).

To regenerate a specific discipline's best-event counts, run its script directly, e.g.
`python Sprints_Events_Counting/best_event_analysis.py` (these are not wired into
`main.py` since each discipline folder has its own set of season-specific script
variants rather than a single parameterized entrypoint).

## Layout

```
non_relays_findings/
├── main.py
├── analyze_rq1_improved.py
├── analyze_rq1b_nationals.py
├── analyze_rq1c_competitiveness.py
├── analyze_rq3_population_iqr.py
├── findings.md                          # full RQ1 narrative (source for Summary_findings.md)
├── Research_Questions.md (in Sprints_Events_Counting/)
├── improved_rq1_outputs/                # bootstrap/χ²/MWU/Fisher outputs
├── RQ1B_Nationals/                      # nationals top-8 distributions + thresholds
├── RQ1C_Competitiveness/                # most-competitive-event recommendations
├── RQ3_Population_Dispersion/           # population WA-score IQR rankings
├── Distance_Events_Counting/            # per-discipline best-event + pairwise scripts, data, plots
├── Sprints_Events_Counting/
├── Hurdles_Events_Counting/
├── Jumps_Events_Counting/
└── Throws_Events_Counting/
```

## Methods

| Component | Definition |
|-----------|------------|
| RQ1A (best event) | Event with the athlete's single highest WA personal best across the window |
| RQ1A (pairwise) | Among athletes with results in both events, which event scores higher more often |
| RQ1B (nationals top 8) | 100m/200m ranked from prelims; all other events from finals (or unlabeled nationals rows); places 1–8 retained |
| RQ1C (most competitive) | Event with the largest margin above the 3-year average 8th-place nationals WA threshold |
| RQ3 (population dispersion) | Median/Q1/Q3/IQR of athletes' personal-best WA scores per event (≥100 unique athletes, 2024–2026 combined) |

Improved-analysis inferential procedures (`analyze_rq1_improved.py`): bootstrap 95% CIs,
Wilson score CIs, chi-square goodness-of-fit, exact binomial tests + Cohen's h,
Mann-Whitney U + rank-biserial r + Cohen's d, Fisher's exact test, binomial test vs. 50%
null — Benjamini-Hochberg FDR correction applied across multiple comparisons (α = 0.05).

## Dependencies on other folders

- Discipline subfolders vendor their own `matplotlib`/`numpy`/`scipy` wheels under
  `.pylibs/` (Distance, Sprints) as a `sys.path` fallback; top-level scripts reuse those
  same `.pylibs/` directories.
- `number_of_events_question/` and `causal_analysis/` both read
  `non_relays_findings/Sprints_Events_Counting/.pylibs/` and
  `non_relays_findings/Distance_Events_Counting/running_event.csv` — do not remove those.
- `relays_findings/` mirrors this folder's RQ1B/RQ1C structure for relay events;
  `number_of_events_question/analyze_point_jump_by_competition_count.py` reads the
  `running_event.csv` lookup table defined here.

## Related documents

- [Summary_findings.md](Summary_findings.md) — condensed key results and limitations
- [findings.md](findings.md) — full RQ1 narrative with all findings/themes
- `improved_rq1_outputs/improved_findings.md` — narrative + statistical-inference addendum
