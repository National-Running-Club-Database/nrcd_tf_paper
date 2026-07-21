# coaches_analysis

**Document-only module.** This folder translates the RQ1B (nationals 8th-place
thresholds) and RQ1C (nationals-margin competitiveness) outputs from
`../relays_findings/` and `../non_relays_findings/` into three practical coaching
reports for roster/event-assignment decisions. There are no analysis scripts here —
only synthesized `.txt` recommendation reports.

A steeplechase-corrected mirror of the distance/steeple recommendations in this folder
lives in `../new_steeplechase_data/coaches_analysis/`.

## What this folder contains

| File | Framing | Use when... |
|------|---------|--------------|
| `Nationals_Scoring_Focus_Men_Women.txt` | Clear-rate / scorer-count: maximize how many athletes clear the 8th-place bar | Your roster is still building toward nationals-caliber marks |
| `Nationals_Scoring_Focus_Clearing_Roster.txt` | Margin + field-density: spread clearing-capable athletes across events by margin and pool size | Many athletes already clear the bar in multiple events, and you want to maximize scoring *lines* |
| `Nationals_Points_Maximization_Clearing_Roster.txt` | Stack thin, high-margin events for total point value (10-8-6-5-4-3-2-1 scoring) | You want to maximize total team points, not just scorer count |

## How to run

```bash
cd coaches_analysis
python main.py     # document-only: prints a pointer to the reports and exits 0
```

There is nothing to compute — `main.py` exists only to satisfy the repo-wide `main.py`
convention and to point readers at the right report for their question.

## Layout

```
coaches_analysis/
├── main.py                                          # document-only stub
├── Nationals_Scoring_Focus_Men_Women.txt
├── Nationals_Scoring_Focus_Clearing_Roster.txt
└── Nationals_Points_Maximization_Clearing_Roster.txt
```

## Methods

All three reports are derived from the same underlying numbers — RQ1B 8th-place WA
thresholds, RQ1C nationals-margin head-to-head comparisons, and national clearing-pool
counts, relay-inclusive, outdoor 2024–2026 (March 1+) — but apply three different
decision frames (clear-rate, margin+spread, points-stacking) to the same data. See
[Summary_findings.md](Summary_findings.md) for the condensed recommendations and how
the three frames differ.

## Dependencies on other folders

- **`relays_findings/RQ1B_Nationals/`** and **`RQ1C_Competitiveness/`** — source
  8th-place thresholds and margin comparisons (relay-inclusive) that these reports
  synthesize.
- **`non_relays_findings/RQ1B_Nationals/`** and **`RQ1C_Competitiveness/`** — same, for
  the individual-events-only baseline referenced in the justification text.
- **`new_steeplechase_data/coaches_analysis/`** — the corrected-steeplechase mirror of
  this folder's distance/steeple recommendations (sprints/hurdles/jumps/throws
  recommendations are unchanged between the two).

## Related documents

- [Summary_findings.md](Summary_findings.md) — condensed recommendations by discipline and gender
