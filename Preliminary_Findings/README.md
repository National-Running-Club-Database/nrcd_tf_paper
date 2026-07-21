# preliminary_findings

**Document-only, early-stage snapshot module.** This folder holds an earlier round of
exploratory outputs (plots and text reports) generated before the corresponding
analyses were split out into their own dedicated, actively-maintained modules. It is
kept for historical reference and to preserve the original figures/thresholds as they
existed at that stage of the project. **For current, maintained versions of this
content, use the successor modules listed below** — do not treat this folder as the
source of truth.

## What this folder contains and where it was superseded

| Subfolder | Content | Superseded by |
|-----------|---------|----------------|
| `NIRCA_Nationals_Strategy/` | 3-year average nationals 8th-place WA rankings, all event groups combined (relay-inclusive; steeplechase from `new_steeplechase_data`) | `../relays_findings/RQ1B_Nationals/`, `../new_steeplechase_data/RQ1B_Nationals/` |
| `Number_Of_Events/` | Early point-jump-by-competition-volume plots/reports and an early causal-analysis report | `../number_of_events_question/`, `../causal_analysis/` |
| `Pairwise_Events/` | Indoor and outdoor pairwise best-event histograms by discipline/gender | `../non_relays_findings/*_Events_Counting/`, `../indoor_analysis/RQ1A_Best_Event/` |
| `Indoor_Outdoor_Interplay/` | Early first-event-of-season (indoor vs outdoor) recommendations | `../indoor_analysis/Indoor_Outdoor_Interplay/` |
| `Time_Models/` | Early indoor/outdoor point-band feature-importance reports | `../time_models/`, `../indoor_analysis/Feature_Importance_Point_Band_Time_Models/` |

## How to run

```bash
cd preliminary_findings
python main.py     # document-only: prints a pointer to the successor modules and exits 0
```

There are no `.py` analysis scripts in this folder — `main.py` exists only to satisfy
the repo-wide convention and to redirect readers to the maintained modules.

## Layout

```
preliminary_findings/
├── main.py                            # document-only stub
├── NIRCA_Nationals_Strategy/
│   └── rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt
├── Number_Of_Events/
│   ├── Averaged_Out_Plots/
│   ├── best_event_proportion_report*.txt
│   ├── causal_analysis_report.txt
│   └── doubling_wa_buffer_report.txt
├── Pairwise_Events/
│   ├── Indoor_Pairwise/
│   └── Outdoor_Pairwise/
└── Time_Models/
    ├── Indoor_Track/
    └── Outdoor_Track/
```

## Dependencies on other folders

None — this folder is a static, self-contained snapshot. It does not read from or feed
into any other module's pipeline. All meaningful downstream dependencies run the other
direction: the successor modules listed above regenerate and supersede this content.

## Related documents

- [Summary_findings.md](Summary_findings.md) — what changed between this snapshot and the current modules
