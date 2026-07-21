# relays_findings

The relay-inclusive counterpart to `../non_relays_findings/`: RQ1A/B/C recomputed with
relay results included, where each relay leg athlete is credited the **full team World
Athletics (WA) score** for that relay appearance. This is the project's RQ4 sensitivity
analysis — it asks how much the RQ1 conclusions change once relays are counted.

## What this folder does

### Discipline-level "best event" counting (per-discipline subfolders)

`Distance_Relays_Findings/`, `Sprinters_Relays_Findings/`, `Hurdles_Relays_Findings/`,
`Jumps_Relays_Findings/`, `Throws_Relays_Findings/` each hold an
`analyze_best_event_relays_*.py` script (built on the shared `relay_best_event_analysis.py`
library) that recomputes best-event counts and pairwise comparisons **including** the
4x100m/4x400m relays alongside that discipline's individual events.

### Top-level cross-discipline analyses (wired into `main.py`)

1. **`analyze_rq1b_nationals_relays.py`** — nationals top-8 WA distributions and 3-year
   average 8th-place thresholds, individual events **plus** 4x100m/4x400m. Outputs to
   `RQ1B_Nationals/`.
2. **`analyze_rq1c_competitiveness_relays.py`** — most-competitive-event recommendation
   (RQ1C) using the relay-inclusive threshold set, including new individual×relay
   head-to-head margin comparisons. Outputs to `RQ1C_Competitiveness/`.
3. **`build_pecking_orders.py`** — for every discipline, ranks events (individual-only
   and individual+relay) by head-to-head pairwise best-event wins, using a Copeland
   score to break cyclic/incomplete comparisons. Outputs `event_pecking_orders.txt`
   (all comparisons) and `event_pecking_orders_min50.txt` (≥50 dual-event athletes only).

### Shared library modules (not run directly)

`relay_best_event_analysis.py` and `relay_rq1_data.py` are imported by the discipline
scripts and the top-level RQ1B/RQ1C scripts — they contain no CLI entrypoint of their own.

## How to run

```bash
cd relays_findings
python main.py                 # all three top-level analyses
python main.py rq1b_nationals
python main.py rq1c
python main.py pecking_orders
python main.py all
```

To regenerate one discipline's relay-inclusive best-event counts, run its script
directly, e.g. `python Sprinters_Relays_Findings/analyze_best_event_relays_sprinters.py`.

## Layout

```
relays_findings/
├── main.py
├── analyze_rq1b_nationals_relays.py
├── analyze_rq1c_competitiveness_relays.py
├── build_pecking_orders.py
├── relay_best_event_analysis.py         # shared library (best-event/pairwise logic)
├── relay_rq1_data.py                    # shared library (data loading, event maps)
├── relay_analysis_trends_summary.txt    # narrative synthesis of relay-only trends
├── relay_inclusion_rq_differences.txt   # with-vs-without-relays comparison (RQ4)
├── event_pecking_orders.txt / _min50.txt
├── RQ1B_Nationals/                      # nationals top-8 distributions + thresholds
├── RQ1C_Competitiveness/                # most-competitive-event recommendations
├── Distance_Relays_Findings/            # per-discipline scripts, data, plots
├── Sprinters_Relays_Findings/
├── Hurdles_Relays_Findings/
├── Jumps_Relays_Findings/
└── Throws_Relays_Findings/
```

## Methods

Same RQ1A/B/C definitions as `non_relays_findings/`, with the event pool expanded to
include 4x100m and 4x400m relays (the only relay types with nonzero WA scoring in this
dataset — 4x200m, 4x800m, DMR, SMR, and Swedish Relay are excluded from threshold-based
comparisons for lack of WA scoring). Relay team WA is credited to every leg athlete;
an athlete's relay-event personal best is the max WA across their relay appearances.

**Pecking orders** (`build_pecking_orders.py`): for each qualifying event pair, count
whose personal-best WA is higher; the event with more wins ranks higher. Cyclic or
incomplete pairwise results are broken by Copeland score (total head-to-head wins).

## Dependencies on other folders

- **`non_relays_findings/Distance_Events_Counting/running_event.csv`** — event ID → name
  lookup, read by `relay_rq1_data.py`.
- **`non_relays_findings/Sprints_Events_Counting/.pylibs/`** — vendored `matplotlib`
  wheels used as a `sys.path` fallback by `relay_best_event_analysis.py`.
- **`non_relays_findings/RQ1B_Nationals/`** and **`RQ1C_Competitiveness/`** — the
  individual-events-only baseline that `relay_inclusion_rq_differences.txt` compares against.
- Downstream: **`number_of_events_question/`** combines this folder's per-discipline
  relay CSVs (`*_relays_findings/*.csv`) into its deduplicated point-jump dataset.

## Related documents

- [Summary_findings.md](Summary_findings.md) — condensed key results and limitations
- `relay_analysis_trends_summary.txt` — full relay-only narrative (10 sections)
- `relay_inclusion_rq_differences.txt` — full with-vs-without-relays comparison (RQ4)
