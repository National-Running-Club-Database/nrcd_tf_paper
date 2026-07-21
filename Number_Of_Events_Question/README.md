# Number Of Events Question

Descriptive (non-causal) analyses of the research question: **does competing in more
races during a season associate with a bigger World Athletics (WA) score improvement,
and when in the season do athletes actually hit their season best?** The causal,
fixed-effects follow-up to this module lives in `../causal_analysis/`.

## What this folder does

1. **`analyze_point_jump_by_competition_count.py`** — builds a relay-inclusive,
   deduplicated combined results dataset, computes each athlete-season-event_group's
   *point jump* (season-max WA − first-day WA), and bins athletes by result count to
   show the average point jump per competition-volume bin (plots + per-season-averaged
   plots + text findings).
2. **`analyze_best_event_proportion.py`** — for each athlete-season-event_group,
   computes `best_meet_index / total_results_in_season` (1.0 = peaked at the very last
   meet, 0.5 = peaked at the midpoint) and reports the distribution overall, by season,
   and by number of meets in the season.
3. **`research_stats.py`** — inferential layer on top of both analyses: bootstrap 95%
   CIs for mean/median point jump by result-count bin, and Spearman rank correlations
   (result count vs. point jump, and result count vs. best-event proportion) with
   p-values and bootstrap CIs on ρ. Runs automatically at the end of
   `analyze_point_jump_by_competition_count.py`, or standalone via `python main.py
   research` (reads the existing CSVs, no re-scrape needed). Outputs go to `research/`.

## How to run

```bash
cd number_of_events_question
python main.py                # both analyses (chains research_stats.py)
python main.py point_jump     # chains research_stats.py at the end
python main.py best_event
python main.py research       # re-run only the inferential-stats layer on existing CSVs
python main.py all
```

Scripts take no CLI flags of their own; `main.py` runs them via `subprocess` with
`sys.executable`. Set `MPLCONFIGDIR` to a writable directory (e.g.
`number_of_events_question/.mplconfig`) if `~/.matplotlib` isn't writable in your
environment; both plotting scripts default to this automatically.

## Layout

```
number_of_events_question/
├── main.py
├── analyze_point_jump_by_competition_count.py
├── analyze_best_event_proportion.py
├── research_stats.py                         # bootstrap CIs + Spearman tests (research/)
├── combined_relay_dataset_deduped.csv        # deduplicated results (relay + individual)
├── athlete_point_jumps_by_season.csv         # athlete-season-event_group panel (consumed by causal_analysis)
├── athlete_best_event_proportion.csv
├── avg_point_jump_by_result_count.csv
├── avg_point_jump_by_result_count_averaged.csv
├── point_jump_findings.txt
├── best_event_proportion_report.txt
├── best_event_proportion_report_by_season.txt
├── plots/                          # point_jump_{gender}_{group}_{season}.png
├── Averaged_Out_Plots/             # point_jump_{gender}_{group}_averaged_2024_2026.png
└── research/
    ├── inferential_report.txt      # bootstrap CIs, Spearman tests, design caveats
    ├── point_jump_bootstrap_ci_by_bin.csv
    ├── spearman_point_jump_by_group.csv
    ├── spearman_best_event_proportion_by_group.csv
    └── plots/                      # point_jump_ci_{gender}_{group}.png, spearman_rho_*.png, +overall scatter
```

## Methods

- **Combined dataset**: all relay-inclusive discipline CSVs from `relays_findings/` (and
  event names from `non_relays_findings/Distance_Events_Counting/running_event.csv`),
  deduplicated by `result_id`. Relay leg athletes each receive the team WA score for
  that relay appearance.
- **Season window**: outdoor year file (2024/2025/2026), results on/after March 1.
- **Point jump**: season-max WA minus first-day WA (max WA if multiple events on the
  first competition date). Athletes with exactly 1 result have point jump = 0 by
  definition — this is a structural floor, not a finding.
- **Plot bins**: only result-count bins with ≥10 athletes are plotted, to avoid noisy
  small-n bins.
- **Best-event proportion**: results ordered chronologically (date, meet, result_id);
  proportion = index of first meet reaching the season max ÷ total meets. Athletes with
  1 result always have proportion 1.0 by definition (Section B of the report excludes
  1-result seasons for a cleaner championship-timing read).

## Dependencies on other folders

- **`relays_findings/{Distance,Hurdles,Jumps,Sprinters,Throws}_relays_findings/*.csv`** —
  source per-discipline relay result files, combined and deduplicated here.
- **`non_relays_findings/Distance_Events_Counting/running_event.csv`** — event ID → name
  lookup table.
- **`non_relays_findings/Sprints_Events_Counting/.pylibs/`** — vendored `matplotlib`
  wheels used as a `sys.path` fallback.
- Downstream: **`causal_analysis/analyze_causal_competition_volume.py`** consumes
  `athlete_point_jumps_by_season.csv` produced here.

## Related documents

- [Summary_findings.md](Summary_findings.md) — key numbers and limitations
- `point_jump_findings.txt`, `best_event_proportion_report.txt`,
  `best_event_proportion_report_by_season.txt` — full generated text reports
- `research/inferential_report.txt` — bootstrap CIs and Spearman hypothesis tests
