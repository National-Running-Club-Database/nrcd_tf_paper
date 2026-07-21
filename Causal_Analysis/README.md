# Causal Analysis — Competition Volume and Seasonal Improvement

Does racing more within a season cause (or at least associate with) a bigger World
Athletics (WA) score improvement for the *same* athlete? This module answers that with
within-athlete fixed-effects regressions, paired season deltas, and within-season
dose–response curves — a step up in rigor from the purely descriptive bin plots in
`number_of_events_question/`.

## What this folder does

`analyze_causal_competition_volume.py` runs four complementary models on the
athlete-season-event_group panel:

1. **Model A** — athlete×event-group fixed-effects (FE) regression of point jump on
   result count, controlling for opening-day WA and season, with a cluster bootstrap CI.
2. **Model B** — pooled OLS (no athlete FE) as a naive baseline for comparison.
3. **Model C** — paired within-athlete season-to-season deltas (Δ result count → Δ point jump).
4. **Model D** — Model A repeated separately for each gender × event-group slice.

It also summarizes **within-season dose–response**: average cumulative WA point jump
by chronological meet index, per gender × event group (plots saved to `plots/`).

`research_stats.py` runs at the end of `analyze_causal_competition_volume.py` (and can
also be run standalone) to add explicit hypothesis tests and cluster-bootstrap CIs on
top of the FE models: a paired Wilcoxon signed-rank test (higher- vs. lower-volume
season point jump, same athlete×event-group) and a Spearman rank correlation on the
within-athlete season-to-season deltas, plus a cluster-bootstrap CI band on the
dose-response curves. Outputs go to `research/` and `plots/`.

## How to run

```bash
cd causal_analysis
python main.py            # runs the causal analysis + research stats (default: all)
python main.py causal     # FE/OLS/delta models, chains research_stats.py at the end
python main.py research   # re-run only the inferential-stats layer on existing CSVs
python main.py all
```

The scripts have no CLI flags of their own — `main.py` invokes them via `subprocess`
using `sys.executable` so they always run with the interpreter that has the project's
scientific-Python stack (`numpy`, `scipy`, `matplotlib`) available. Set `MPLCONFIGDIR`
to a writable directory (e.g. `causal_analysis/.mplconfig`) if `~/.matplotlib` isn't
writable in your environment; both scripts default to this automatically.

## Layout

```
causal_analysis/
├── main.py                              # CLI entrypoint (this module)
├── analyze_causal_competition_volume.py # 4-model causal/associative analysis
├── research_stats.py                    # hypothesis tests + bootstrap CIs (research/)
├── causal_analysis_report.txt           # generated findings report
├── fe_regression_global.csv             # Model A coefficients
├── fe_regression_by_event_group.csv     # Model D coefficients (per group)
├── within_athlete_season_deltas.csv     # Model C paired deltas (+ raw per-season values)
├── within_season_dose_response.csv      # cumulative-jump-by-meet-index data
├── plots/
│   ├── dose_response_{men,women}_{sprints,distance,hurdles,jumps,throws}.png  # + 95% CI band
│   ├── fe_beta_forest_plot.png          # FE β(result_count) forest plot, all groups
│   └── delta_dose_response_scatter.png  # Δresult_count vs Δpoint_jump, Spearman ρ annotated
└── research/
    ├── inferential_report.txt           # hypothesis tests, CIs, causal-inference caveats
    ├── wilcoxon_volume_test_overall.csv / _by_group.csv
    ├── spearman_delta_test_overall.csv / _by_group.csv
    ├── dose_response_bootstrap_ci.csv
    └── fe_regression_by_event_group.csv # copy, for convenience alongside the other CSVs
```

## Methods summary

- **Panel construction**: athlete×event-group treated as the fixed-effect unit;
  seasons with only 1 result are excluded (mechanical point jump = 0 by definition).
- **Within transform**: values are demeaned within each panel unit before OLS, so
  coefficients reflect within-athlete variation only, not between-athlete differences.
- **Inference**: cluster bootstrap (resampling panel units) for the FE coefficient CI.
  `research_stats.py` adds a paired Wilcoxon signed-rank test and a Spearman
  correlation on the same within-athlete deltas, both with cluster-bootstrap CIs, as a
  distribution-free complement to the OLS-based FE coefficients.
- **Framing**: results are explicitly labeled *associative, not causal* — no
  randomized assignment of race volume exists in this dataset. See "Causal credibility
  assessment" in the generated report, the design notes at the top of
  `research/inferential_report.txt`, and the Limitations section of
  [Summary_findings.md](Summary_findings.md).

## Dependencies on other folders

- **`number_of_events_question/athlete_point_jumps_by_season.csv`** — the input panel
  (athlete × season × event_group point-jump and result-count rows). This file is
  produced by `number_of_events_question/analyze_point_jump_by_competition_count.py`,
  which must be run first if the CSV is missing.
- **`non_relays_findings/Sprints_Events_Counting/.pylibs/`** — vendored `numpy`/`matplotlib`
  wheels are added to `sys.path` as a fallback if those packages aren't already
  importable in the active environment.

## Related documents

- [Summary_findings.md](Summary_findings.md) — key coefficients, dose-response results, and limitations
- `causal_analysis_report.txt` — full generated text report (FE/OLS/delta models)
- `research/inferential_report.txt` — Wilcoxon + Spearman hypothesis tests, bootstrap CIs
