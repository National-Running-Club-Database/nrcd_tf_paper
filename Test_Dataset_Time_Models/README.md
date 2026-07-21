# Men's Outdoor Time-Model Test Dataset

External validation of club-fit outdoor **event-pair time models** on dated NCAA Division I men's results.

## What this folder does

1. **Preprocess** — scrape dated outdoor marks from PDFs, score with World Athletics 2025 outdoor tables, engineer routing features (`bal_spec`, `best_event`, chronology).
2. **Validate** — apply steeplechase-band pair models (pooled + feature-routed) under season-PB and chronological protocols.
3. **Research stats** — bootstrap CIs, Wilcoxon bias / paired tests, Spearman correlation, Clopper–Pearson tolerance CIs, and publication figures.

## Quick start

```bash
# from repo root, with the project venv
cd test_dataset_time_models
../.venv/bin/python main.py              # preprocess → validate → research
../.venv/bin/python main.py preprocess
../.venv/bin/python main.py validate     # includes research stats
../.venv/bin/python main.py research     # stats/plots only (needs predictions.csv)
```

## Layout

```
test_dataset_time_models/
├── main.py                 # CLI entrypoint
├── preprocess.py           # scrape + feature engineering
├── validate.py             # model application + descriptive summaries
├── research_stats.py       # inferential tests + figures
├── README.md
├── Summary_findings.md
├── documents/
│   ├── men/                # source PDFs/HTML (dated PDFs are scraped)
│   └── women/              # reserved
├── models/steeplechase/    # band model report .txt files
├── wa_scoring/             # WA 2025 outdoor coefficients
└── output/
    ├── men_test_dataset_results.csv
    ├── men_athlete_season_features.csv
    ├── README.txt
    └── model_validation/
        ├── predictions.csv
        ├── summary_*.csv
        ├── validation_report.txt
        └── research/
            ├── inferential_*.csv
            ├── feature_vs_pooled_tests.csv
            ├── inferential_report.txt
            └── plots/
```

## Data inclusion rules

| Role | Sources | Validation use |
|------|---------|----------------|
| **Primary (dated)** | SIUE PDF, North Florida PDF, DePaul PDF (men pages) | season-PB **and** chronological |
| **Supplementary (undated season bests)** | USI PDF, North Central / Oshkosh / Keiser HTML matrices | season-PB only |

Season-PB is the stronger external protocol for these club-fit models, so undated season lists are retained as `Source_Role=supplementary_season_pb` rather than discarded. Chronological evaluation filters to `primary_dated` rows with real `Competition_Date` values.

Women's pages inside the DePaul PDF are skipped in the men's run.

## Methods (research design)

### Population and shift

- **Training domain:** club / National Running Club outdoor models (WA point bands 750–950, 800–1000, 850–1050).
- **Test domain:** NCAA D1 men (SIUE, North Florida, DePaul) — intentional **distribution shift** check, not an i.i.d. holdout.

### Evaluation protocols

| Mode | Definition |
|------|------------|
| `season_pb` | Predict target-event season best from source-event season best (matches training setup). |
| `chronological` | Earliest dated source mark that precedes a later target mark (early → later season transfer). |

### Routes

- **pooled** — band-level formula from the model report.
- **feature** — recommended cohort formula (`bal_spec`, `best_event`, etc.) when the athlete's label exists; otherwise falls back to pooled.

### Primary metrics

Absolute seconds are **not commensurate** across 100m–5000m. Cross-event reporting uses:

- **MedAPE** — median absolute percent error (primary).
- **MedAE** — median absolute error in seconds (within event pair / event group only).
- **Within-tolerance rate** — starter absolute-second bands (see preprocess tolerances).
- **Spearman ρ** — rank correlation of predicted vs actual.
- **Signed bias** — pred − actual; tested with Wilcoxon signed-rank (two-sided α = 0.05).

### Inferential procedures (`research_stats.py`)

| Procedure | Purpose |
|-----------|---------|
| Percentile bootstrap (B = 2000, seed = 2026) | 95% CI for MedAE / MedAPE / MAE |
| Wilcoxon signed-rank | Test nonzero median signed bias |
| Paired Wilcoxon | Feature vs pooled on \|error\| and \|%\| error |
| Spearman ρ + p-value | Pred–actual association |
| Clopper–Pearson | Exact 95% CI for tolerance hit rate |

### Figures

Saved under `output/model_validation/research/plots/`:

- `pred_vs_actual_by_group.png` — calibration by event group  
- `bland_altman_pct.png` — agreement on percent scale  
- `medape_by_pair.png` — pair-level MedAPE with bootstrap CIs  
- `feature_vs_pooled_pct.png` — paired \|%\| error comparison  
- `medape_by_band.png` / `medape_by_mode.png` — band and protocol contrasts  

## Dependencies

Project venv packages used here: `pandas`, `numpy`, `scipy`, `matplotlib`, `pdfplumber`.

## Related documents

- [Summary_findings.md](Summary_findings.md) — results narrative and limitations  
- `output/model_validation/validation_report.txt` — descriptive validation dump  
- `output/model_validation/research/inferential_report.txt` — inferential summary  
