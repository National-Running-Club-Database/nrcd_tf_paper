# time_models

Cross-event time-prediction models for outdoor track: given an athlete's personal best
in one event, predict their likely time in another event (and, by extension, their World
Athletics point total). This is the largest and most heavily subdivided module in the
repo — it explores many candidate modeling strategies (pooled linear, point-banded,
cohort/feature-routed, specialization-stratified, race-count-stratified) rather than
shipping a single final model.

## What this folder does

### Root-level build scripts (wired into `main.py`)

1. **`build_cross_event_time_models.py`** — the baseline: pooled linear OLS
   `to_event = a + b × from_event` for every ordered Sprints/Distance event pair
   (steeplechase excluded). Outputs `cross_event_time_models.csv` /
   `cross_event_time_models_report.txt`. This is the baseline every other subfolder
   compares against.
2. **`generate_higher_races_formula_reports.py`** — turns `model_search/race_count/`
   CSVs into human-readable formula reports for the 5-race, 5-or-6-race, and
   "higher race count" cohorts (`time_models_*_formulas.txt`).
3. **`research_stats.py`** — inferential layer on feature-importance CV tables
   (`research/inferential_report.txt`, bootstrap CIs, Wilcoxon/binomial tests, plots).

### Subfolder pipelines (self-contained; run their own scripts directly)

| Subfolder | Question | Primary script(s) |
|-----------|----------|--------------------|
| `Point_Bands_Time_Models/` | Does banding athletes by WA score range improve accuracy? | `analyze_point_bands_time_models.py`, `generate_point_band_formulas.py` |
| `Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/` | Do cohort features (bal/spec, best_event, etc.) beat the pooled band model? | `analyze_feature_importance_point_bands.py`, `generate_new_feature_important_time_models.py` |
| `Point_Bands_Time_Models/Balanced_Specialization_Counts/` | How many athletes are "balanced" vs "specialized" per band? | `count_balanced_specialization_by_band.py` |
| `specialized_time_models/` | Does event-group specialization (WA spread) help or hurt prediction? | `analyze_specialization_time_models.py`, `build_specialization_time_models.py` |
| `Short_Long_Specialization_Time_Models/` | Short-sprint vs long-sprint / mid- vs long-distance stratification | `analyze_short_long_specialization.py` |
| `Combined_Specialization_Time_Models/` | Specialization × another factor combined | `analyze_combined_specialization.py` |
| `six_to_ten_races_time_models/` | Do 6-10-race-cohort models differ from the 5-6-race baseline? | `analyze_6_to_10_races_time_models.py`, `build_bal_spec_6_to_10_models.py` |
| `new_factors/` | Do new engineered factors (doubling, combined bal×best-event) help? | `analyze_new_factors.py` |
| `new_factors/doubling_analysis/` | Does doubling (2+ events per meet) change scores or predictability? | `analyze_doubling.py`, `analyze_doubling_wa_buffer.py` |
| `new_factors/best_event_bal_spec_6races/` | Combined best-event × bal/spec labels, 6-race cohort | `analyze_combined_factors_6races.py` |
| `model_search/` | Which functional form (linear/quadratic/ratio/kNN/multivariate) wins per pair? | `compare_time_model_candidates.py`, `compare_time_models_by_race_count.py` |

Each subfolder writes its own `*_findings.txt` / `*_report.txt` and (where applicable)
`time_models_*_formulas.txt` with ready-to-use formulas.

## How to run

```bash
cd time_models
python main.py                     # both root-level build scripts
python main.py cross_event
python main.py higher_races_report
python main.py all

# subfolder pipelines (run directly), e.g.:
python Point_Bands_Time_Models/analyze_point_bands_time_models.py
python specialized_time_models/analyze_specialization_time_models.py
python new_factors/doubling_analysis/analyze_doubling.py
python six_to_ten_races_time_models/analyze_6_to_10_races_time_models.py
```

## Layout

```
time_models/
├── main.py
├── build_cross_event_time_models.py
├── generate_higher_races_formula_reports.py
├── cross_event_time_models.csv / _report.txt
├── time_models_5_races_formulas.txt
├── time_models_5_or_6_races_formulas.txt
├── time_models_higher_races_formulas.txt
├── Point_Bands_Time_Models/
│   ├── Feature_Importance_Point_Band_Time_Models/
│   └── Balanced_Specialization_Counts/
├── specialized_time_models/
├── Short_Long_Specialization_Time_Models/
├── Combined_Specialization_Time_Models/
├── six_to_ten_races_time_models/        # renamed from "6-10 races_time_models"
├── new_factors/
│   ├── doubling_analysis/
│   └── best_event_bal_spec_6races/
└── model_search/
    └── race_count/
```

> **Note:** `six_to_ten_races_time_models/` was renamed from `6-10 races_time_models`
> (the space and hyphen in the old name caused shell-quoting friction). All internal
> "Source:" text labels inside that folder's generated reports were updated to match;
> no other folder references the old path.

## Methods (cross-cutting)

- **Population**: athlete-season personal bests (fastest valid mark per athlete per
  event, WA > 0), outdoor 2024–2026, March 1+, relay-inclusive CSVs.
- **Baseline model**: `to_event_seconds = intercept + slope × from_event_seconds`
  (pooled linear OLS), fit separately per ordered event pair (A→B ≠ B→A).
- **Evaluation**: 5-fold cross-validation, median absolute error (seconds) as the
  primary metric, RMSE as secondary; seed = 42 throughout.
- **Point banding**: restrict the population to a WA point range (e.g., 750–950) before
  fitting, on the hypothesis that athletes of similar ability transfer between events
  more predictably than the full population.
- **Cohort/feature routing**: within a band, route athletes to a cohort-specific formula
  based on a label (balanced/specialized, best-event, which event they're coming
  *from*, etc.) instead of one pooled formula for the whole band.
- **Alternative functional forms** (`model_search/`): median ratio, quadratic,
  log-linear, k-NN median, multivariate OLS, binned ratio, robust trimmed linear,
  chain-residual-adjusted — compared against the linear baseline per pair.

## Dependencies on other folders

- Uses relay-inclusive discipline CSVs from `../relays_findings/` (via each subfolder's
  own data-loading code) and, where noted in individual scripts, steeplechase-corrected
  distance data from `../new_steeplechase_data/`.
- `indoor_analysis/Point_Bands_Time_Models/` and
  `indoor_analysis/Feature_Importance_Point_Band_Time_Models/` mirror this folder's
  point-band methodology for indoor track.
- `../test_dataset_time_models/` externally validates a subset of these formulas
  (steeplechase-band pair models) against dated NCAA D1 results — see that folder's
  `Summary_findings.md` for out-of-sample accuracy.
- `../coaches_analysis/` and `../new_steeplechase_data/coaches_analysis/` translate
  some of these formulas into roster-planning recommendations.

## Related documents

- [Summary_findings.md](Summary_findings.md) — condensed key results across all subfolders
- `Point_Bands_Time_Models/point_band_combined_report.txt` — full point-band leaderboard
- `Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/feature_importance_combined_report.txt` — full cohort-routing comparison
- `model_search/model_search_report.txt` — full functional-form comparison
