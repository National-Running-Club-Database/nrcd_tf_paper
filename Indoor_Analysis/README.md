# indoor_analysis

Indoor-track counterpart to the outdoor RQ1/point-band modules. Covers event
specialization (RQ1A only — there is no indoor Nationals, so RQ1B/RQ1C do not apply),
indoor point-band time models, and the indoor→outdoor season-opener transition.

## What this folder does

### Top-level analyses (wired into `main.py`)

1. **`analyze_indoor_rq1a.py`** — best-event counts and pairwise comparisons for
   indoor Sprints (60m/200m/400m), Distance (800m/Mile/3000m/5000m), Hurdles
   (55m/60m Hurdles), Jumps (LJ/TJ/HJ), and Throws (Shot Put/Discus/Javelin).
   Outputs per-discipline folders under `RQ1A_Best_Event/`.
2. **`analyze_indoor_point_bands.py`** — searches WA point-band widths/placements for
   indoor time-equivalency models (Sprints 60/200/400, Distance 800/Mile/3000; relays
   excluded). Outputs to `Point_Bands_Time_Models/`.

### Subfolder analyses (run individually — not wired into `main.py`)

- **`Indoor_Outdoor_Interplay/analyze_indoor_outdoor_opener_wa.py`** — compares indoor
  vs. outdoor season-opener WA scores for athletes who compete in both seasons.
- **`Feature_Importance_Point_Band_Time_Models/analyze_indoor_feature_importance_point_bands.py`**
  — cohort-formula feature importance for indoor point bands (mirrors the outdoor
  version in `../time_models/Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/`).

## How to run

```bash
cd indoor_analysis
python main.py            # both top-level analyses
python main.py rq1a
python main.py point_bands
python main.py all

# subfolder scripts (run directly):
python Indoor_Outdoor_Interplay/analyze_indoor_outdoor_opener_wa.py
python Feature_Importance_Point_Band_Time_Models/analyze_indoor_feature_importance_point_bands.py
```

## Layout

```
indoor_analysis/
├── main.py
├── analyze_indoor_rq1a.py
├── analyze_indoor_point_bands.py
├── RQ1A_Best_Event/
│   ├── rq1a_indoor_summary.txt
│   └── {Distance,Sprints,Hurdles,Jumps,Throws}/   # best-event + pairwise counts/histograms
├── Point_Bands_Time_Models/
│   ├── point_band_findings_width_200.txt
│   ├── point_band_report_width_200.txt
│   └── point_band_{cohort,summary,pair_results}_width_200.csv
├── Feature_Importance_Point_Band_Time_Models/     # cohort-formula feature importance (indoor)
├── Indoor_Outdoor_Interplay/                      # season-opener WA comparison
├── Indoor_Distance/, Indoor_Sprints/, Indoor_Hurdles/,
│   Indoor_Jumps/, Indoor_Throws/                  # source data CSVs (2024-2026)
```

## Methods

- **RQ1A**: same method as outdoor RQ1A — overall best event + pairwise dual-event
  comparisons — but using all dates present in the indoor files (no outdoor March-1
  filter, since indoor season timing differs). Relay team WA is credited to every leg
  athlete; specialization reads should prefer the `*_2plus_individual.txt` /
  `pairwise_*_individual.txt` outputs, since "all athletes" counts are relay-inflated
  for Hurdles/Jumps/Throws.
- **Point bands**: WA point bands of width 200, tested at 50-point starting increments;
  ranked by mean cross-validated (CV) time-prediction error in seconds, among pairs with
  sufficient sample size.
- **Indoor↔Outdoor opener comparison**: for athletes with ≥2 indoor and ≥2 outdoor meets
  in the same year, compares WA score at the season-opener meet for "similar" event pairs
  (e.g., 60m→100m, Mile→1500m), restricted to outdoor-opener WA ∈ [750, 950).

## Dependencies on other folders

- **`new_steeplechase_data/`** — the indoor↔outdoor opener comparison uses corrected
  outdoor distance/steeplechase WA scores from this folder when available.
- **`time_models/Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/`** —
  the outdoor analogue of this folder's indoor feature-importance analysis; useful for
  side-by-side indoor/outdoor comparison.

## Related documents

- [Summary_findings.md](Summary_findings.md) — key numbers and limitations
- `RQ1A_Best_Event/rq1a_indoor_summary.txt` — full RQ1A narrative
- `Point_Bands_Time_Models/point_band_findings_width_200.txt` — full point-band leaderboard
- `Indoor_Outdoor_Interplay/indoor_outdoor_opener_wa_findings_band_750_950.txt` — full opener-comparison report
