# D1_Only External Validation Findings

**Filter:** `College_Division == NCAA D1` only (men).  
**Protocol:** season-PB only (chronological evaluation omitted).  
**Tolerances (target event):** 100m 0.15s · 200m 0.30s · 400m 0.80s · 800m 1.50s · 1500m 3.0s · 3000m 6.0s · 3000m SC 8.0s · 5000m 10.0s.  
**Models:** club-fit event-pair formulas, WA bands 750–950 / 800–1000 / 850–1050; **feature** route preferred (pooled reported for comparison).  
**Run:** `python main.py d1_only`  
**Note:** `2026_Outdoor_Marks.pdf` is Air Force (pages 1–4 = men), not Colorado State.

## Dataset

| | |
|--|--|
| Result rows | **1,322** (1,212 primary dated + 110 supplementary season-PB) |
| Schools | Air Force, DePaul, Indiana State, Marquette, North Florida, Portland State, Providence, SIUE, Southern Indiana, Troy, UTEP, Utah State |
| Prediction rows (feature) | **606** |
| Athlete-seasons tested | **88** |

Montana top-5 PDF was present but its multi-column layout yielded 0 rows.

## Headline results (season-PB)

| Route | *n* | MedAPE [95% CI] | Within tolerance [95% CI] |
|-------|-----|-----------------|---------------------------|
| **feature** | 606 | **2.05%** [1.86, 2.26] | **37.3%** [33.4%, 41.3%] |
| pooled | 606 | 3.13% [2.86, 3.48] | 24.9% [21.5%, 28.6%] |

Feature routing beats pooled (paired Wilcoxon on \|%\| error, *p* ≪ 0.001). Positive bias overall (predictions tend slower than actual), especially into 5000m / 3000m SC.

---

## Prediction success definition

For each athlete with season PBs in events A and B, and each applicable WA band:

1. Predict B from A using the band’s time model (feature route when available).
2. Compare predicted B to actual season PB in B.
3. **Successful prediction** if \|pred − actual\| ≤ starter tolerance for event B.

**Athlete success** for a pair: the athlete has **at least one** successful prediction for that pair (any band counts).

---

## Highest success rates by WA band (prediction-level)

Success rate = successful predictions / predictions for that band×pair. Ranked within each band (*n* ≥ 5).

### Band 750–950 (overall 43.0%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| **100m → 200m** | 16 | 14 | **87.5%** | 0.80% |
| **200m → 100m** | 16 | 14 | **87.5%** | 0.61% |
| 3000m SC → 5000m | 8 | 5 | 62.5% | 0.90% |
| 200m → 400m | 17 | 10 | 58.8% | 1.21% |
| 800m → 1500m | 21 | 12 | 57.1% | 0.90% |

### Band 800–1000 (overall 30.2%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| **200m → 100m** | 16 | 14 | **87.5%** | 0.88% |
| **100m → 200m** | 16 | 12 | **75.0%** | 0.93% |
| 100m → 400m | 7 | 4 | 57.1% | 0.92% |
| 400m → 100m | 7 | 3 | 42.9% | 1.80% |
| 800m → 1500m | 21 | 7 | 33.3% | 2.10% |

### Band 850–1050 (overall 38.2%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| **200m → 100m** | 15 | 12 | **80.0%** | 0.72% |
| **100m → 200m** | 15 | 11 | **73.3%** | 0.75% |
| 800m → 1500m | 20 | 9 | 45.0% | 1.28% |
| 200m → 400m | 18 | 8 | 44.4% | 2.10% |
| 5000m → 1500m | 23 | 7 | 30.4% | 2.67% |

**Best pair per band (*n* ≥ 10):** 100↔200 in every band (87.5% / 87.5% / 80.0%).

---

## Highest athlete success rates by event pair

Athlete success rate = athletes with ≥1 successful prediction for the pair ÷ athletes with ≥1 prediction for the pair (bands pooled).

| Pair | Athletes | Successful | Athlete success rate | Row success rate | MedAPE |
|------|----------|------------|----------------------|------------------|--------|
| **100m → 200m** | 17 | 16 | **94.1%** | 78.7% | 0.84% |
| **200m → 100m** | 17 | 15 | **88.2%** | 85.1% | 0.72% |
| 800m → 5000m | 4 | 3 | 75.0% *(small n)* | 37.5% | 4.04% |
| 100m → 400m | 9 | 6 | 66.7% | 50.0% | 1.48% |
| **800m → 1500m** | 24 | 15 | **62.5%** | 45.2% | 1.33% |
| 3000m SC → 5000m | 8 | 5 | 62.5% | 62.5% | 0.90% |
| **200m → 400m** | 20 | 12 | **60.0%** | 41.5% | 2.29% |
| 400m → 100m | 9 | 5 | 55.6% | 43.8% | 1.66% |
| 400m → 200m | 20 | 10 | 50.0% | 26.4% | 2.03% |
| 5000m → 1500m | 26 | 12 | 46.2% | 27.4% | 2.31% |
| 1500m → 5000m | 26 | 11 | 42.3% | 15.1% | 3.24% |
| 1500m → 800m | 24 | 10 | 41.7% | 25.8% | 2.03% |
| 3000m SC → 1500m | 17 | 7 | 41.2% | 27.3% | 2.43% |
| 1500m → 3000m SC | 17 | 3 | 17.6% | 12.1% | 5.33% |
| 5000m → 3000m SC | 8 | 1 | 12.5% | 12.5% | 4.82% |

Among pairs with **≥10 athletes**, leaders are **100↔200**, then **800→1500** and **200→400**. Weakest transfers are **into** steeple.

**Overall athlete-level (≥1 hit on any pair):** 69 / 88 = **78.4%** (feature), vs row-level within-tol **37.3%**.

---

## Interpretation

- Adjacent same-family pairs (especially **100↔200**) transfer best from club-fit models onto D1 men.
- Mid-distance adjacent (**800→1500**) is the strongest distance-family pair on athlete success.
- Far pairs and flat→steeple remain poorly calibrated; SC as *source* (SC→5000) is much better than SC as *target*.
- Athlete success rates exceed row rates when athletes appear in multiple bands (one hit counts as athlete success).

## Artifacts (`output/`)

- `D1_Only_men_test_dataset_results.csv`
- `D1_Only_men_athlete_season_features.csv`
- `D1_Only_men_supplementary_season_pb.csv`
- `D1_Only_README.txt`
- `model_validation/D1_Only_predictions.csv`
- `model_validation/D1_Only_summary_overall.csv`
- `model_validation/D1_Only_summary_by_band.csv`
- `model_validation/D1_Only_summary_by_pair.csv`
- `model_validation/D1_Only_validation_report.txt`
- `model_validation/D1_Only_success_rate_by_band_pair.csv`
- `model_validation/D1_Only_athlete_success_rate_by_pair.csv`
- `model_validation/D1_Only_research/D1_Only_inferential_*.csv|txt` + plots
