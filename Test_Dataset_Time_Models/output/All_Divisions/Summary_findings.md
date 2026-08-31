# All_Divisions External Validation Findings

**Filter:** none — all scraped men's documents (NCAA D1 / D3 / NAIA).
**Protocol:** season-PB only (chronological omitted).
**Tolerances (target event):** 100m 0.15s · 200m 0.30s · 400m 0.80s · 800m 1.50s · 1500m 3.0s · 3000m 6.0s · 3000m SC 8.0s · 5000m 10.0s.
**Models:** club-fit event-pair formulas, WA bands 750–950 / 800–1000 / 850–1050; **feature** route preferred.
**Run:** `python main.py all_divisions`
**Note:** `2026_Outdoor_Marks.pdf` is Air Force (pages 1–4 = men), not Colorado State.

## Dataset

| | |
|--|--|
| Result rows | **1,769** (1,212 primary dated + 557 supplementary season-PB) |
| Divisions | NAIA, NCAA D1, NCAA D3 |
| Schools | Air Force, DePaul, Indiana State, Keiser, Marquette, North Central, North Florida, Portland State, Providence, SIUE, Southern Indiana, Troy, UTEP, Utah State, Wisconsin-Oshkosh |
| Prediction rows (feature) | **1,100** |
| Athlete-seasons tested | **157** |

## Headline results (season-PB)

| Route | *n* | MedAPE | Within tolerance |
|-------|-----|--------|------------------|
| **feature** | 1100 | **1.69%** | **42.7%** |
| pooled | 1100 | 2.71% | 28.6% |

---

## Prediction success definition

For each athlete with season PBs in events A and B, and each applicable WA band:

1. Predict B from A using the band’s time model (feature route when available).
2. Compare predicted B to actual season PB in B.
3. **Successful prediction** if |pred − actual| ≤ starter tolerance for event B.

**Athlete success** for a pair: the athlete has **at least one** successful prediction for that pair (any band counts).

## Highest success rates by WA band (prediction-level)

### Band 750-950 (overall 45.8%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 40 | 36 | **90.0%** | 0.61% |
| 100m → 200m | 40 | 33 | **82.5%** | 0.78% |
| 800m → 1500m | 42 | 26 | **61.9%** | 0.85% |
| 400m → 200m | 28 | 16 | **57.1%** | 1.27% |
| 200m → 400m | 28 | 15 | **53.6%** | 1.34% |

### Band 800-1000 (overall 36.9%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 38 | 33 | **86.8%** | 0.74% |
| 100m → 200m | 38 | 26 | **68.4%** | 0.82% |
| 800m → 1500m | 37 | 18 | **48.6%** | 1.28% |
| 100m → 400m | 16 | 7 | **43.8%** | 2.51% |
| 5000m → 800m | 11 | 4 | **36.4%** | 2.24% |

### Band 850-1050 (overall 45.8%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 31 | 25 | **80.6%** | 0.66% |
| 100m → 200m | 31 | 24 | **77.4%** | 0.62% |
| 800m → 1500m | 30 | 17 | **56.7%** | 1.10% |
| 200m → 400m | 24 | 11 | **45.8%** | 1.83% |
| 5000m → 1500m | 34 | 13 | **38.2%** | 1.95% |

## Highest athlete success rates by event pair

| Pair | Athletes | Successful | Athlete success rate | Row success rate | MedAPE |
|------|----------|------------|----------------------|------------------|--------|
| 100m → 200m | 44 | 39 | **88.6%** | 76.1% | 0.77% |
| 200m → 100m | 44 | 39 | **88.6%** | 86.2% | 0.65% |
| 800m → 1500m | 45 | 33 | **73.3%** | 56.0% | 1.06% |
| 400m → 200m | 31 | 20 | **64.5%** | 39.2% | 1.83% |
| 200m → 400m | 31 | 18 | **58.1%** | 43.0% | 1.94% |
| 100m → 400m | 19 | 11 | **57.9%** | 37.1% | 2.50% |
| 400m → 100m | 19 | 11 | **57.9%** | 37.1% | 2.08% |
| 5000m → 1500m | 47 | 26 | **55.3%** | 35.2% | 1.90% |
| 5000m → 800m | 13 | 7 | **53.8%** | 33.3% | 1.96% |
| 800m → 5000m | 13 | 7 | **53.8%** | 33.3% | 2.14% |
| 1500m → 800m | 45 | 20 | **44.4%** | 29.4% | 1.92% |
| 1500m → 5000m | 47 | 20 | **42.6%** | 18.0% | 2.79% |

**Overall athlete-level (≥1 hit on any pair):** 132 / 157 = **84.1%** (feature), vs row-level within-tol **42.7%**.

## Artifacts (`output/All_Divisions/`)

- `men_test_dataset_results.csv`
- `men_athlete_season_features.csv`
- `men_supplementary_season_pb.csv`
- `README.txt`
- `model_validation/predictions.csv`
- `model_validation/summary_overall.csv`
- `model_validation/summary_by_band.csv`
- `model_validation/summary_by_pair.csv`
- `model_validation/validation_report.txt`
- `model_validation/success_rate_by_band_pair.csv`
- `model_validation/athlete_success_rate_by_pair.csv`
- `model_validation/research/` (inferential CSVs, report, plots)

