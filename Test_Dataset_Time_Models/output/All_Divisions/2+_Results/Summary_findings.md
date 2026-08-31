# All_Divisions · 2+_Results External Validation Findings

**Filter:** ≥2 results in **both** from-event (A) and to-event (B) within the athlete-season before predicting A→B. Same All_Divisions men's scrape (NCAA D1 / D3 / NAIA).
**Protocol:** season-PB only (chronological omitted).
**Tolerances (target event):** 100m 0.15s · 200m 0.30s · 400m 0.80s · 800m 1.50s · 1500m 3.0s · 3000m 6.0s · 3000m SC 8.0s · 5000m 10.0s.
**Models:** club-fit event-pair formulas, WA bands 750–950 / 800–1000 / 850–1050; **feature** route preferred.
**Run:** `python main.py all_divisions_2plus`
**Note:** `2026_Outdoor_Marks.pdf` is Air Force (pages 1–4 = men), not Colorado State.

## Dataset

| | |
|--|--|
| Result rows (source) | **1,769** (1,212 primary dated + 557 supplementary season-PB) |
| Divisions | NAIA, NCAA D1, NCAA D3 |
| Schools | Air Force, DePaul, Indiana State, Keiser, Marquette, North Central, North Florida, Portland State, Providence, SIUE, Southern Indiana, Troy, UTEP, Utah State, Wisconsin-Oshkosh |
| Prediction rows (feature) | **166** |
| Athlete-seasons tested | **26** |
| Min results/event | **≥2** in both A and B |

## Headline results (season-PB)

| Route | *n* | MedAPE | Within tolerance |
|-------|-----|--------|------------------|
| **feature** | 166 | **1.95%** | **38.6%** |
| pooled | 166 | 3.26% | 22.9% |

---

## Prediction success definition

For each athlete with season PBs in events A and B, and each applicable WA band:

0. Require ≥2 recorded results in A **and** ≥2 in B that season (otherwise skip the pair).

1. Predict B from A using the band’s time model (feature route when available).
2. Compare predicted B to actual season PB in B.
3. **Successful prediction** if |pred − actual| ≤ starter tolerance for event B.

**Athlete success** for a pair: the athlete has **at least one** successful prediction for that pair (any band counts).

## Highest success rates by WA band (prediction-level)

### Band 750-950 (overall 42.4%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 100m → 200m | 6 | 6 | **100.0%** | 0.80% |
| 200m → 100m | 6 | 6 | **100.0%** | 0.50% |
| 200m → 400m | 5 | 4 | **80.0%** | 1.19% |
| 800m → 1500m | 9 | 6 | **66.7%** | 0.90% |
| 1500m → 800m | 9 | 2 | **22.2%** | 1.98% |

### Band 800-1000 (overall 29.3%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 6 | 6 | **100.0%** | 0.69% |
| 100m → 200m | 6 | 4 | **66.7%** | 1.10% |
| 800m → 1500m | 7 | 3 | **42.9%** | 1.28% |
| 1500m → 3000m SC | 5 | 1 | **20.0%** | 5.32% |
| 1500m → 800m | 7 | 1 | **14.3%** | 1.92% |

### Band 850-1050 (overall 45.2%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 6 | 6 | **100.0%** | 0.61% |
| 100m → 200m | 6 | 5 | **83.3%** | 0.48% |
| 200m → 400m | 5 | 3 | **60.0%** | 1.23% |
| 800m → 1500m | 7 | 4 | **57.1%** | 1.12% |
| 1500m → 800m | 7 | 0 | **0.0%** | 3.36% |

## Highest athlete success rates by event pair

| Pair | Athletes | Successful | Athlete success rate | Row success rate | MedAPE |
|------|----------|------------|----------------------|------------------|--------|
| 100m → 200m | 6 | 6 | **100.0%** | 83.3% | 0.77% |
| 200m → 100m | 6 | 6 | **100.0%** | 100.0% | 0.57% |
| 800m → 5000m | 2 | 2 | **100.0%** | 50.0% | 7.79% |
| 3000m SC → 5000m | 1 | 1 | **100.0%** | 100.0% | 0.75% |
| 200m → 400m | 5 | 4 | **80.0%** | 46.7% | 2.06% |
| 800m → 1500m | 9 | 7 | **77.8%** | 56.5% | 1.12% |
| 1500m → 800m | 9 | 3 | **33.3%** | 13.0% | 1.98% |
| 1500m → 5000m | 3 | 1 | **33.3%** | 11.1% | 3.09% |
| 5000m → 1500m | 3 | 1 | **33.3%** | 22.2% | 3.15% |
| 1500m → 3000m SC | 5 | 1 | **20.0%** | 20.0% | 5.57% |
| 3000m SC → 1500m | 5 | 0 | **0.0%** | 0.0% | 4.81% |
| 400m → 200m | 5 | 0 | **0.0%** | 0.0% | 2.55% |

**Overall athlete-level (≥1 hit on any pair):** 19 / 26 = **73.1%** (feature), vs row-level within-tol **38.6%**.

## Artifacts (`output/All_Divisions/2+_Results/`)

- Source scrape reused from parent `All_Divisions/men_test_dataset_results.csv`
- `model_validation/predictions.csv`
- `model_validation/summary_overall.csv`
- `model_validation/summary_by_band.csv`
- `model_validation/summary_by_pair.csv`
- `model_validation/validation_report.txt`
- `model_validation/success_rate_by_band_pair.csv`
- `model_validation/athlete_success_rate_by_pair.csv`
- `model_validation/research/` (inferential CSVs, report, plots)

