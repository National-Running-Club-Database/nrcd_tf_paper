# All_Divisions · 2+_From_Only External Validation Findings

**Filter:** ≥2 results in from-event (A) only; no minimum on to-event (B). Same All_Divisions men's scrape (NCAA D1 / D3 / NAIA).
**Protocol:** season-PB only (chronological omitted).
**Tolerances (target event):** 100m 0.15s · 200m 0.30s · 400m 0.80s · 800m 1.50s · 1500m 3.0s · 3000m 6.0s · 3000m SC 8.0s · 5000m 10.0s.
**Models:** club-fit event-pair formulas, WA bands 750–950 / 800–1000 / 850–1050; **feature** route preferred.
**Run:** `python main.py all_divisions_2plus_from`
**Note:** `2026_Outdoor_Marks.pdf` is Air Force (pages 1–4 = men), not Colorado State.

## Dataset

| | |
|--|--|
| Result rows (source) | **1,769** (1,212 primary dated + 557 supplementary season-PB) |
| Divisions | NAIA, NCAA D1, NCAA D3 |
| Schools | Air Force, DePaul, Indiana State, Keiser, Marquette, North Central, North Florida, Portland State, Providence, SIUE, Southern Indiana, Troy, UTEP, Utah State, Wisconsin-Oshkosh |
| Prediction rows (feature) | **236** |
| Athlete-seasons tested | **47** |
| Min results | A ≥**2**; B ≥**1** |

## Headline results (season-PB)

| Route | *n* | MedAPE | Within tolerance |
|-------|-----|--------|------------------|
| **feature** | 236 | **1.98%** | **35.2%** |
| pooled | 236 | 3.17% | 22.9% |

### Contrast with other filters (feature route)

| Filter | *n* | MedAPE | Within tol | Athlete-seasons |
|--------|-----|--------|------------|-----------------|
| Unrestricted (A≥1, B≥1) | 1100 | 1.69% | 42.7% | 157 |
| A≥2 only (this file) | 236 | 1.98% | 35.2% | 47 |
| A≥2 and B≥2 | 166 | 1.95% | 38.6% | 26 |

---

## Prediction success definition

For each athlete with season PBs in events A and B, and each applicable WA band:

0. Require ≥2 recorded results in A that season; B may have a single result (otherwise skip the pair).

1. Predict B from A using the band’s time model (feature route when available).
2. Compare predicted B to actual season PB in B.
3. **Successful prediction** if |pred − actual| ≤ starter tolerance for event B.

**Athlete success** for a pair: the athlete has **at least one** successful prediction for that pair (any band counts).

## Highest success rates by WA band (prediction-level)

### Band 750-950 (overall 42.1%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 6 | 6 | **100.0%** | 0.50% |
| 100m → 200m | 8 | 7 | **87.5%** | 0.80% |
| 200m → 400m | 5 | 4 | **80.0%** | 1.19% |
| 3000m SC → 5000m | 5 | 4 | **80.0%** | 0.75% |
| 800m → 1500m | 13 | 8 | **61.5%** | 0.90% |

### Band 800-1000 (overall 27.2%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 6 | 6 | **100.0%** | 0.69% |
| 100m → 200m | 9 | 6 | **66.7%** | 1.00% |
| 800m → 1500m | 10 | 4 | **40.0%** | 1.30% |
| 1500m → 3000m SC | 6 | 1 | **16.7%** | 3.71% |
| 400m → 200m | 6 | 1 | **16.7%** | 2.02% |

### Band 850-1050 (overall 35.0%)

| Pair | *n* | Successes | Rate | MedAPE |
|------|-----|-----------|------|--------|
| 200m → 100m | 6 | 6 | **100.0%** | 0.61% |
| 100m → 200m | 9 | 6 | **66.7%** | 0.56% |
| 200m → 400m | 5 | 3 | **60.0%** | 1.23% |
| 800m → 1500m | 10 | 4 | **40.0%** | 1.28% |
| 1500m → 5000m | 6 | 1 | **16.7%** | 3.04% |

## Highest athlete success rates by event pair

| Pair | Athletes | Successful | Athlete success rate | Row success rate | MedAPE |
|------|----------|------------|----------------------|------------------|--------|
| 200m → 100m | 6 | 6 | **100.0%** | 100.0% | 0.57% |
| 800m → 5000m | 2 | 2 | **100.0%** | 50.0% | 7.79% |
| 100m → 200m | 9 | 8 | **88.9%** | 73.1% | 0.80% |
| 200m → 400m | 5 | 4 | **80.0%** | 46.7% | 2.06% |
| 3000m SC → 5000m | 5 | 4 | **80.0%** | 80.0% | 0.75% |
| 800m → 1500m | 13 | 9 | **69.2%** | 48.5% | 1.23% |
| 1500m → 5000m | 6 | 3 | **50.0%** | 16.7% | 3.75% |
| 5000m → 3000m SC | 2 | 1 | **50.0%** | 50.0% | 1.40% |
| 5000m → 1500m | 9 | 3 | **33.3%** | 16.0% | 2.91% |
| 1500m → 800m | 12 | 3 | **25.0%** | 10.0% | 2.07% |
| 3000m SC → 1500m | 9 | 2 | **22.2%** | 11.1% | 3.45% |
| 1500m → 3000m SC | 6 | 1 | **16.7%** | 16.7% | 3.95% |

**Overall athlete-level (≥1 hit on any pair):** 31 / 47 = **66.0%** (feature), vs row-level within-tol **35.2%**.

## Artifacts (`output/All_Divisions/2+_From_Only/`)

- Source scrape reused from parent `All_Divisions/men_test_dataset_results.csv`
- `model_validation/predictions.csv`
- `model_validation/summary_overall.csv`
- `model_validation/summary_by_band.csv`
- `model_validation/summary_by_pair.csv`
- `model_validation/validation_report.txt`
- `model_validation/success_rate_by_band_pair.csv`
- `model_validation/athlete_success_rate_by_pair.csv`
- `model_validation/research/` (inferential CSVs, report, plots)

