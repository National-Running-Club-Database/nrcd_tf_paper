# Summary Findings — time_models

**Data:** relay-inclusive outdoor Sprints/Distance CSVs, 2024–2026, results on/after
March 1; athlete-season personal bests; steeplechase excluded from the baseline cross-event
models (see `new_steeplechase_data/` for the steeplechase-inclusive point-band variant).
**Evaluation:** 5-fold cross-validation, median absolute error (seconds), seed = 42.
**Full reports:** see the per-subfolder `*_findings.txt` cited in each section below.

---

## 1. Bottom line

1. **Point-banding by WA score beats a single pooled model.** The best band
   (850–1050, width 200) cuts mean CV error from **6.768s (unbanded) to 2.412s** —
   a 64% reduction — though only over 10 pairs / 187 athlete-seasons.
2. **Within a band, cohort/feature-routed formulas beat the pooled band formula**
   in every tested band (750–950, 800–1000, 850–1050); `bal_x_best_event`,
   `best_is_from`, and `from_stronger_wa` are the most consistently useful routing features.
3. **Specialization (large WA spread across events) makes athletes *harder* to predict
   cross-event, not easier** — balanced athletes are more predictable than specialized
   ones in the 6-race cohort.
4. **Doubling (2+ events per meet) does not clearly improve or hurt time-model accuracy**,
   and within-athlete double-meet results average **15.3 WA points lower** than solo-meet
   results for the same athlete.
5. **More races per season modestly improves predictability**: the 6–10-race cohort
   (mean CV 3.700s) beats both the "exactly 6 races" cohort (4.931s) and the full,
   unbanded population (6.768s).
6. **Alternative functional forms beat plain linear regression for most pairs**: 20 of
   24 event pairs improve by >0.01s CV median error when using multivariate OLS,
   quadratic, or speed-binned-ratio models instead of simple linear regression.

**Inferential confirmation (`research/`):** Across FI point-band pairs, the best
feature strategy beats pooled CV MedAE on a majority of pairs (binomial *p* ≪ 0.05
overall); median Δ and beat-rate CIs are in `research/inferential_report.txt`.

## 2. Point-banding (`Point_Bands_Time_Models/`)

Testing WA point bands of widths 100/150/200 (Sprints + Distance, non-relay, non-steeple):

| Band (width 200) | Mean CV error | Pairs | Athlete-seasons |
|-------------------|---------------|-------|------------------|
| **850–1050** | **2.412s** | 10 | 187 |
| 800–1000 | 2.838s | 18 | — |
| 750–950 | 3.871s | 22 | — |
| 700–900 | 4.528s | 24 | — |
| 650–850 | 5.046s | 24 | — |
| Unbanded (all) | 6.768s | 24 | — |

Results are consistent across widths 100/150/200 — the top band always lands near
850–950/1000/1050 with mean CV 2.412s, but with fewer pairs/athletes as the score range
narrows (a coverage/accuracy trade-off).

## 3. Cohort feature importance within bands

Testing whether cohort labels beat the pooled band-level formula:

| Band | Pairs (athlete-seasons) | Best feature (win rate) | Verdict |
|------|--------------------------|--------------------------|---------|
| 750–950 | 22 (556) | `pair_wa_gap_50` 18/22, `best_is_from` 18/22, `bal_x_best_event` 17/22 | YES — multiple features help |
| 800–1000 | 18 (351) | `from_stronger_wa` 11/18, `best_is_from` 10/18 | YES |
| 850–1050 | 10 (187) | `bal_x_best_event` 8/10, `best_event` 8/10 | YES |

Pooled (no routing) is the best choice for only 0–2 pairs per band — in nearly every
case, some cohort formula beats the single pooled curve.

## 4. Does specialization help or hurt? (`specialized_time_models/`)

**Short answer from the source report: "No — specialization does not help; it hurts."**
Among athletes with exactly 6 season races: Men Sprints 39 athlete-seasons (54%
specialized, median WA spread 63), Men Distance 58 (69% specialized, median spread 63).
Most high-volume athletes are specialized, not balanced, by the ≥50-WA-spread rule — but
balanced athletes are the more predictable ones cross-event.

**Short/Long stratification** (`Short_Long_Specialization_Time_Models/`), at 5-or-6 races:
cohort formulas beat pooled in **12/24 (50%)** of tested slices; routing by short/mid vs.
long specialization beats the plain group model in **7/12 (58%)**; long-specialization
athletes are more predictable in 7/12 comparisons (mean Δ = +2.635s advantage where long wins).

## 5. New factors beyond balanced/specialized (`new_factors/`)

Ranked by how often a factor's joint model beats the bal/spec baseline (16 pairs, 5-or-6-race cohort):

| Factor | Joint beats bal/spec | Mean Δ |
|--------|------------------------|--------|
| `pair_wa_gap_50` (|WA gap| ≥ 50 pts) | **14/16** | +1.664s |
| `pair_wa_gap_median` | 14/16 | +1.262s |
| `from_stronger_wa` | 12/16 | +0.869s |
| `best_is_from` | 11/16 | +0.642s |
| `best_is_to` / `predict_to_best` | 11/16 | +0.388s |
| `best_event` (identity) | 10/16 | -0.097s |
| `events_competed` (2 vs 3+) | 7/14 | -0.051s |
| `season_span` | 7/16 | -0.159s |

The WA-gap-based factors (`pair_wa_gap_50`, `pair_wa_gap_median`) are the single most
useful engineered features found across the whole `time_models/` module.

### Doubling (`new_factors/doubling_analysis/`)

- 98.1% of the 5-or-6-race population **ever doubled** (2+ events in a meet) — too
  common to compare doubled vs. never-doubled reliably (never-doubled n=6).
- **Within-athlete**: double-meet results average **-15.3 WA** vs. solo-meet results for
  the same athlete — doubling does not raise scores, and if anything is associated with
  a small drop.
- **Time-model verdict**: no clear accuracy gain from doubling-based factors vs. pooled
  models (`double_meet_share` beats pooled only 7/16; `ever_doubled` beats pooled only 3/8).

## 6. Race-count cohorts (`six_to_ten_races_time_models/`, `model_search/race_count/`)

| Cohort | Pairs | Mean best CV error |
|--------|-------|---------------------|
| All (unbanded) | 24 | 6.768s |
| Exactly 6 races | 10 | 4.931s |
| Exactly 7 races | 4 | 1.991s |
| Exactly 8 races | 2 | 0.204s |
| **6–10 races combined** | **18** | **3.700s** |

More races per season is generally associated with more accurate cross-event prediction
(smaller cohorts of high-volume racers have less noisy PBs), though pair counts shrink
sharply at exactly 9–10 races (no pairs meet the n≥20 threshold). The pooled 6–10-race
cohort (3.700s) is a reasonable middle ground between coverage (18 pairs) and accuracy.
Matched 6-vs-6–10 comparison: the wider 6–10 cohort wins on 7 of 10 matched pairs
(mean Δ -0.107s).

## 7. Alternative functional forms (`model_search/`)

Comparing candidate model types against the linear-OLS baseline (5-fold CV, seed 42):
**20 of 24 event pairs** improve by more than 0.01s CV median absolute error when using
a non-linear or multivariate alternative. Notable wins:

- **Men 100m → 400m**: multivariate OLS (using both 100m and 200m as inputs) achieves
  CV median error **1.337s vs. 1.990s** for linear-from-100m-alone — a 33% improvement.
  Formula: `400m = 5.39 − 0.770×100m + 2.4323×200m`.
- **Men 100m → 200m**: `median_ratio` (`200m = 2.0440 × 100m`) edges out linear
  (0.422s vs. 0.454s CV median error).
- **Men 200m → 100m**: `quadratic` edges out linear (0.189s vs. 0.203s).

The largest accuracy gains come from **using two known marks** (multivariate) or
speed-binned ratios, especially for long cross-event "hops" (100m→400m, 800m→5000m).

## 8. Baseline cross-event formulas (root-level, `cross_event_time_models.csv`)

Representative pooled-linear formulas (men's sprints, full population, not banded):

| Pair | n | r | Formula | RMSE | Median |error|| 
|------|---|---|---------|------|--------------|
| 100m → 200m | 447 | 0.897 | `200m = 2.236×100m − 2.199` | 0.894s | 0.441s |
| 100m → 400m | 174 | 0.809 | `400m = 5.046×100m − 5.145` | 3.675s | 1.984s |
| 200m → 100m | 447 | 0.897 | `100m = 3.144 + 0.360×200m` | 0.359s | 0.203s |
| 400m → 200m | 293 | 0.849 | `200m = 5.305 + 0.353×400m` | 1.193s | 0.550s |

This is the module's baseline — every point-banded, cohort-routed, and alternative-form
model above is evaluated relative to it.

## 9. Limitations

1. **Small n in the most accurate slices** — the best point band (850–1050) has only 10
   pairs / 187 athlete-seasons; race-count cohorts of 8–10 races have as few as 2 pairs.
   Point estimates in these high-accuracy, low-n slices should be treated as
   directional, not definitive.
2. **Multiple comparisons, no correction** — dozens of candidate bands/factors/functional
   forms are compared per pair without a formal multiple-testing correction; "beats
   pooled X/N" win-rate framing is descriptive, not inferential.
3. **Non-independent observations** — the same athlete can appear in multiple pairs,
   bands, and cohorts; cross-validation is done per-pair, not with athlete-level blocking
   across the whole module, so reported accuracy may be somewhat optimistic.
4. **Club-to-club transfer only within this module** — see
   `../test_dataset_time_models/Summary_findings.md` for external validation against
   dated NCAA D1 results, which finds meaningfully higher error (MedAPE ~1.5–3%) and a
   systematic slow-prediction bias when club-fit formulas are applied out-of-sample.
5. **Steeplechase excluded from the baseline** models in this folder; see
   `new_steeplechase_data/` for the steeplechase-inclusive point-band variant.
6. **Doubling and specialization analyses are cross-sectional/associative** — no causal
   claims are made about *why* balanced athletes are more predictable or why doubling
   correlates with slightly lower WA.

## 10. Reproducibility

```bash
cd time_models
python main.py all
```

Subfolder pipelines must be run individually — see [README.md](README.md) for the full
list of scripts and their outputs.
