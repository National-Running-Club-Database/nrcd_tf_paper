# Summary Findings — non_relays_findings (RQ1 + RQ3)

**Data:** NIRCA club outdoor track, National Running Club Database, 2024–2026, individual (non-relay) events.
**Metric:** gender-specific World Athletics (WA) scoring points.
**Full reports:** [findings.md](findings.md), [improved_rq1_outputs/improved_findings.md](improved_rq1_outputs/improved_findings.md), [RQ3_Population_Dispersion/rq3_population_wa_iqr_by_event.txt](RQ3_Population_Dispersion/rq3_population_wa_iqr_by_event.txt)

---

## 1. Bottom line

1. **The 1500m dominates absolute specialization** for both genders (42% of men's, 46%
   of women's distance best-event assignments), but is not always the best nationals
   path (RQ1C).
2. **Nationals top-8 thresholds are highest in middle-distance track and sprints**
   (800m/1500m/200m/100m, ~790–830 WA) and lowest in steeplechase and discus.
3. **Gender asymmetry is strongest in distance**: men peak in the 1500m even when they
   also run 5000m; women who double in 1500m/5000m more often peak in the **5000m**.
4. **Steeplechase is a "nationals opportunity" event, not a peak-scoring event**: only
   ~3% of best-event assignments, but the **highest nationals clear-rate** among
   women's distance events (16.3% vs. 8.0% for 1500m) and the **lowest 8th-place bar**
   among men's distance events (456.0 vs. 827.0 for 1500m).

## 2. RQ1A — best event (specialization)

**Combined 2024–2026 distance best-event counts:**

| Event | Men (n=1,933) | Women (n=946) |
|-------|---------------|---------------|
| 1500m | **816 (42%)** | **436 (46%)** |
| 800m | 626 (32%) | 238 (25%) |
| 5000m | 429 (22%) | 243 (26%) |
| 3000m Steeplechase | 62 (3%) | 29 (3%) |

**Pairwise (dual-event athletes, 2024–2026 combined):**

| Pairing | Men (A–B) | Women (A–B) |
|---------|-----------|--------------|
| 800m vs. 1500m | 232–**349** | 58–**214** |
| 1500m vs. 5000m | **294**–142 | 55–**89** |
| 1500m vs. Steeple | **170**–1 | **79**–2 |

Men peak in the 1500m even against the 5000m; women more often peak in the 5000m when
they run both (62% of dual-event pairwise comparisons, statistically significant at
p = 0.006 per the improved analysis).

**Sprints (combined 2024–2026):** Men 100m 465 (37%) > 400m 431 (35%) > 200m 352 (28%);
Women 100m 245 (45%) > 400m 192 (35%) > 200m 105 (19%). Women's 100m beats 200m 84% of
the time head-to-head, but 400m beats 200m among women who run both (94 vs. 34).

**Statistical significance (improved analysis):** best-event distributions are
significantly non-uniform (χ² p < 0.001) for every discipline × gender group **except**
women's hurdles (χ² = 0.08, p = 0.96). Men's sprint pairwise comparisons are **not**
statistically significant (all p > 0.15) — no single sprint dominates.

## 3. RQ1B — nationals top-8 thresholds

**3-year average 8th-place WA thresholds (highest → lowest):**

| Rank | Men | Women |
|------|-----|-------|
| 1 | 800m — 829.3 | 100m — 814.3 |
| 2 | 1500m — 827.0 | 5000m — 808.5 |
| 3 | 200m — 820.3 | 200m — 799.3 |
| 4 | 100m — 818.0 | 1500m — 791.7 |
| 5 | 400m / Long Jump — 791.7 | 400m — 788.3 |
| … | … | … |
| 14 | 3000m SC — **456.0** | Discus — **504.3** |

Nationals top-8 distributions sit **200–300 WA points above the season-wide median** in
every event (e.g., 2026 men's 100m: nationals top-8 mean 879/median 870 vs. full-season
mean 606/median 623). Mann-Whitney U confirms this is significant in **83/83**
event-season tests (p < 0.001 after BH-FDR), with rank-biserial r typically 0.55–0.95.

Nationals fields range from **~40 competitors** (men's steeple) to **~290** (men's 1500m,
2025); smaller fields show more year-to-year volatility (men's steeple 8th-place: 420–484
across years). Women's 5000m had the **highest** women's 8th-place bar in 2026 (846, vs.
825 for 100m and 838 for 1500m).

## 4. RQ1C — most competitive event

| Discipline | RQ1A=RQ1C agreement (men) | RQ1A=RQ1C agreement (women) |
|------------|---------------------------|------------------------------|
| Sprints | 94.0% | 91.4% |
| Throws | 96.4% | 65.8% |
| Jumps | 75.5% | 78.1% |
| Distance | **75.8%** | **68.6%** |
| Hurdles | 66.7% | 83.0% |

**24–35% of multi-event distance athletes** should pursue a different event than their
absolute best to maximize nationals competitiveness; sprinters show the highest
agreement (91–94%). All agreement rates are significantly above a 50% null (p < 0.001).

**Men's distance (2024–2026):** 1500m dominates RQ1A (816 athletes) but 5000m has a
lower nationals bar (786.0 vs. 827.0) and slightly higher clear-rate (4.8% vs. 4.3%);
**steeplechase has the highest clear-rate (8.5%)** despite being the rarest best event
(62 athletes).

**Head-to-head margin comparisons** (athletes competing in both events): 5000m more
competitive than 1500m for both women (79 vs. 65) and men (228 vs. 209), though this
margin comparison does **not** reach significance (men p = 0.39; women p = 0.28) — the
descriptive trend is directionally consistent but statistically noisier than the raw
score comparisons. Steeple beats 1500m on margin for men (132 vs. 39) and women (75 vs. 6).

**Lowest 8th-place bars / highest clear-rates** (population-level nationals opportunity):
Men — 110m Hurdles 28.4%, High Jump 27.9%, Triple Jump 23.0%; Women — High Jump 37.3%,
400m Hurdles 34.8%, Triple Jump 31.6%. By contrast, 1500m/800m/100m clear-rates are only
4–10%.

## 5. RQ3 — population-level WA dispersion (IQR)

Ranking events by median personal-best WA score (≥100 unique athletes, 2024–2026 combined):

**Men — top and bottom:**

| Event | N | Median | IQR |
|-------|---|--------|-----|
| High Jump (highest median) | 104 | 646.0 | 200.2 |
| 200m | 706 | 606.5 | 256.8 |
| 1500m | 1,179 | 512.0 | 289.5 |
| 3000m Steeplechase (lowest median) | 265 | 194.0 | 257.0 |

**Women — top and bottom:**

| Event | N | Median | IQR |
|-------|---|--------|-----|
| Long Jump (highest median) | 157 | 660.0 | 156.0 |
| 1500m | 577 | 550.0 | 223.0 |
| 3000m Steeplechase (lowest median) | 135 | 345.0 | 205.0 |

Steeplechase has both the **lowest median WA score** and (for men) among the **widest
IQR** — a wide, low-scoring population that nonetheless clears the nationals bar at a
disproportionately high rate (per RQ1C), reinforcing steeple's role as a competitive
opportunity event rather than a peak-scoring one.

## 6. Limitations

1. **WA points are table-based comparability scores**, not direct physiological measures.
2. **Single best result per athlete** can be influenced by outlier performances (wind,
   pacing, tactics, one exceptional race).
3. **NIRCA club athletes only** — findings may not generalize to NCAA/NAIA varsity programs.
4. **Nationals round labeling is inconsistent** across years (Finals vs. unlabeled rows);
   a 100m/200m-prelim rule was applied as a standardization convention.
5. **Selection bias** — athletes and coaches choose which events to enter; observed
   patterns reflect both ability and opportunity, not a randomized assignment.
6. **Women's 5000m had no 2025 nationals event**, affecting that event's 3-year average threshold.
7. **Relay events are excluded** here (see `relays_findings/` for the relay-inclusive version).
8. **Multiple-comparisons correction** (BH-FDR) is applied in the improved analysis, but
   descriptive vs. inferential results can diverge (e.g., women's 5000m-vs-1500m RQ1A
   pairwise is significant while the RQ1C margin comparison is not) — report both, not
   just the more favorable one.

## 7. Reproducibility

```bash
cd non_relays_findings
python main.py all
```

See [README.md](README.md) for the discipline-subfolder scripts (run individually,
not wired into `main.py`) and folder dependencies.
