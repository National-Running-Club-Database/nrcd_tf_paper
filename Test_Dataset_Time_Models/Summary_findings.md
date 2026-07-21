# Summary Findings — Men's Outdoor Time-Model External Validation

**Dataset:** Men's outdoor marks — primary dated NCAA D1 meets (SIUE, North Florida, DePaul) plus supplementary undated season-best matrices (USI, North Central, Wisconsin-Oshkosh, Keiser)  
**Models:** Club-fit outdoor pair models with steeplechase, WA bands 750–950 / 800–1000 / 850–1050  
**Sample (season-PB · feature, with supplementary):** 802 prediction pairs · 40 Men pair models parsed  
**Chronological sample (dated primary only):** 139 prediction pairs (unchanged by supplementary lists)  
**Primary metric:** median absolute percent error (MedAPE); absolute seconds only within event group/pair  

Figures: `output/model_validation/research/plots/`  
Full inferential tables: `output/model_validation/research/`  
Supplementary extract: `output/men_supplementary_season_pb.csv`

---

## 1. Bottom line

1. **Season-PB is the preferred external protocol** (matches training targets and outperforms chronological transfer). Because of that, previously unused undated season lists are included as **supplementary season-best rows** (`Source_Role=supplementary_season_pb`). Chronological checks still use dated primary rows only.
2. **Feature-routed formulas outperform pooled formulas** on the expanded season-PB set (paired Wilcoxon on \|%\| error, *p* ≪ 0.05). Season-PB MedAPE falls from **2.39%** (pooled) to **1.43%** (feature); 95% bootstrap CIs do not overlap.
3. **Sprint pairs transfer cleanly** at scale (feature MedAE ≈ 0.21 s, MedAPE ≈ 1.0%); distance is usable on percent error but steeplechase / long-distance pairs remain weaker.
4. **Chronological early→later transfer** (dated only) remains slightly worse than season-PB (feature MedAPE **1.80%** vs **1.43%**), supporting the decision to expand season-PB coverage rather than force undated lists into chronological checks.
5. **Positive bias** persists (models tend to predict slower than actual collegiate marks), consistent with club→college distribution shift.

---

## 2. Data roles

| Role | Sources | Rows | Used in |
|------|---------|------|---------|
| `primary_dated` | SIUE, North Florida, DePaul | 419 | season-PB + chronological |
| `supplementary_season_pb` | USI, North Central, Oshkosh, Keiser | 523 | season-PB only |

Supplementary scrapers emit **one season-best row per athlete–event** (no fake chronology). Divisions include NCAA D1 / D3 / NAIA.

---

## 3. Overall performance (with 95% CIs)

| Mode | Route | *n* | MedAPE [95% CI] | Within tolerance [95% CI] | Spearman ρ |
|------|-------|-----|-----------------|---------------------------|------------|
| season_pb | feature | 802 | **1.43%** [1.32, 1.65] | 46.0% [42.5%, 49.5%] | 0.989 |
| season_pb | pooled | 802 | 2.39% [2.09, 2.68] | 31.5% [28.3%, 34.9%] | 0.983 |
| chronological | feature | 139 | 1.80% [1.46, 2.12] | 33.8% [26.0%, 42.3%] | 0.983 |
| chronological | pooled | 139 | 2.35% [1.67, 2.91] | 33.1% [25.4%, 41.6%] | 0.977 |

### Within event group (season-PB · feature)

| Group | *n* | MedAE [95% CI] | MedAPE |
|-------|-----|----------------|--------|
| Sprints | 298 | **0.209 s** [0.166, 0.264] | 1.03% |
| Distance | 504 | 4.78 s [3.81, 6.03] | 1.90% |

---

## 4. Feature routing vs pooled (paired Wilcoxon)

On the expanded season-PB set (*n* = 802 paired rows): feature MedAPE **1.43%** vs pooled **2.39%**, Wilcoxon *p* = 3.7×10⁻²⁵ — feature better.

Chronological feature-vs-pooled contrast remains significant but smaller (*n* = 139).

---

## 5. Best-calibrated pairs (season-PB · feature · *n* ≥ 5)

| Band | Pair | *n* | MedAPE [95% CI] | MedAE |
|------|------|-----|-----------------|-------|
| 750–950 | 200m → 100m | 31 | 0.61% [0.51, 0.81] | 0.069 s |
| 850–1050 | 100m → 200m | 23 | 0.62% [0.49, 0.93] | 0.138 s |
| 850–1050 | 200m → 100m | 23 | 0.65% [0.32, 1.08] | 0.069 s |
| 800–1000 | 200m → 100m | 29 | 0.73% [0.46, 1.05] | 0.080 s |
| 750–950 | 800m → 1500m | 35 | 0.75% [0.55, 1.02] | 1.85 s |

Weakest pairs remain steeplechase and long-distance transfers (wide CIs; treat as exploratory).

---

## 6. Limitations

1. Supplementary rows lack competition dates — they support season-PB only.
2. HTML matrices can include prelim/final noise; scrapers take the best parsed mark per athlete–event.
3. Mixed divisions (D1/D3/NAIA) in the supplementary set increase heterogeneity vs club training data.
4. Non-independence across bands/pairs; prefer CIs and effect sizes alongside *p*-values.
5. Positive prediction bias remains under distribution shift.

---

## 7. Reproducibility

```bash
cd test_dataset_time_models
../.venv/bin/python main.py all
```

See [README.md](README.md) for methods and file layout.
