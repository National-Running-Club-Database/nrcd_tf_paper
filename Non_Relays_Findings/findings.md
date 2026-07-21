# RQ1 Findings: Event Specialization and Competitiveness in Collegiate Club Outdoor Track

**Data:** National Running Club Database (NRCD), outdoor track 2024–2026  
**Metric:** World Athletics scoring points (gender-specific)  
**Disciplines:** Sprints, Distance, Hurdles, Jumps, Throws

---

## Executive Summary

RQ1 asks how World Athletics scoring reveals **where athletes are strongest** (RQ1A), **what nationals-caliber performance looks like by event** (RQ1B), and **which events athletes should pursue to maximize competitiveness** (RQ1C).

Three central conclusions emerge:

1. **The 1500m (men) and 1500m/100m (women) dominate absolute specialization (RQ1A)**, but this does not always align with the best path to nationals (RQ1C).
2. **Nationals top-8 scoring sets a high bar in sprints and middle-distance track** (8th-place thresholds ~790–830 WA points), while **field events and hurdles offer lower thresholds and higher rates of athletes clearing the nationals bar**.
3. **Gender asymmetry is strongest in distance:** men score highest in the 1500m; women who double in 1500m and 5000m are more often relatively stronger in the **5000m**, and **steeplechase** offers the highest nationals clear-rate among women's distance events despite rarely being the absolute best event.

---

## Methodology Overview

| Component | Definition |
|-----------|------------|
| **RQ1A (best event)** | Event associated with the athlete's single highest World Athletics personal best across the analysis window |
| **RQ1A (pairwise)** | Among athletes with results in both events, compare top score in Event A vs. top score in Event B |
| **RQ1B (nationals top 8)** | Rank nationals results: 100m/200m from **prelims**; all other events from **finals** (or unlabeled nationals rows). Retain places 1–8 by performance |
| **RQ1C (most competitive)** | Event with the largest margin above the 3-year average **8th-place nationals WA threshold**; if none clear the bar, the smallest deficit |

**Analysis windows:** Individual seasons (2024, 2025, 2026) and combined 2024–2026.  
**Outputs:** Discipline folders (`*_Events_Counting/`), `RQ1B_Nationals/`, `RQ1C_Competitiveness/`.

---

## RQ1A — Which Event Are Athletes Best At?

### Finding 1: The 1500m is the modal best distance event for both genders

**Combined 2024–2026 distance best-event counts:**

| Event | Men (n=1,933) | Women (n=946) |
|-------|---------------|---------------|
| 1500m | **816 (42%)** | **436 (46%)** |
| 800m | 626 (32%) | 238 (25%) |
| 5000m | 429 (22%) | 243 (26%) |
| 3000m Steeplechase | 62 (3%) | 29 (3%) |

The ranking **1500m > 800m > 5000m > steeple** is stable across all three seasons for men. For women, 1500m leads overall, but **5000m (26%) rivals 800m (25%)** as the second most common best event.

### Finding 2: Pairwise comparisons show the mile beats both shorter and longer distances — for men

**Men's distance (2024–2026 combined, dual-event athletes):**

| Pairing | Event A wins | Event B wins |
|---------|--------------|--------------|
| 800m vs. 1500m | 232 | **349** (~60%) |
| 1500m vs. 5000m | **294** | 142 (~67%) |
| 800m vs. 5000m | 109 | 114 (even) |
| 1500m vs. Steeple | **170** | 1 |

**Women's distance (2024–2026 combined):**

| Pairing | Event A wins | Event B wins |
|---------|--------------|--------------|
| 800m vs. 1500m | 58 | **214** (~79%) |
| 1500m vs. 5000m | 55 | **89** (~62%) |
| 800m vs. 5000m | 15 | **70** (~82%) |
| 1500m vs. Steeple | **79** | 2 |

**Key gender difference:** Men peak in World Athletics points at the **1500m** even when they also run 5000m. Women who compete in both **1500m and 5000m** more often peak in the **5000m** — the clearest gender asymmetry in the distance dataset.

### Finding 3: Steeplechase is rarely the absolute best event

Despite meaningful participation (281 men, 135 women across three seasons), steeplechase accounts for only **~3%** of best-event assignments. In pairwise comparisons against 1500m, steeple wins **1 of 171** matchups (men) and **2 of 81** (women).

*Note: Early 2026 men's steeple data had zero World Athletics points; results reported here use corrected data.*

### Finding 4: Sprint specialization favors shorter events, with gender differences in the 200m/400m split

**Combined 2024–2026 sprint best-event counts:**

| Event | Men (n=1,248) | Women (n=542) |
|-------|---------------|---------------|
| 100m | 465 (37%) | **245 (45%)** |
| 400m | 431 (35%) | 192 (35%) |
| 200m | 352 (28%) | 105 (19%) |

**Pairwise highlights:**
- **Men:** 100m beats 200m (239 vs. 208); 200m beats 400m (155 vs. 136) — a gradient toward shorter events.
- **Women:** 100m dominates 200m (171 vs. 32, **84%**); but **400m beats 200m** (94 vs. 34) among dual athletes — women who extend to the 400m tend to score higher there than in the 200m.

### Finding 5: Hurdles, jumps, and throws show discipline-specific specialization

| Discipline | Men — modal best event | Women — modal best event |
|------------|------------------------|--------------------------|
| Hurdles | 400m Hurdles (154) | 400m Hurdles (75) |
| Jumps | Long Jump (198) | Long Jump (119) |
| Throws | Shot Put (127) | Shot Put (99) |

### Finding 6: Season-to-season patterns are stable

Proportional best-event distributions do not shift dramatically year to year. The 2024–2026 combined results closely mirror individual seasons, suggesting these specialization patterns are structural rather than driven by a single anomalous year.

**Exception:** Women's 5000m participation dropped sharply in 2025 (54 competitors vs. 158 in 2024), affecting that season's pairwise comparisons but not the combined trend.

---

## RQ1B — Nationals Top-8 World Athletics Distributions

### Finding 7: Nationals top-8 scores are far above the season median in every event

For every event examined, the **nationals top-8 distribution** sits well above the **full-season distribution**. Typical gap: **200–300 World Athletics points** between the nationals top-8 median and the season-wide median.

Example (2026 men's 100m):
- Nationals top-8: mean **879**, median **870**
- Full season: mean **606**, median **623**

This holds across sprints, distance, hurdles, jumps, and throws.

### Finding 8: The 8th-place nationals bar is highest in middle-distance track and sprints

**3-year average 8th-place WA thresholds (highest to lowest):**

**Men:**
1. 800m — 829.3
2. 1500m — 827.0
3. 200m — 820.3
4. 100m — 818.0
5. 400m / Long Jump — 791.7
6. 5000m — 786.0
…
14. 3000m Steeplechase — **456.0**

**Women:**
1. 100m — 814.3
2. 5000m — 808.5
3. 200m — 799.3
4. 1500m — 791.7
5. 400m — 788.3
…
14. Discus — **504.3**

**Implication:** Reaching nationals in the 800m or 1500m requires the highest absolute scoring level. Steeplechase and discus require the lowest 8th-place scores — but this reflects both field depth and the scoring curve, not necessarily that the events are "easier" athletically.

### Finding 9: Event depth at nationals varies substantially

Nationals pools range from **~40 competitors** (men's steeple) to **~290** (men's 1500m in 2025). Events with larger nationals fields tend to have more stable year-to-year 8th-place thresholds; smaller events show greater volatility (e.g., men's steeple: 420–484 across years).

### Finding 10: Women's 5000m had the highest 8th-place bar in 2026

In 2026, women's 5000m 8th-place WA (**846**) ranked **#1 for women that year** — above 100m (825) and 1500m (838). This reflects an exceptionally strong women's 5000m nationals field in 2026 and supports treating the 5K as a premier women's distance at the championship level.

---

## RQ1C — Which Event Should Athletes Pursue to Be Most Competitive?

### Finding 11: RQ1A and RQ1C diverge most in distance and technical events

**RQ1A = RQ1C agreement among multi-event athletes:**

| Discipline | Men | Women |
|------------|-----|-------|
| Sprints | 94.0% | 91.4% |
| Throws | 96.4% | 65.8% |
| Jumps | 75.5% | 78.1% |
| Distance | **75.8%** | **68.6%** |
| Hurdles | 66.7% | 83.0% |

**24–35% of multi-event distance athletes** should pursue a different event than their absolute best to maximize nationals competitiveness. Sprinters show the highest agreement (~91–94%).

### Finding 12: The "best" event is not always the most competitive event

**Distance example (men, 2024–2026):**

| Metric | 1500m | 5000m | Steeple |
|--------|-------|-------|---------|
| RQ1A best event (count) | **816** | 429 | 62 |
| RQ1C recommended (count) | **669** | 479 | 209 |
| % of competitors clearing 8th-place bar | 4.3% | 4.8% | **8.5%** |
| 8th-place threshold | 827.0 | 786.0 | **456.0** |

1500m dominates absolute strength (RQ1A), but **5000m has a lower nationals bar** and a slightly higher clear-rate. **Steeplechase** has the highest clear-rate among men's distance events (8.5%) despite being the rarest best event — many steeple runners are closer to the nationals threshold than their 1500m marks would suggest.

### Finding 13: Women's distance competitiveness favors 5000m over 1500m in head-to-head

Among athletes who compete in **both 1500m and 5000m** (margin above 8th-place threshold):
- **Women:** 5000m more competitive — **79 vs. 65**
- **Men:** 5000m more competitive — **228 vs. 209** (despite 1500m dominating RQ1A)

Among athletes who compete in **both 1500m and steeplechase**:
- **Men:** Steeple more competitive — **132 vs. 39**
- **Women:** Steeple more competitive — **75 vs. 6**

**Coaching implication:** For women choosing between the mile and 5K, World Athletics points suggest specializing in the event where they are absolutely strongest (often 1500m), but **nationals competitiveness often favors the 5000m**. For steeple/1500m doublers, the steeple is almost always the better nationals path on margin.

### Finding 14: Population-level "easiest nationals paths" differ from highest-scoring events

**Highest % of season competitors clearing the 8th-place nationals bar:**

| Men | Clear rate | Women | Clear rate |
|-----|------------|-------|------------|
| 110m Hurdles | **28.4%** | High Jump | **37.3%** |
| High Jump | 27.9% | 400m Hurdles | 34.8% |
| Triple Jump | 23.0% | Triple Jump | 31.6% |
| Discus | 19.1% | Discus | 29.1% |
| … | | … | |
| 1500m | **4.3%** | 800m | 6.3% |
| 800m | 4.6% | 1500m | 8.0% |

Events with the **highest absolute 8th-place thresholds** (800m, 1500m, 100m) have the **lowest clear-rates** (4–10%). Technical and field events offer more viable nationals pathways for the typical club athlete, even when peak World Athletics scores are lower.

### Finding 15: Sprint RQ1C recommendations slightly favor the 400m for men

**Men's sprint RQ1C (most competitive event):** 400m (466) > 100m (462) > 200m (320)  
**Men's sprint RQ1A (best event):** 100m (467) > 400m (430) > 200m (351)

The 400m's lower 8th-place threshold (791.7 vs. 818–820 for 100m/200m) shifts some athletes toward the one-lap event for nationals purposes despite the 100m being their absolute best.

---

## Cross-Cutting Themes

### Theme A: Absolute strength ≠ competitive opportunity

RQ1A measures **how fast/strong** an athlete is in scoring terms. RQ1C measures **how close** they are to nationals relative to the field. These diverge when:
- An event has a very high 8th-place bar (1500m, 800m, 100m)
- An athlete's secondary event has a much lower nationals threshold (steeple, discus, high jump)
- Gender-specific field depth differs (women's 5000m vs. 1500m)

### Theme B: Gender differences are most pronounced in distance

| Pattern | Men | Women |
|---------|-----|-------|
| Modal best event (distance) | 1500m | 1500m |
| 1500m vs. 5000m (pairwise RQ1A) | 1500m wins | **5000m wins** |
| 1500m vs. 5000m (pairwise RQ1C margin) | 5000m wins | **5000m wins** |
| 800m vs. 1500m (pairwise) | 1500m ~60% | 1500m **~79%** |
| Best steeple clear-rate (distance) | 8.5% | **16.3%** |

### Theme C: Steeplechase is a specialization outlier

- **3%** of best-event assignments
- **Highest nationals clear-rate** among women's distance events
- **Lowest 8th-place threshold** among all men's distance events
- Wins **77–98%** of RQ1C margin comparisons vs. 1500m among dual athletes
- Rarely chosen as absolute best event

Steeplechase represents a **nationals opportunity event** more than a **peak-scoring event** in this dataset.

### Theme D: Multi-event participation is common

| Discipline | Men with 2+ events | Women with 2+ events |
|------------|-------------------|---------------------|
| Sprints | 647 / 1,248 (52%) | 290 / 542 (54%) |
| Distance | 961 / 1,933 (50%) | 424 / 946 (45%) |
| Throws | 112 / 191 (59%) | 73 / 128 (57%) |

Roughly **half** of athletes in sprints, distance, and throws compete in multiple events within their discipline — making pairwise and RQ1C analyses directly relevant to roster decisions.

---

## Hypothesis Summary

| Hypothesis | Result |
|------------|--------|
| **H1A:** Athletes show identifiable event specialization via World Athletics points | ✅ Supported — stable modal events per discipline/gender |
| **H1B:** Nationals top-8 distributions differ by event and are stable year-to-year | ✅ Supported — large event-specific gaps; moderate year-to-year variation |
| **H1C-1:** RQ1A ≠ RQ1C for a meaningful share of athletes | ✅ Supported — 6–35% mismatch depending on discipline |
| **H1C-2:** Men favor 1500m; women favor 5000m in dual-event competitiveness | ⚠️ Partially — women clearly yes; men 1500m modal in RQ1A but 5000m wins pairwise RQ1C |
| **H1C-3:** Steeple high conditional competitiveness, low population impact | ✅ Supported |

---

## Limitations

1. **World Athletics points** are table-based comparability scores, not direct physiological measures.
2. **Single best result** may be influenced by outlier performances (wind, pacing, tactics).
3. **NIRCA club athletes** — findings may not generalize directly to NCAA/NAIA varsity programs.
4. **Nationals round labeling** is inconsistent across years (`Finals` vs. unlabeled); 100m/200m prelim rule applied as specified.
5. **Selection bias** — athletes choose which events to enter; observed patterns reflect both ability and opportunity.
6. **Women's 5000m 2025** had no nationals event, affecting 3-year averages for that event.
7. **Relay events** excluded from RQ1 (addressed in proposed RQ4 extension).

---

## Data & Output Reference

| Analysis | Location |
|----------|----------|
| RQ1A best-event counts & histograms | `{Discipline}_Events_Counting/best_event_counts_{gender}_{year}.txt` |
| RQ1A pairwise comparisons | `{Discipline}_Events_Counting/pairwise_best_event_counts_{gender}_{year}.txt` |
| RQ1B nationals distributions | `RQ1B_Nationals/rq1b_summary_{year}.txt`, `rq1b_distributions_{year}.png` |
| RQ1B 8th-place rankings | `RQ1B_Nationals/rq1b_nationals_8th_place_rankings_by_year_gender.txt` |
| RQ1B 3-year average thresholds | `RQ1B_Nationals/rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt` |
| RQ1C competitiveness analysis | `RQ1C_Competitiveness/rq1c_competitiveness_analysis.txt` |
| Analysis scripts | `analyze_rq1b_nationals.py`, discipline-specific `analyze_*.py` files |
| Research question framework | `Research_Questions.md` |

---

*Generated from NRCD outdoor track analysis, 2024–2026. Companion to the NRCD cross country paper: "Faster Results From A Smarter Schedule."*
