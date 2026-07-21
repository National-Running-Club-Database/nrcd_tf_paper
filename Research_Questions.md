# Research Questions & Analysis Plan
## Event Specialization in Collegiate Club Track & Field

**Target venue:** *Journal of Quantitative Analysis in Sports* (JQAS)  
**Data source:** National Running Club Database (NRCD), outdoor track seasons 2024–2026  
**Companion work:** *Faster Results From A Smarter Schedule* (NRCD cross country; standardized performance & scheduling)

### Scoring systems (locked framing)

| Role | Metrics | Notes |
|------|---------|-------|
| **Scientific** | Gardner–Purdy points + Mercier (1999) | Purdy: Portuguese / Hoffman implementation. Mercier: documented linear `Points = A·x + B` reconstruction (not Mercier–Rioux). |
| **Sports / coaching** | World Athletics Points + VDOT | WA 2025 outdoor tables; Daniels–Gilbert Oxygen Power VDOT. |

Shared implementation: [`scoring/`](scoring/). Field events: Purdy/VDOT are undefined → use Mercier (scientific) and WA (sports). Primary published tables remain WA-based; sensitivity under Purdy/Mercier/VDOT is reported via `--metric` / `python main.py compare-metrics`.

---

## 1. Paper framing (JQAS-oriented)

### 1.1 Central problem
Collegiate club athletes frequently compete in multiple track & field events within a discipline (e.g., 800m, 1500m, 5000m), but roster and lineup decisions are often made without quantitative evidence about **where an athlete’s relative strength lies** or **which event offers the best path to championship competitiveness**.

### 1.2 Proposed contribution
This paper introduces a reproducible, multi-metric scoring framework for **event specialization analysis**. The work extends the NRCD research program from cross country (environment-adjusted times) to outdoor track, with explicit attention to:

- **Individual-level specialization** (best event identification)
- **Population-level competitiveness** (nationals-caliber score distributions)
- **Gender differences** in specialization patterns
- **Robustness across scientific vs sports scoring systems**
- **Robustness extensions** (relays, indoor/outdoor, cross country background)

### 1.3 Why JQAS
JQAS emphasizes original statistical thinking applied to difficult sports problems. This manuscript should go beyond descriptive counts by:

1. Formalizing **measurement** (commensurate points / VDOT as latent ability proxies under dual framing)
2. Testing **hypotheses** with uncertainty (not only point estimates)
3. Addressing **selection bias** (who competes in which events, who reaches nationals)
4. Providing **actionable inference** for lineup and development decisions

---

## 2. Core research questions

### RQ1 — Event specialization and competitiveness

**Overarching question:** *Among collegiate club track athletes, how does World Athletics scoring reveal event specialization, and which events offer the greatest opportunity for championship-level competitiveness?*

RQ1 is decomposed into three sub-questions:

---

#### RQ1A — Which event are athletes relatively best at? ✅ **COMPLETE (outdoor)**

**Question:** For athletes competing in multiple events within a discipline, which event corresponds to their highest World Athletics point value?

**Status:** Completed for outdoor track 2024, 2025, 2026, and 2024–2026 combined, by gender and discipline:
- Sprints (100m, 200m, 400m)
- Distance (800m, 1500m, 3000m Steeplechase, 5000m)
- Hurdles, Jumps, Throws (parallel analyses)

**Method (implemented):**
- **Overall best event:** Assign each athlete the event associated with their single highest gender-specific World Athletics score across target events in the analysis window.
- **Pairwise best event:** Among athletes with ≥1 result in both events of a pair, compare each athlete’s top score in Event A vs. top score in Event B.

**Key preliminary findings:**
- Men: 1500m is the modal best distance event (~42%); pairwise 1500m beats 800m and 5000m among dual-event athletes.
- Women: 1500m leads overall (~46%), but dual 1500m/5000m athletes favor 5000m — a clear gender asymmetry.
- Steeplechase is rarely the best event (~3%) despite meaningful participation.

**JQAS upgrade needed:** Move from descriptive counts to **mixed-effects or multinomial models** for best-event assignment with covariates (grade, seasons competed, team, event-entry count).

---

#### RQ1B — Nationals score distributions by event and season ✅ **COMPLETE**

**Question:** For each season, what is the distribution of World Athletics scores for the **top 8 finishers** at the national championship in each event?

**Status:** Completed via `analyze_rq1b_nationals.py`. Outputs in `RQ1B_Nationals/` (summaries, CSV, distribution plots, 8th-place rankings).

**Operational definition (proposed):**
- Filter: `nationals == True` (dataset column; equivalent to NIRCA nationals indicator)
- Unit of analysis: one observation per athlete per event final
- Ranking: assign place within `(season, gender, running_event_id, meet_id)` by **competition result** (time for running events; mark for field events), then retain places 1–8
- Output: per event, per season — mean, SD, median, IQR, and full distribution (violin/box plots) of World Athletics points for places 1–8

**Hypotheses:**
- H1B-1: The score gap between 1st and 8th varies by event (sprints vs. distance vs. field).
- H1B-2: Year-to-year nationals score distributions are stable after accounting for cohort quality.
- H1B-3: Women's nationals score distributions show greater relative dispersion in some distance events (linked to participation volatility).

**Critical methodological issues to resolve before analysis:**

| Issue | Risk | Mitigation |
|-------|------|------------|
| **No explicit `place` column** | Cannot directly identify top 8 | Rank within meet-event by `result_time` / mark; validate against published NIRCA results where possible |
| **Multiple nationals rows per event** | 2024 men's 1500m has 246 nationals rows — likely heats, sections, or duplicates | Define filtering rules: finals only, best-of-day, or highest round; document exclusion of prelims |
| **`nationals` may include non-championship meets** | Mislabeled population | Cross-check `meet_id`, `start_date`, and meet metadata; sensitivity analysis with strict nationals definition |
| **World Athletics points vs. place** | Ranking by time ≠ ranking by points if scoring nonlinearities differ | Report both place-by-time and place-by-points; discuss any divergence |
| **Relays excluded initially** | RQ1B for relays requires different unit (team, split aggregation) | Analyze individual events first; add relay extension in RQ3 |

**Deliverables:**
- Tables: top-8 score summary statistics by event × season × gender
- Figures: faceted distributions (2024, 2025, 2026)
- Statistical test: Kruskal-Wallis or Bayesian hierarchical model comparing score distributions across events

---

#### RQ1C — Which event should athletes pursue to be most competitive? ✅ **COMPLETE (threshold method)**

**Question:** Conditional on an athlete’s ability profile, which event maximizes their probability of reaching nationals-caliber performance (e.g., top-8 score threshold)?

**Conceptual distinction from RQ1A:**
- **RQ1A** asks: *Where is the athlete already strongest?* (descriptive specialization — highest absolute World Athletics personal best)
- **RQ1C** asks: *Where would they be most competitive relative to the field?* (normative — largest margin above the nationals 8th-place threshold)

These diverge when an athlete’s best event is also the deepest/hardest event nationally.

**Method implemented (2024–2026 combined):**
1. **Threshold:** 3-year average nationals 8th-place World Athletics score per event × gender (`RQ1B_Nationals/rq1b_nationals_8th_place_rankings_3yr_avg_by_gender.txt`).
2. **Athlete personal best** per event across all outdoor results in the analysis window.
3. **RQ1C assignment:** Event with the **largest margin** (personal best − 8th-place threshold). If no event clears the bar, assign the event with the **smallest deficit** (closest to nationals).
4. **Comparison:** RQ1A vs. RQ1C agreement rate among multi-event athletes (2+ events in discipline).

**Full output:** `RQ1C_Competitiveness/rq1c_competitiveness_analysis.txt`

---

**Findings**

**1. RQ1A and RQ1C often disagree — especially in distance**

| Discipline | Gender | Multi-event athletes | RQ1A = RQ1C agreement |
|------------|--------|----------------------|----------------------|
| Sprints | Men | 647 | 94.0% |
| Sprints | Women | 290 | 91.4% |
| Distance | Men | 961 | **75.8%** |
| Distance | Women | 424 | **68.6%** |
| Hurdles | Men | 93 | 66.7% |
| Jumps | Men | 106 | 75.5% |
| Throws | Women | 73 | 65.8% |

→ **H1C-1 supported:** 6–35% of multi-event athletes should pursue a different event than their absolute best to maximize nationals competitiveness.

**2. Distance: men should lean 1500m; women should lean 5000m in dual-event choices**

Pairwise competitiveness (margin above 8th-place threshold, athletes in both events):

| Pairing | Men | Women |
|---------|-----|-------|
| 1500m vs. 5000m | 5000m: 228 / 1500m: 209 | **5000m: 79 / 1500m: 65** |
| 800m vs. 1500m | 1500m: 361 / 800m: 225 | 1500m: 178 / 800m: 90 |

→ **H1C-2 partially supported:** Women's 5000m beats 1500m among dual-event athletes (79 vs. 65). Men's 1500m is the modal RQ1C recommendation (669 athletes) but **loses head-to-head to 5000m** among athletes who compete in both (228 vs. 209).

**3. Steeplechase: high clear-rate, low absolute bar — but RQ1C rarely recommends it**

| | Men | Women |
|--|-----|-------|
| % of competitors clearing 8th-place threshold | 8.5% (24/281) | **16.3%** (22/135) |
| Athletes with RQ1C = steeple | 209 | 122 |
| Athletes with RQ1A = steeple | 62 | 29 |

Among dual 1500m/steeple athletes, steeple wins the competitiveness margin **132–75 (men)** and **75–6 (women)** — athletes are much closer to the nationals bar in steeple. Yet few athletes have steeple as their *absolute* best event.

→ **H1C-3 supported:** Steeplechase offers a **lower nationals threshold** (456 men / 511 women) and the **highest clear-rate among women's distance events** (16.3%), but small participation and lower absolute scoring limit population-level impact.

**4. Population-level: “easiest” nationals paths differ from highest-scoring events**

Events with the **highest % of season competitors clearing the 8th-place bar**:

| Men | Women |
|-----|-------|
| 110m Hurdles (28.4%) | High Jump (37.3%) |
| High Jump (27.9%) | 400m Hurdles (34.8%) |
| Triple Jump (23.0%) | Triple Jump (31.6%) |
| … | … |
| 1500m (**4.3%**) | 800m (6.3%) |

Sprints and middle-distance track events have **high 8th-place thresholds** (800m–829 men) but **low clear-rates** (4–10%). Technical/field events have **lower thresholds** and **higher clear-rates** — a better nationals *path* for many athletes even when absolute WA scores are lower.

**5. Practical coaching recommendations (distance focus)**

| If athlete profile is… | RQ1A says… | RQ1C says pursue… |
|------------------------|------------|-------------------|
| Men, strong miler, also runs 5K | 1500m | Often still 1500m, but check 5K margin — 5000m competitive for 228 dual athletes |
| Women, strong miler, also runs 5K | 1500m | **5000m** more often (79 vs. 65) |
| Either, runs 1500m + steeple | 1500m | **Steeplechase** if margin to nationals bar is larger (very common) |
| Sprinter doubling 100/200 | 100m | Usually same (94% agreement); 400m recommended when margin favors longer sprint |

---

**Hypothesis verdicts:**
- **H1C-1:** ✅ Supported (material mismatch rates, especially distance)
- **H1C-2:** ⚠️ Partially supported (women yes; men 1500m modal but 5000m wins pairwise)
- **H1C-3:** ✅ Supported (steeple high conditional clear-rate, low RQ1A share)

**Remaining for JQAS:**
- Logistic regression / hierarchical model for \( P(\text{nationals top-8} \mid \text{event}) \) with bootstrap CIs (Phase 3 upgrade from threshold-only method).

---

## 3. Extension research questions

### RQ2 — Cross country background and outdoor distance specialization

**Question:** Can we detect systematic differences in outdoor distance specialization and World Athletics profiles between athletes who did and did not compete in cross country?

**Motivation:** Links this track paper to the existing NRCD cross country work (*Faster Results From A Smarter Schedule*). XC participation may indicate aerobic base, durability, and different developmental paths.

**Design:**
- **Treatment:** Athlete has ≥1 NRCD cross country result in the academic year preceding outdoor season (or within a defined rolling window)
- **Control:** No XC results in that window
- **Outcomes:** Best event (RQ1A), nationals score (RQ1B), competitiveness index (RQ1C), outdoor race frequency

**Methods:**
- Propensity score matching or inverse probability weighting to address self-selection into XC
- Difference-in-differences across seasons if XC participation changes within athlete
- Report gender-stratified effects

**Data requirement:** ⚠️ **Not currently in track event-counting folders.** Requires linking `athlete_id` to NRCD cross country dataset used in the XC paper.

**Threats to validity:**
- XC runners are a selected subset (fitness, commitment, coaching philosophy)
- Team-level confounding (programs that emphasize XC may also emphasize 5K/10K)

---

### RQ3 — Indoor vs. outdoor as distinct competitive seasons

**Question:** Are indoor and outdoor track distinct seasons in terms of World Athletics score patterns and event specialization?

**Sub-questions:**
- RQ3A: Do athletes’ best events differ between indoor and outdoor?
- RQ3B: Are indoor and outdoor World Athletics distributions for the same nominal event (e.g., 800m) statistically separable?
- RQ3C: Does indoor performance predict outdoor nationals competitiveness?

**Methods:**
- Paired within-athlete comparisons (indoor best vs. outdoor best)
- Mixed-effects models with season type (indoor/outdoor) as fixed effect, athlete as random effect
- Correlation / rank stability across seasons

**Data requirement:** ⚠️ **Indoor CSV files not yet present in this repository.** Analysis is planned but blocked until indoor NRCD extracts are added.

**Anticipated finding to test:** Indoor marks are generally faster for sprints/shorter events (banked tracks, different scheduling), so **raw World Athletics comparability across indoor/outdoor may still be valid** (scoring tables are event-specific) but **specialization patterns may shift** (e.g., more 400/600 specialists indoors).

---

### RQ4 — Robustness of RQ1 when relays are included

**Question:** How does event specialization change when relay events are incorporated into RQ1?

**Motivation:** Relays are team events with split times and different strategic value. Coaches may place athletes in relays based on depth charts rather than individual best events.

**Relay events (from `running_event.csv`):**
- 4×100m, 4×200m, 4×400m, 4×800m, SMR, DMR, etc.

**Methodological challenges:**

| Challenge | Implication |
|-----------|-------------|
| Multiple `athlete_id` columns (`athlete_id` … `athlete_id_4`) | Must expand relay rows to leg-level athlete records |
| `relay_split` vs. full relay time | Use leg split World Athletics points if available; do not assign full relay time to one athlete |
| Same athlete in multiple legs/meets | Deduplicate; take best leg score per event type |
| Relay is not an individual “specialization” in the same sense | Consider two frameworks: (i) include relay as an event category, (ii) treat relay as team outcome separate from individual RQ1 |

**Recommended design:**
- **Primary analysis:** Individual events only (current RQ1A)
- **Sensitivity analysis:** Add best relay leg as an additional “event” and quantify how many athletes’ best-event label changes
- **Separate team-level RQ:** Do teams maximize points by aligning relay assignments with RQ1A best events?

---

### RQ5 — Indoor vs. outdoor robustness for RQ1 specifically

**Question:** What differences emerge in RQ1 (best event identification) when indoor and outdoor results are analyzed separately vs. pooled?

**Planned comparisons:**
1. Outdoor-only best event (current analysis) ✅
2. Indoor-only best event
3. Combined indoor+outdoor best event (athlete’s top score across both seasons)
4. Agreement rate: how often indoor best event = outdoor best event for the same athlete in the same calendar year

**Expected error mode:** Pooling indoor and outdoor without season indicators may **artificially inflate** best-event counts for events with more competition opportunities (e.g., 800m indoors and outdoors).

---

## 4. Unified analysis plan

### Phase 1 — Descriptive specialization (complete)
| Task | Status |
|------|--------|
| RQ1A overall best-event counts by discipline | ✅ |
| RQ1A pairwise comparisons by discipline | ✅ |
| Gender-stratified summaries | ✅ |
| Seasonal (2024/2025/2026) + combined panels | ✅ |
| Trend synthesis | ✅ |

### Phase 2 — Nationals competitiveness ✅ **COMPLETE**
| Task | Status |
|------|--------|
| Validate `nationals` flag and meet structure | ✅ |
| Define finals-only top-8 extraction rule | ✅ |
| RQ1B distributions by event × season × gender | ✅ |
| Publish threshold table for RQ1C | ✅ |

### Phase 3 — Normative specialization (RQ1C) ✅ **COMPLETE (threshold method)**
| Task | Status |
|------|--------|
| Build top-8 nationals score thresholds from RQ1B | ✅ |
| Athlete-level competitiveness classification | ✅ |
| Compare RQ1A vs. RQ1C agreement rates | ✅ |
| Hierarchical logistic models for nationals probability | ⬜ (JQAS upgrade) |

### Phase 4 — Extensions & robustness
| Task | Status | Dependency |
|------|--------|------------|
| RQ2 XC vs. non-XC distance runners | ⬜ | XC dataset link |
| RQ3 indoor/outdoor season comparison | ⬜ | Indoor data |
| RQ4 relay-inclusive RQ1 sensitivity | ⬜ | Relay leg parsing |
| RQ5 indoor/outdoor RQ1 comparison | ⬜ | Indoor data |

### Phase 5 — JQAS manuscript assembly
| Task | Status |
|------|--------|
| Methods section with formal notation | ⬜ |
| Reproducible analysis pipeline (GitHub + Zenodo) | ⬜ |
| Limitations & external validity (NIRCA → NCAA) | ⬜ |
| Submission formatting per JQAS author guidelines | ⬜ |

---

## 5. Statistical methods roadmap (JQAS standard)

To elevate the paper from exploratory analytics to JQAS-level inference:

1. **Multinomial logistic regression** — Best event ~ gender + grade + event participation counts + season
2. **Bayesian hierarchical models** — Athlete random effects for repeated seasons (following JQAS precedents on elite performance modeling)
3. **Bootstrap confidence intervals** — For pairwise proportions and best-event shares
4. **Multiple comparisons control** — Across six pairwise tests per discipline (Benjamini-Hochberg)
5. **Sensitivity analyses** — Documented in Section 5 (relays, nationals definition, tie-breaking, indoor/outdoor)
6. **Temporal validation** — Train competitiveness thresholds on 2024–2025; validate on 2026

---

## 6. Known errors, limitations, and mitigations

### 6.1 Data quality (documented in project history)
| Error | Impact | Resolution |
|-------|--------|------------|
| Steeplechase `World_Athletics_Points` = 0 for all men's 2026 results | Steeplechase never won pairwise comparisons | **Fixed** in updated CSVs; rerun analyses |
| Column name `nationals` vs. `nirca_nationals` | Confusion in query specs | Standardize on `nationals` in code; note alias in paper |
| No finish `place` column | RQ1B top-8 requires derived ranking | Define ranking algorithm; validate on published results |

### 6.2 Methodological limitations
- **World Athletics points are not performance in absolute physiological terms** — they are table-based comparability scores tied to elite reference performances.
- **Best single result ≠ expected performance** — one-off outliers (wind, pacing, tactical races) can misclassify specialization; consider using seasonal top-k mean or trimmed max.
- **Tie-breaking in overall best event** — fixed priority order (800m → 1500m → steeple → 5000m) affects rare ties; pairwise analysis reports ties separately (preferred).
- **Selection bias** — athletes choose which events to enter; observed best event reflects both ability and opportunity.
- **NIRCA generalizability** — findings may not transfer directly to NCAA/NAIA varsity roster constraints.

### 6.3 Recommended improvements before submission
1. Replace single-best-result rule with **robust ability estimate** (e.g., best 2-of-3, or 90th percentile of seasonal scores).
2. Add **uncertainty bands** to all bar charts and pairwise proportions.
3. Pre-register nationals top-8 extraction rules before running RQ1B/RQ1C.
4. Link XC dataset for RQ2 rather than treating as exploratory footnote.
5. Include **preregistered hypothesis registry** for gender interaction effects (central to JQAS rigor).

---

## 7. Proposed manuscript outline (JQAS)

1. **Introduction** — Event specialization problem; NRCD; connection to XC scheduling paper
2. **Related work** — Performance modeling, World Athletics scoring, gender parity in running research
3. **Data** — NRCD outdoor track 2024–2026; disciplines, events, gender, nationals flag
4. **Methods** — World Athletics scoring; best-event definitions; pairwise comparison; nationals threshold; extensions
5. **Results**
   - 5.1 RQ1A: Specialization patterns (sprints, distance, hurdles, jumps, throws)
   - 5.2 RQ1B: Nationals top-8 score distributions
   - 5.3 RQ1C: Competitive event recommendations
   - 5.4 Gender interactions
   - 5.5 Robustness: relays, indoor/outdoor, XC background
6. **Discussion** — Coaching implications; differences from varsity track; alignment with XC scheduling findings
7. **Limitations & future work**
8. **Conclusion**
9. **Data & code availability** — Zenodo + GitHub (mirror XC paper pattern)

---

## 8. Research question summary table

| ID | Question | Type | Status | Data ready? |
|----|----------|------|--------|-------------|
| **RQ1A** | Which event are athletes best at (World Athletics)? | Descriptive / classification | ✅ Complete (outdoor) | Yes |
| **RQ1B** | Distribution of nationals top-8 World Athletics scores by event & season? | Descriptive / distributional | ✅ Complete | Yes |
| **RQ1C** | Which event should athletes pursue to be most competitive? | Normative / predictive | ✅ Complete (threshold) | Yes |
| **RQ2** | XC vs. non-XC distance runner differences? | Causal / comparative | ⬜ Planned | Needs XC link |
| **RQ3** | Are indoor & outdoor distinct seasons (World Athletics)? | Comparative / paired | ⬜ Planned | Needs indoor data |
| **RQ4** | How do relays change RQ1? | Robustness | ⬜ Planned | Yes (with parsing) |
| **RQ5** | Indoor vs. outdoor differences in RQ1? | Robustness | ⬜ Planned | Needs indoor data |

\*RQ1B/RQ1C require operational clarification of nationals finals and top-8 ranking before implementation.

---

## 9. Immediate next steps

1. ~~**RQ1B:** Nationals top-8 distributions and 8th-place threshold tables~~ ✅
2. ~~**RQ1C:** Competitiveness analysis and RQ1A vs. RQ1C mismatch rates~~ ✅
3. **JQAS upgrade:** Hierarchical logistic models with bootstrap CIs for RQ1C nationals probability.
4. **Ingest indoor NRCD extracts** to unlock RQ3 and RQ5.
5. **Link XC athlete IDs** to unlock RQ2.
6. **Upgrade RQ1A inference** with multinomial models for JQAS submission quality.

---

*Last updated: July 2026*  
*Authors: NRCD research team (track & field analytics extension)*
