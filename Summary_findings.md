# Summary Findings — NRCD Track Specialization Paper

**Project:** Event specialization in collegiate club track & field  
**Target venue:** *Journal of Quantitative Analysis in Sports*  
**Data:** National Running Club Database, outdoor (and indoor) seasons 2024–2026  
**Detailed RQ plan:** [Research_Questions.md](Research_Questions.md)

**Scoring framing**

| Role | Metrics |
|------|---------|
| Scientific | Gardner–Purdy + Mercier (1999) |
| Sports / coaching | World Athletics Points + VDOT |

Cross-metric artifacts: [`scoring/Summary_findings.md`](scoring/Summary_findings.md), [`scoring/output/`](scoring/output/). Baseline narrative below uses **WA** unless noted; Purdy/Mercier/VDOT sensitivity lives under `improved_rq1_outputs/by_metric/` and `number_of_events_question/by_metric/`.

This note synthesizes module-level `Summary_findings.md` files. Prefer those documents (and their `research/` reports) for tables, CIs, and figures.

---

## 1. Central conclusions

1. **Specialization is measurable with commensurate points (RQ1A).** Among multi-event distance athletes, the **1500m** is the modal absolute best event for men (~42%) and women (~46%) under WA; pairwise patterns show a clear **gender asymmetry**: men who also run 5000m still tend to peak at 1500m; women dual 1500/5000 athletes more often peak at **5000m**. Best-event identity agrees **~86%** (WA vs Purdy) and **~82%** (WA vs Mercier) across outdoor slices.
2. **Absolute best ≠ most competitive path to nationals (RQ1C).** Agreement between RQ1A and RQ1C drops to **~76% (men)** and **~69% (women)** in distance. Steeplechase is rarely the absolute best event (~3%) but offers **high nationals clear-rates** and low 8th-place bars — especially for women (~16% clear).
3. **Nationals depth is event-specific (RQ1B).** Middle-distance and sprint 8th-place thresholds sit near **~790–830 WA**; field/hurdles paths often combine **lower thresholds** with **higher clear-rates**.
4. **Racing more is associated with larger within-season WA jumps.** Cross-sectionally, Spearman ρ(result count, point jump) ≈ **0.62**. Within athletes (FE), each additional result associates with ~**+9.9 WA points** of seasonal jump (cluster bootstrap 95% CI ≈ [8.1, 11.7]); paired Wilcoxon on higher- vs lower-volume seasons: median +4.0 WA (*p* ≪ 0.001). Effects are clearest in **sprints/distance**, weaker/uncertain in several technical groups. Volume analyses can be re-run with `--metric vdot|purdy|mercier`.
5. **Feature-routed time models beat pooled formulas on external D1 validation.** Season-PB MedAPE **1.55%** [1.34, 1.89] (feature) vs **2.91%** [2.21, 3.48] (pooled); paired Wilcoxon *p* = 2.5×10⁻¹⁰. Sprint pairs transfer well; **steeplechase / long-distance** transfers remain high-uncertainty.
6. **Corrected steeplechase WA scoring** (see `new_steeplechase_data/`) is required for distance RQ credibility; treat legacy Relays distance steeple points as superseded.

---

## 2. Results by research block

### RQ1 — Specialization & competitiveness (outdoor)

| Sub-question | Primary modules | Headline |
|--------------|-----------------|----------|
| RQ1A best event | `non_relays_findings`, `relays_findings`, `new_steeplechase_data` | 1500m modal; steeple ~3%; women 1500/5000 favor 5000 pairwise |
| RQ1B nationals top-8 | same | High bars in 800/1500/sprints; lower bars + higher clear-rates in many field/hurdles |
| RQ1C competitiveness | same | Material RQ1A≠RQ1C mismatch (esp. distance); steeple as opportunity event |

See `non_relays_findings/Summary_findings.md` and `improved_rq1_outputs/improved_findings.md` for full tables. Prefer **new_steeplechase_data** for distance/steeple numbers.

### Indoor & indoor↔outdoor

`indoor_analysis` mirrors outdoor RQ1A structure indoors and studies opener WA relationships outdoor. See that module’s `Summary_findings.md` for discipline-specific indoor best-event patterns and interplay results.

### Volume / improvement

| Design | Module | Claim strength |
|--------|--------|----------------|
| Cross-sectional bins + Spearman | `number_of_events_question` | Associative (selection: committed athletes race more) |
| Athlete×group FE + paired deltas + dose–response | `causal_analysis` | Within-athlete association; **not** randomized causal proof |

### Time models & external validation

| Layer | Module | Note |
|-------|--------|------|
| Club fit / FI / point bands | `time_models`, `new_steeplechase_data/.../Feature_Importance_*` | Training/CV metrics |
| External D1 validation | `test_dataset_time_models` | Inferential suite + plots under `output/model_validation/research/` |

### Coaching narratives

`coaches_analysis` (and steeple-aware twin under `new_steeplechase_data/coaches_analysis/`) translates RQ1B/C into roster / scoring-focus guidance. Prefer the steeple-corrected coaches folder for distance.

---

## 3. Statistical practices (repo-wide)

| Practice | Where applied |
|----------|----------------|
| Bootstrap / cluster-bootstrap CIs | Causal FE β; dose–response; time-model MedAPE; volume bins |
| Wilcoxon (paired / signed-rank) | Causal volume contrast; external validation bias & feature-vs-pooled |
| Spearman ρ + *p* | Volume↔jump; pred↔actual in validation |
| Clopper–Pearson rate CIs | External validation tolerance hit rates |
| Event-pair / event-group stratification | Time models (seconds not pooled across 100m–5000m) |

Modules with dedicated `research/` (or equivalent) layers: `causal_analysis`, `number_of_events_question`, `indoor_analysis`, `relays_findings`, `new_steeplechase_data`, `time_models`, `test_dataset_time_models`, plus improved RQ1 statistical text under `non_relays_findings/improved_rq1_outputs/`.

---

## 4. Limitations (paper-facing)

1. **Observational volume effects** — coaches may assign more races to athletes already improving; FE reduces but does not eliminate that threat.
2. **Nationals definition** — top-8 construction depends on prelim/final filtering rules; sensitivity analyses remain advisable.
3. **WA points as ability proxy** — nonlinear scoring and event-specific participation shape both specialization and thresholds. Dual-framing sensitivity (Purdy, Mercier 1999, VDOT) is reported in `scoring/output/`; Purdy/VDOT do not cover field events.
4. **Mercier 1999 vs modern tables** — scientific Mercier scores use the documented 1995–1998 linear reconstruction, not proprietary Mercier–Rioux tables.
5. **Small *n* in some pairs/bands** — especially steeple and external-validation distance pairs; report CIs, not point estimates alone.
6. **Distribution shift** — club-trained time models vs NCAA D1 external set; positive prediction bias is expected.
7. **Men’s external validation only so far** — `documents/women/` reserved; women’s external holdout not yet run.
7. **Non-independence** — same athletes appear in multiple pairs/bands; naive *p*-values can be optimistic.

---

## 5. How to reproduce

```bash
python main.py list
python main.py run non_relays_findings all
python main.py run new_steeplechase_data all
python main.py run causal_analysis all
python main.py run number_of_events_question research
python main.py run test_dataset_time_models all
```

Full re-runs can be slow (PDF scraping, large CSVs). Prefer `research` subcommands when only inferential layers need refreshing.
