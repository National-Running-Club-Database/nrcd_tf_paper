# Summary Findings — Causal Analysis of Competition Volume

**Question:** If the same athlete raced more within a season, did they improve more (WA points)?
**Data:** 15,128 athlete-season-event_group rows → 7,106 after excluding 1-result seasons (mechanical zero point-jump).
**Full report:** [causal_analysis_report.txt](causal_analysis_report.txt)

---

## 1. Bottom line

Within-athlete fixed-effects models find a **consistently positive association** between
result count and seasonal WA point improvement, even after controlling for opening-day WA
and season. The effect is **not uniform across event groups** — strong and precisely
estimated for Distance and Sprints, but statistically inconclusive (CI spans zero) for
Men's Hurdles, Jumps, and Throws.

## 2. Fixed-effects coefficients (Model A — global)

- Observations: 2,273 | Panel units: 1,029
- **β(result_count) = 9.889** (SE 0.671), within R² = 0.4135
- Cluster bootstrap 95% CI: **[8.105, 11.709]** — excludes zero
- β(first_wa) = -0.503 (SE 0.013) — higher opening-day WA predicts a smaller later jump (regression to the mean / ceiling effect)

**Reading:** each additional race in a season is associated with **~9.9 more WA points**
of seasonal improvement for that same athlete, holding their baseline ability (fixed
effect) and opening-day mark constant.

## 3. Robustness checks

| Model | Setup | β(result_count) | 95% CI / SE | n |
|-------|-------|------------------|-------------|---|
| A — athlete×group FE | within-athlete, bootstrap CI | **9.889** | [8.105, 11.709] | 2,273 (1,029 units) |
| B — pooled OLS (no FE) | naive baseline | 9.871 | SE 0.433 | 7,106 |
| C — paired season deltas | Δresult_count → Δpoint_jump | 9.283 | SE 0.758 | 1,459 season pairs |

The FE, pooled, and paired-delta estimates all cluster tightly around **~9.3–9.9 WA
points per additional race**, which is reassuring — the within-athlete result isn't an
artifact of ignoring athlete ability.

## 4. By event group (Model D, athlete×group FE with bootstrap CI)

| Group | β | 95% CI | n (units) | Verdict |
|-------|---|--------|-----------|---------|
| Women Distance | **16.960** | [11.618, 22.089] | 273 (124) | positive |
| Women Hurdles | **22.020** | [9.383, 34.999] | 67 (30) | positive |
| Men Distance | **12.150** | [8.711, 16.269] | 759 (339) | positive |
| Women Sprints | **9.417** | [6.114, 13.715] | 193 (89) | positive |
| Women Jumps | **7.294** | [2.028, 13.751] | 86 (39) | positive |
| Men Sprints | **7.708** | [4.036, 11.310] | 395 (179) | positive |
| Men Hurdles | 4.430 | [-7.472, 15.025] | 115 (51) | mixed/uncertain |
| Women Throws | 6.796 | [-2.746, 19.572] | 72 (32) | mixed/uncertain |
| Men Throws | 0.591 | [-5.073, 5.133] | 164 (76) | mixed/uncertain |
| Men Jumps | -0.522 | [-8.408, 6.261] | 149 (70) | mixed/uncertain |

Distance and Sprints (both genders) show the clearest, most precisely-estimated positive
effect. Technical events (Hurdles/Jumps/Throws, especially for men) have wide CIs that
include zero — small panel sizes (51–76 units) limit statistical power there.

## 5. Within-season dose–response (descriptive)

Average cumulative WA jump from first-day mark, by meet index (see `plots/`):

| Group | Meet range | Cumulative gain |
|-------|-----------|-----------------|
| Women Distance | 1 → 9 | **+143.1 WA** |
| Men Hurdles | 1 → 7 | +99.6 WA |
| Women Hurdles | 1 → 6 | +90.8 WA |
| Men Distance | 1 → 9 | +83.6 WA |
| Women Sprints | 1 → 11 | +64.8 WA |
| Men Sprints | 1 → 13 | +50.4 WA |
| Men Jumps | 1 → 8 | +44.7 WA |
| Women Jumps | 1 → 8 | +36.7 WA |
| Women Throws | 1 → 8 | +28.1 WA |
| Men Throws | 1 → 9 | +21.6 WA |

All ten gender × event-group curves rise from meet 1 to the last observed meet index,
consistent with (but not proof of) a positive dose–response relationship.

## 5b. Inferential tests on within-athlete deltas (`research/inferential_report.txt`)

Non-parametric complement to the FE regressions, both on the 1,459 within-athlete
season-pair deltas (n=1,089 pairs with a nonzero race-count contrast for the Wilcoxon
test); cluster-bootstrapped by athlete×event-group unit, seed 2026:

- **Paired Wilcoxon signed-rank** (higher- vs. lower-volume season point jump, same
  athlete×event group): median difference **+4.0 WA points** [95% CI 0.0, 13.0],
  *p* = 5.9×10⁻¹⁸ (one-sided). Significant by gender×group for Men/Women Distance,
  Men/Women Sprints, and Women Hurdles/Throws; **not** significant for Men
  Hurdles/Jumps/Throws or Women Jumps — the same technical-event groups with wide FE
  CIs above.
- **Spearman ρ(Δresult_count, Δpoint_jump)** = **0.251** [95% CI 0.196, 0.310],
  *p* = 1.9×10⁻²² (n=1,459). By group, ρ ranges from 0.51 (Women Sprints) to
  essentially 0 for Men Hurdles/Jumps/Throws — consistent with the FE model's
  CI-includes-zero groups.
- **Dose–response cluster-bootstrap CIs**: e.g. Men Distance rises from 0.0 [0.0, 0.0]
  at meet 1 to 83.6 [60.8, 108.1] WA points at meet 9; Women Distance rises to 143.1
  [83.0, 219.9] at meet 9 — CIs exclude 0 by the final observed meet index in every
  gender×group slice with enough athlete-meet observations.

New research figures: `plots/fe_beta_forest_plot.png` (FE β by group, forest plot) and
`plots/delta_dose_response_scatter.png` (Δresult_count vs Δpoint_jump scatter with
Spearman ρ). The `plots/dose_response_*.png` figures now include a shaded 95%
cluster-bootstrap CI band.

## 6. Causal credibility assessment

The analysis explicitly separates **associative evidence** from a **causal claim**:

- ✅ β(result_count) > 0 with CI excluding 0 in the global FE model and most event groups
- ✅ Dose–response curves rise with meet index in every slice
- ⚠️ Remaining threats to causal interpretation (from the report):
  - Coaches may assign more races to athletes who are *already* trending up within the season (reverse causation / selection).
  - Season context (team dynamics, health, weather) is not fully observed.
  - Relay assignments can drive volume for an athlete's non-primary events.

**Verdict:** the fixed-effects estimate (~9.9 WA points per additional race) is the best
available answer in this observational dataset, but should be framed as **within-athlete
associative evidence**, not a randomized-experiment causal effect.

## 7. Limitations

1. **Not a randomized experiment** — race volume is chosen by athletes/coaches, not assigned; unobserved confounders (form, health, coaching strategy) can bias estimates in either direction.
2. **Small per-group panels** for technical events (as few as 30–76 athlete×group units) — CIs are wide and some effects are statistically inconclusive rather than "null."
3. **Homoskedastic OLS standard errors** in Models A/B; only Model A/D use a cluster bootstrap, which is the more defensible inference procedure here.
4. **Mechanical exclusion** of 1-result seasons removes ~53% of raw rows (15,128 → 7,106), which could bias the panel toward more competitive/available athletes.
5. **Point jump ceiling effect** (β(first_wa) ≈ -0.5) means athletes who start strong have structurally less room to "jump," which is controlled for but worth keeping in mind when comparing across athletes of different starting ability.

## 8. Reproducibility

```bash
cd causal_analysis
python main.py all
```

Requires `number_of_events_question/athlete_point_jumps_by_season.csv` to exist (see
[README.md](README.md) for the dependency chain). Bootstrap draws: 500, seed: 42.
