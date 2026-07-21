# Summary Findings — Number Of Events Question

**Question:** Does competing in more races associate with a bigger seasonal WA
improvement, and when do athletes actually peak relative to their schedule?
**Data:** combined relay-inclusive deduplicated dataset — 18,153 results → 15,128
athlete-season-event_group records (7,106 with ≥2 results), outdoor 2024–2026, March 1+.
**Full reports:** [point_jump_findings.txt](point_jump_findings.txt),
[best_event_proportion_report.txt](best_event_proportion_report.txt)

---

## 1. Point jump vs. competition volume (descriptive)

Every gender × event_group × season slice shows **higher average point jump in
higher-result-count bins** than in the 1-result bin (which is 0.0 by construction). Example
end-to-end trends (lowest → highest plotted bin, ≥10 athletes/bin):

| Slice | Season | Bins | Trend |
|-------|--------|------|-------|
| Men Distance | 2025 | 1 → 7 results | **+98.5 WA** (n=10 at high end) |
| Men Hurdles | 2026 | 1 → 4 results | **+108.6 WA** (n=14) |
| Women Distance | 2024 | 1 → 7 results | +97.6 WA (n=10) |
| Men Distance | 2024 | 1 → 9 results | +92.3 WA (n=13) |
| Women Hurdles | 2024 | 1 → 6 results | +92.6 WA (n=12) |
| Men Sprints | 2026 | 1 → 6 results | +66.3 WA (n=27) |
| Women Sprints | 2026 | 1 → 5 results | +67.2 WA (n=18) |

This pattern holds consistently across all 5 event groups × 2 genders × 3 seasons in
`point_jump_findings.txt` — no slice shows a declining trend.

**Important caveat directly from the source report:** athletes with exactly 1 result have
point jump = 0 *by definition* (first-day WA = season max), so part of every "increasing
with volume" trend reflects this structural floor rather than a pure training effect.
The rigorous, confound-controlled version of this question is answered in
[`../causal_analysis/Summary_findings.md`](../causal_analysis/Summary_findings.md), which
finds a positive within-athlete effect (β ≈ 9.9 WA points/race) even after removing this
structural artifact and controlling for opening WA.

## 2. When do athletes hit their season best? (best-event proportion)

Definition: `best_meet_index / total_results_in_season` (1.0 = peaked at the very last
meet of the season).

### All seasons combined, including 1-result athletes (n = 15,128)

| Group (men) | n | Avg proportion | Peaked in final 25% |
|---|---|---|---|
| Hurdles | 1,426 | 0.908 | 82.5% |
| Jumps | 1,545 | 0.893 | 79.8% |
| Throws | 1,512 | 0.889 | 79.3% |
| Sprints | 2,226 | 0.846 | 73.1% |
| Distance | 3,298 | 0.835 | 69.5% |

Women show a similar pattern (Hurdles 0.900 avg / 80.9% late-peak down to Distance 0.850 / 72.1%).

### Multi-meet seasons only, ≥2 results (n = 7,106) — primary coaching read

Once athletes with only 1 meet are excluded, the picture is more balanced: roughly
**half of athletes peak in the final quarter of their schedule, and 40%+ peak by the
midpoint** — higher race volume does not automatically mean "peak and taper," in this
dataset.

| Group (cross-gender) | n | Avg proportion | Median | Late-peak (final 25%) |
|---|---|---|---|---|
| Hurdles | 767 | 0.728 | 0.667 | 48.4% |
| Sprints | 1,779 | 0.715 | 0.714 | 50.0% |
| Distance | 2,742 | 0.713 | 0.667 | 46.7% |
| Jumps | 931 | 0.712 | 0.667 | 46.1% |
| Throws | 887 | 0.693 | 0.667 | 43.6% |

Men Distance specifically: n=1,879, avg proportion 0.711, 46.5% peak in the final 25% of
meets. Men Sprints: n=1,198, avg proportion 0.714, 50.0% peak in the final 25%.

### By number of meets in season (men, selected)

Peaking is not monotonic in schedule length — e.g., Men Sprints: 2 meets → avg proportion
0.753 (median 1.000); 5 meets → avg proportion drops to 0.645 (median 0.600); 9 meets →
back up to 0.698 (median 0.778). There is no clean "more meets = earlier/later peak" rule
in the raw proportions; see `best_event_proportion_report.txt` §B for full breakdowns.

## 2b. Inferential tests (`research/inferential_report.txt`)

Bootstrap 95% CIs and Spearman rank correlations layered on top of the descriptive bins
above (2024–2026 pooled, cross-sectional — not within-athlete; see caveat below):

- **Spearman ρ(result_count, point_jump)**, all genders/groups pooled: **0.621**
  [95% CI 0.611, 0.630], *p* < 1×10⁻³⁰⁰, n = 15,128. Positive and significant (ρ =
  0.57–0.67) in every one of the 10 gender×event_group slices.
- **Bootstrap CIs on mean point jump by bin** (2024–2026 pooled, e.g. Men Distance):
  1 result → 0.0 [0.0, 0.0] WA; 9 results → 87.5 [61.6, 114.1] WA. All ten
  gender×group curves have non-overlapping CIs between the 1-result bin and their
  highest-volume plotted bin.
- **Supplementary — result_count vs. best-event proportion** (≥2-result seasons only):
  overall ρ = **−0.131** [−0.154, −0.106], *p* = 1.6×10⁻²⁸, n = 7,106. Negative and
  significant in 9/10 gender×group slices (Women Hurdles is the one exception, CI
  includes 0) — athletes who race more tend to reach their season best *proportionally
  earlier* in their own schedule, consistent with the "not monotonic" finding in §2
  above and the schedule-length caveat in §3.

New research figures under `research/plots/`: `point_jump_ci_{gender}_{group}.png`
(bootstrap CI error bars, 10 figures), `point_jump_vs_result_count_overall.png`
(pooled scatter with Spearman ρ), and `spearman_rho_point_jump_by_group.png` /
`spearman_rho_best_event_proportion_by_group.png` (forest-style ρ comparison by group).

**Caveat:** unlike `causal_analysis/`, these tests are cross-sectional (compare
different athletes and athlete-seasons, including the same athlete across years, without
an athlete fixed effect), so p-values here are optimistic. Read them as
descriptive/associative dose-response signal, not as strict inference — the
within-athlete version of the same tests lives in
[`../causal_analysis/Summary_findings.md`](../causal_analysis/Summary_findings.md).

## 3. Limitations

1. **Point jump floor artifact** — the 1-result-season = 0 point-jump rule mechanically
   inflates the appearance of an "increasing with volume" trend; use `causal_analysis/`
   for a controlled estimate.
2. **Plot bins require ≥10 athletes**, so high-volume bins (10+ meets) are often sparse
   or unplotted, and reported "trend" figures compare only the lowest vs. highest
   *plotted* bin, not the full range of observed volumes.
3. **Best-event proportion depends mechanically on schedule length** — more scheduled
   meets create more opportunity for both an early peak (lower proportion) and more
   "room" before hitting 1.0, so cross-athlete comparisons should control for meets/season
   where possible (see the "by number of results" breakdowns).
4. **Relay WA scores** can shift an individual leg athlete's "best mark" timing, since
   relay marks are pooled into their event_group panel.
5. **Associative only** — like the causal module's baseline models, everything in this
   folder is descriptive; no adjustment for confounders (athlete quality, coaching,
   injury) is made here.

## 4. Reproducibility

```bash
cd number_of_events_question
python main.py all
```

See [README.md](README.md) for full methods and folder dependencies.
