# Summary Findings — preliminary_findings (Early-Stage Snapshot)

**Status:** archival / historical. This folder captures an earlier stage of the project
before its contents were split into dedicated, actively-maintained modules. The numbers
below are preserved as they existed at that snapshot and **may differ from current
numbers** in the successor modules (e.g., steeplechase scoring was later corrected in
`new_steeplechase_data/`, and relay-inclusive RQ1B rankings were later refined in
`relays_findings/`). Always prefer the successor module for citation-grade numbers.

---

## 1. What's preserved here

### Nationals 8th-place WA rankings (`NIRCA_Nationals_Strategy/`)

An early, relay-inclusive, all-events combined 3-year-average 8th-place ranking. Men's
top 4: 800m (829.3) > 1500m (827.0) > 200m (820.3) > 100m (818.0); men's steeplechase
already reflects the corrected scoring from `new_steeplechase_data` (740.0, not the
original ~456 pre-correction value) — this file was updated after the steeplechase fix.
Women's top: 100m (814.3) > 5000m (808.5) > 200m (799.3). This snapshot is superseded by
the live versions in `relays_findings/RQ1B_Nationals/` and
`new_steeplechase_data/RQ1B_Nationals/`.

### Number of events (`Number_Of_Events/`)

Contains an early version of the point-jump-by-competition-volume plots/reports (same
underlying question as `number_of_events_question/`) plus an early causal-analysis
report (matching the one now in `causal_analysis/`: 15,128 raw athlete-season-event_group
rows → 7,106 after excluding 1-result seasons; β(result_count) ≈ 9.89 WA/race in the
global fixed-effects model). It also contains a **doubling WA buffer** analysis not
carried forward elsewhere in this exact form: for athletes with both solo and same-day
multi-event ("doubling") days, median WA is **-8.2** (sprints, n=122) and **-33.5**
(distance, n=124) lower on doubling days than solo days — i.e., a same-day second race
tends to score noticeably worse, especially in distance, motivating a "buffer" to
subtract from time-model predictions for an athlete's 2nd+ race of the day.

### Pairwise events (`Pairwise_Events/`)

Indoor and outdoor pairwise best-event histograms (PNG only, no text reports) by
discipline and gender. Superseded by the maintained pairwise outputs in
`non_relays_findings/*_Events_Counting/` (outdoor) and `indoor_analysis/RQ1A_Best_Event/`
(indoor).

### Indoor↔outdoor interplay (`Indoor_Outdoor_Interplay/`)

An early "best first event of the season" analysis: for 546 athlete-years with both an
indoor and outdoor result where outdoor-opener WA ∈ [750, 950), mean indoor first-event
WA was **782.7** vs. mean outdoor first-event WA **810.1** (Δ = +27.4) — directionally
consistent with the more developed opener-comparison analysis later built in
`indoor_analysis/Indoor_Outdoor_Interplay/` (which reports Δμ = +27.0 on a similar,
refined sample). This snapshot additionally ranks which specific event to race first
each season by gender/discipline (e.g., men's highest-mean first-indoor event among
n≥10 events was 60m Hurdles, mean WA 824.1).

### Time models (`Time_Models/`)

Early indoor and outdoor point-band feature-importance reports. Superseded by
`time_models/Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/`
(outdoor) and `indoor_analysis/Feature_Importance_Point_Band_Time_Models/` (indoor).

## 2. Why this folder still has value

1. It documents the **evolution of the analysis** — e.g., the steeplechase-scoring fix
   and the refinement of the opener-comparison sample size can be traced by diffing this
   snapshot against the current modules.
2. It preserves the **doubling WA buffer** finding, which was not carried forward as its
   own dedicated module but is referenced conceptually by the doubling analysis in
   `time_models/new_factors/doubling_analysis/` (which found a similar-direction, smaller
   -15.3 WA within-athlete effect using a different comparison design).

## 3. Limitations

1. **Stale by construction** — this is a point-in-time snapshot; any number here should
   be cross-checked against the successor module before being cited in the paper.
2. **No scripts, no reproducibility** — there is no code in this folder to regenerate
   these exact figures; only the successor modules are reproducible.
3. **Mixed correction state** — some files here already reflect later corrections (e.g.,
   steeplechase scoring in `NIRCA_Nationals_Strategy/`) while others may not — treat each
   file's numbers independently rather than assuming a single consistent data vintage
   across the whole folder.
4. **Doubling WA buffer methodology caveat** (directly from the source report): there is
   no heat/start-clock field in the CSVs, so "1st vs 2nd race of the day" is proxied by
   same-day multi-event vs. solo-day comparison within the same athlete-season — this
   cannot distinguish which race was actually run first.

## 4. Reproducibility

```bash
cd preliminary_findings
python main.py
```

No computation is performed — see [README.md](README.md) for the successor-module map.
